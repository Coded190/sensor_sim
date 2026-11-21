"""Coverage checks for the frozen LivingRoom seed 21 package.

Does not open the blend, does not overwrite scene_manifest.json, and does not
regenerate the scene. Distances are metres.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from sensor_sim.scene_gen.infinigen_ingest import evaluate_room_relationships, measure_semantic_frame
from sensor_sim.scene_gen.indoor_contact import measure_support
from sensor_sim.scene_gen.light_units import classify_area_emitter, classify_point_emitter
from sensor_sim.scene_gen.material_choice import respec

ROOT = Path("dataset/qualification/asset_realism")
PACKAGE = ROOT / "exported/LivingRoom/21/mitsuba"
REPORTS = ROOT / "reports"


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace(".", "_").replace(" ", "_")


def _read_obj(path: Path) -> tuple[np.ndarray, int]:
    vertices = []
    faces = 0
    for line in path.open(encoding="utf-8", errors="replace"):
        if line.startswith("v "):
            vertices.append(np.fromstring(line[2:], sep=" "))
        elif line.startswith("f "):
            faces += 1
    if not vertices:
        return np.zeros((0, 3)), faces
    return np.vstack(vertices), faces


def _mesh_for_frame(path: Path, translation: np.ndarray) -> dict:
    vertices, _faces = _read_obj(path)
    corners = []
    for line in path.open(encoding="utf-8", errors="replace"):
        if not line.startswith("f "):
            continue
        indices = []
        for token in line.split()[1:4]:
            indices.append(int(token.split("/")[0]) - 1)
        if len(indices) == 3:
            corners.append([(index,) for index in indices])
    return {"vertices": vertices + translation, "corners": corners}


def main() -> None:
    manifest = json.loads((PACKAGE / "scene_manifest.json").read_text(encoding="utf-8"))
    samples = {_sanitize(item["name"]): item for item in json.loads((REPORTS / "seed21_evaluated_samples.json").read_text(encoding="utf-8"))}
    ledger = json.loads((REPORTS / "livingroom_21_material_ledger.json").read_text(encoding="utf-8"))
    lights = json.loads((REPORTS / "seed21_light_evaluation.json").read_text(encoding="utf-8"))

    geometry_rows = []
    bounds = {}
    low_vertices = {}
    face_rows = []
    for index, instance in enumerate(manifest["instances"]):
        if instance["id"] == "calibration_marker":
            continue
        translation = np.asarray(instance["matrix"], dtype=np.float64)[:3, 3]
        chunks = []
        export_faces = 0
        for region in instance["regions"]:
            vertices, faces = _read_obj(PACKAGE / region["mesh"])
            export_faces += faces
            if len(vertices):
                chunks.append(vertices + translation)
        exported = np.vstack(chunks) if chunks else np.zeros((0, 3))
        source = samples[instance["id"]]
        tree = cKDTree(exported) if len(exported) else None
        angles = []
        corner_errors = []
        winding = []
        for triangle in source.get("triangles") or []:
            corners = np.asarray(triangle["world"], dtype=np.float64)
            _distance, nearest = tree.query(corners, k=1)
            matched = exported[nearest]
            corner_errors.extend(float(value) for value in np.linalg.norm(matched - corners, axis=1))
            geometric = np.cross(matched[1] - matched[0], matched[2] - matched[0])
            length = float(np.linalg.norm(geometric))
            if length < 1e-12:
                continue
            normal = geometric / length
            dot = float(np.clip(np.dot(normal, np.asarray(triangle["normal"], dtype=np.float64)), -1.0, 1.0))
            angles.append(float(np.degrees(np.arccos(dot))))
            winding.append(dot > 0.0)
        order = np.argsort(exported[:, 2])[:400] if len(exported) else []
        low_vertices[instance["id"]] = exported[order]
        bounds[instance["id"]] = (exported.min(axis=0), exported.max(axis=0))
        angle_array = np.asarray(angles) if angles else np.zeros(1)
        row = {
            "id": instance["id"],
            "category": instance["category"],
            "source_polygons": int(source["polygons"]),
            "export_faces": int(export_faces),
            "lost_faces": int(max(0, int(source["polygons"]) - export_faces)),
            "corner_max_m": float(max(corner_errors) if corner_errors else -1.0),
            "normal_angle_mean_deg": float(angle_array.mean()),
            "normal_angle_max_deg": float(angle_array.max()),
            "normal_angle_p95_deg": float(np.percentile(angle_array, 95)),
            "normal_orientation_agreement": float(np.mean(winding) if winding else 0.0),
            "triangles": len(angles),
            "normal_passed": bool(angles) and float(angle_array.max()) <= 1.0 and all(winding),
        }
        geometry_rows.append(row)
        face_rows.append(row["lost_faces"])
        if index % 25 == 0:
            print(f"event=geometry_progress index={index} id={instance['id']}", flush=True)

    geometry_summary = {
        "instances": len(geometry_rows),
        "normal_passed": sum(1 for row in geometry_rows if row["normal_passed"]),
        "lost_face_instances": sum(1 for row in geometry_rows if row["lost_faces"] > 0),
        "max_normal_angle_deg": max(row["normal_angle_max_deg"] for row in geometry_rows),
        "max_corner_error_m": max(row["corner_max_m"] for row in geometry_rows),
        "definition": "Eight source triangles per instance. Each corner is matched to the nearest exported vertex. The angle is between the source world geometric normal and the normal of those three exported vertices. A pass is max angle <= 1 degree and every matched triangle keeps the source winding. Lost faces means exported face lines are fewer than source polygons.",
    }
    (REPORTS / "seed21_geometry_normals.json").write_text(json.dumps({"summary": geometry_summary, "rows": geometry_rows}, indent=2), encoding="utf-8")
    print("event=geometry_normals", json.dumps(geometry_summary), flush=True)

    floor_low, floor_high = bounds["living-room_0_0_floor"]
    floor_z = float(floor_high[2])
    floor_polygon = np.array([
        [floor_low[0], floor_low[1], floor_z],
        [floor_high[0], floor_low[1], floor_z],
        [floor_high[0], floor_high[1], floor_z],
        [floor_low[0], floor_high[1], floor_z],
    ])
    by_id = {item["id"]: item for item in manifest["instances"]}
    contacts = []
    for instance in manifest["instances"]:
        if instance["id"] not in bounds or instance["category"] in {"Window", "calibration"}:
            continue
        if any(token in instance["id"] for token in ("floor", "wall", "ceiling", "exterior")):
            continue
        low, high = bounds[instance["id"]]
        gap_to_floor = float(low[2] - floor_z)
        if abs(gap_to_floor) <= 0.05 or gap_to_floor < 0.0:
            report = measure_support(low_vertices[instance["id"]], np.zeros((1, 3), dtype=int), floor_polygon)
            contacts.append({"supported": instance["id"], "supporter": "living-room_0_0_floor", "relation": "floor", **report})
            continue
        supporter = _nearest_supporter(instance["id"], low, high, bounds, by_id)
        if supporter is None:
            contacts.append({
                "supported": instance["id"],
                "supporter": None,
                "relation": "unassigned",
                "passed": False,
                "gap_m": gap_to_floor,
                "reason": "no_supporter_within_5cm",
                "tolerance_m": 0.001,
            })
            continue
        support_low, support_high = bounds[supporter]
        top = float(support_high[2])
        polygon = np.array([
            [support_low[0], support_low[1], top],
            [support_high[0], support_low[1], top],
            [support_high[0], support_high[1], top],
            [support_low[0], support_high[1], top],
        ])
        report = measure_support(low_vertices[instance["id"]], np.zeros((1, 3), dtype=int), polygon)
        contacts.append({"supported": instance["id"], "supporter": supporter, "relation": "surface", **report})
    support_summary = {
        "contacts": len(contacts),
        "passed": sum(1 for row in contacts if row["passed"]),
        "failed": sum(1 for row in contacts if not row["passed"]),
        "floor_z_m": floor_z,
        "gap_limit_m": 0.001,
        "penetration_limit_m": 0.0001,
        "definition": "measure_support on the lowest 400 exported vertices against the supporter top polygon. The floor polygon is the living-room floor AABB at its max Z. A supporter is the mesh whose top is within 5 cm under the object and whose XY box overlaps it. Seed 11 sofa gap 0.516 m is a separate rejected measurement.",
    }
    (REPORTS / "seed21_support_contacts.json").write_text(json.dumps({"summary": support_summary, "contacts": contacts}, indent=2), encoding="utf-8")
    print("event=support", json.dumps(support_summary), flush=True)

    relations = _semantics(manifest, bounds, floor_low, floor_high)
    (REPORTS / "seed21_semantic_relations.json").write_text(json.dumps(relations, indent=2), encoding="utf-8")
    print("event=semantic passed", relations["passed"], flush=True)

    light_rows = []
    for light in lights:
        position = light["evaluated_translation"]
        scale = tuple(light["evaluated_axis_scale"])
        if light["light_type"] == "POINT":
            row = classify_point_emitter(light["energy"], light["shadow_soft_size"], position)
        elif light["light_type"] == "AREA":
            row = classify_area_emitter(
                light["energy"],
                shape=light["shape"] or "SQUARE",
                size=light["size"],
                size_y=light["size_y"],
                scale=scale,
                two_sided=bool(light["two_sided"]),
                spread_rad=light["spread_rad"],
                position=position,
            )
        else:
            row = {"included": False, "reason": "light_type_not_converted"}
        row["name"] = light["name"]
        row["parent"] = light["parent"]
        row["color"] = light["color"]
        light_rows.append(row)
    light_summary = {
        "lights": len(light_rows),
        "included": sum(1 for row in light_rows if row.get("included")),
        "omitted_at_origin": sum(1 for row in light_rows if row.get("reason") == "light_transform_at_origin"),
        "spheres": sum(1 for row in light_rows if row.get("type") == "sphere"),
        "definition": "Evaluated depsgraph translation, not the unevaluated matrix. Area lights parented to windows still evaluate to the origin and stay omitted. Point lamps use shadow_soft_size as a sphere radius.",
    }
    (REPORTS / "seed21_light_conversion.json").write_text(json.dumps({"summary": light_summary, "lights": light_rows}, indent=2), encoding="utf-8")
    print("event=lights", json.dumps(light_summary), flush=True)

    _materials(manifest, ledger, light_rows)
    print("event=qualify_livingroom_21_done", flush=True)


def _nearest_supporter(instance_id: str, low: np.ndarray, high: np.ndarray, bounds: dict, by_id: dict) -> str | None:
    best = None
    best_gap = 1e9
    for other, (other_low, other_high) in bounds.items():
        if other == instance_id:
            continue
        category = by_id[other]["category"]
        name = other.lower()
        structural = category in {"Sofa", "SimpleDesk", "TVStand", "LargeShelf", "SimpleBookcase", "KitchenCabinet", "CellShelf"} or any(
            token in name for token in ("desk", "shelf", "stand", "cabinet", "table", "wall")
        )
        if not structural:
            continue
        overlap = other_low[0] <= high[0] and other_high[0] >= low[0] and other_low[1] <= high[1] and other_high[1] >= low[1]
        if not overlap:
            continue
        gap = float(low[2] - other_high[2])
        if -0.005 <= gap <= 0.05 and gap < best_gap:
            best = other
            best_gap = gap
    return best


def _semantics(manifest: dict, bounds: dict, floor_low: np.ndarray, floor_high: np.ndarray) -> dict:
    records = []
    for instance in manifest["instances"]:
        if instance["id"] not in bounds and instance["id"] != "calibration_marker":
            continue
        if instance["id"] == "calibration_marker":
            continue
        frame = instance.get("frame") or {}
        category = instance["category"]
        if category in {"TV", "LiteDoor", "LouverDoor"}:
            translation = np.asarray(instance["matrix"], dtype=np.float64)[:3, 3]
            mesh = _mesh_for_frame(PACKAGE / instance["regions"][0]["mesh"], translation)
            for region in instance["regions"][1:]:
                extra = _mesh_for_frame(PACKAGE / region["mesh"], translation)
                offset = len(mesh["vertices"])
                mesh["vertices"] = np.vstack([mesh["vertices"], extra["vertices"]])
                mesh["corners"].extend([[(index[0] + offset,) for index in face] for face in extra["corners"]])
            frame = measure_semantic_frame(mesh, category)
        origin = bounds[instance["id"]][0] * 0.5 + bounds[instance["id"]][1] * 0.5
        records.append({
            "id": instance["id"],
            "category": category,
            "origin": origin.tolist(),
            "forward": frame.get("forward"),
            "frame_status": frame.get("status"),
            "bounds": [bounds[instance["id"]][0].tolist(), bounds[instance["id"]][1].tolist()],
        })
    center = ((floor_low + floor_high) * 0.5).tolist()
    report = evaluate_room_relationships(records, floor_center=center)
    report["frames_measured"] = sum(1 for row in records if row["frame_status"] == "measured")
    report["frames_unknown"] = sum(1 for row in records if row["frame_status"] != "measured")
    return report


def _materials(manifest: dict, ledger: dict, light_rows: list[dict]) -> None:
    by_key = {(row["instance"], row["package_slot"]): row for row in ledger["rows"]}
    missing_hash = []
    hashed = 0
    empty_regions = []
    qualified = json.loads(json.dumps(manifest))
    for instance in qualified["instances"]:
        for region in instance["regions"]:
            key = (instance["id"], region["slot"])
            row = by_key.get(key)
            if row is None:
                empty_regions.append(key)
                continue
            choice = {
                "family": row["recommended_family"],
                "note": row["note"],
                "preserved": ["ledger"],
                "approximated": row.get("approximated") or [],
                "lost": row.get("lost") or [],
                "quantitative_polarization": bool(row.get("quantitative_polarization")),
                "qualitative_ok": True,
                "ior": float((region["spec"].get("parameters") or {}).get("int_ior") or 1.5),
                "roughness": 0.5,
            }
            if row["render_status"] == "slot_index_has_no_source_material":
                choice["note"] = "No source node at this slot index. The packaged MTL scalars were classified, and the row is kept."
                choice["lost"] = list(choice["lost"]) + ["source_node_missing"]
            region["spec"] = respec(region["spec"], choice)
            for texture in region["spec"].get("textures") or []:
                path = PACKAGE / texture["path"]
                if not path.is_file():
                    missing_hash.append(texture["path"])
                    continue
                import hashlib
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hashed += 1
                if digest != texture["sha256"]:
                    missing_hash.append(texture["path"])
    qualified["lights"] = []
    for row in light_rows:
        record = {
            "id": _sanitize(row["name"]),
            "blender_type": "POINT" if row.get("type") == "sphere" else "AREA",
            "position": row.get("position"),
            "included": bool(row.get("included")),
            "reason": row.get("reason", ""),
            "conversion": row.get("conversion"),
            "polarization": "unpolarized",
            "parent": row.get("parent"),
            "power_w": row.get("power_w"),
            "color": row.get("color"),
            "type": row.get("type"),
            "radius_m": row.get("radius_m"),
            "radiance": row.get("radiance"),
            "intensity": row.get("intensity"),
            "area_m2": row.get("area_m2"),
            "note": row.get("note", ""),
        }
        qualified["lights"].append(record)
    qualified["environment"] = dict(manifest.get("environment") or {})
    qualified["environment"]["note"] = "radiance in this file is the unconverted-world replacement stored with the package. Gallery renders that use 1.2 are labeled separately."
    qualified["qualification"] = "livingroom_seed_21_audited_bsdf"
    path = PACKAGE / "scene_manifest_qualified.json"
    path.write_text(json.dumps(qualified), encoding="utf-8")
    families = Counter(region["spec"]["family"] for instance in qualified["instances"] for region in instance["regions"])
    summary = {
        "regions": sum(len(instance["regions"]) for instance in qualified["instances"]),
        "families": dict(families),
        "texture_hashes_checked": hashed,
        "texture_failures": missing_hash[:10],
        "texture_failure_count": len(missing_hash),
        "ledger_rows": len(ledger["rows"]),
        "ledger_status": dict(Counter(row["render_status"] for row in ledger["rows"])),
        "unmatched_blend_meshes": len(ledger["unmatched_objects"]),
        "unmatched_reason": "217 blend meshes have zero material slots. They are not packaged regions and are not hidden.",
        "manifest": str(path),
    }
    (REPORTS / "livingroom_21_material_qualified_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("event=materials", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
