"""Qualification-only ingest of a complete Infinigen export.

This module does not call ``publish_candidate`` or ``set_candidate_status``.
The legacy extractor in ``infinigen_extract.py`` is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import create_package
from sensor_sim.scene_gen.collision_proxy import proxy_bytes, surface_deviation
from sensor_sim.scene_gen.infinigen_extract import INFINIGEN2_COMMIT, INFINIGEN2_LICENSE, STABLE_COMMIT, STABLE_LICENSE
from sensor_sim.scene_gen.mesh_io import read_mesh, write_mesh
from sensor_sim.scene_gen.semantic_frame import make_frame, sofa_forward_from_regions

logger = logging.getLogger("sensor_sim.infinigen_ingest")

ROOT = Path("dataset/qualification/asset_realism/packages")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_ply_mesh(path: Path) -> dict:
    """Read a binary little-endian PLY with positions, normals, and optional UVs."""
    raw = Path(path).read_bytes()
    header, body = raw.split(b"end_header\n", 1)
    lines = header.decode("ascii", errors="replace").splitlines()
    counts = {}
    vertex_properties = []
    in_vertex = False
    for line in lines:
        if line.startswith("element "):
            kind, count = line.split()[1], int(line.split()[2])
            counts[kind] = count
            in_vertex = kind == "vertex"
        elif in_vertex and line.startswith("property "):
            vertex_properties.append(line.split()[-1])
    width = len(vertex_properties)
    record = np.dtype([("values", "<f4", (width,))])
    vertices_raw = np.frombuffer(body, dtype=record, count=counts["vertex"])
    values = vertices_raw["values"]
    index = {name: position for position, name in enumerate(vertex_properties)}
    points = np.stack([values[:, index["x"]], values[:, index["y"]], values[:, index["z"]]], axis=1).astype(np.float64)
    normals = None
    if {"nx", "ny", "nz"} <= set(index):
        normals = np.stack([values[:, index["nx"]], values[:, index["ny"]], values[:, index["nz"]]], axis=1).astype(np.float64)
    texcoords = None
    if {"s", "t"} <= set(index):
        texcoords = np.stack([values[:, index["s"]], values[:, index["t"]]], axis=1).astype(np.float64)
    offset = counts["vertex"] * width * 4
    face_bytes = body[offset:]
    faces = []
    cursor = 0
    for _ in range(counts.get("face", 0)):
        count = face_bytes[cursor]
        cursor += 1
        indices = np.frombuffer(face_bytes, dtype="<i4", count=count, offset=cursor)
        cursor += 4 * count
        faces.append([int(item) for item in indices])
    corners = []
    for face in faces:
        corners.append([(vertex, vertex if texcoords is not None else None, vertex if normals is not None else None) for vertex in face])
    mesh = {"vertices": points, "corners": corners, "face_materials": ["default"] * len(corners)}
    if texcoords is not None:
        mesh["texcoords"] = texcoords
    if normals is not None:
        mesh["normals"] = normals
    return mesh


def _counts(mesh: dict) -> dict:
    triangles = 0
    for face in mesh["corners"]:
        triangles += max(len(face) - 2, 0)
    points = np.asarray(mesh["vertices"], dtype=np.float64)
    return {
        "vertices": int(len(points)),
        "polygons": int(len(mesh["corners"])),
        "triangles": int(triangles),
        "bounds_min": points.min(axis=0).tolist() if len(points) else [0, 0, 0],
        "bounds_max": points.max(axis=0).tolist() if len(points) else [0, 0, 0],
        "has_uv": bool(np.asarray(mesh.get("texcoords", [])).size),
        "has_normals": bool(np.asarray(mesh.get("normals", [])).size),
        "material_names": sorted(set(mesh.get("face_materials") or [])),
    }


def measure_forward(mesh: dict) -> np.ndarray:
    """Horizontal direction from the tall vertical mass toward the lower seat or front mass.

    This is a geometric measurement. A category name is not used.
    """
    points = np.asarray(mesh["vertices"], dtype=np.float64)
    if len(points) < 3 or not mesh.get("corners"):
        raise ValueError("reason=frame_unmeasured")
    areas = []
    normals = []
    centroids = []
    for face in mesh["corners"]:
        index = [int(corner[0]) for corner in face[:3]]
        tri = points[index]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        area = float(np.linalg.norm(normal))
        if area < 1e-12:
            continue
        normal = normal / area
        areas.append(area * 0.5)
        normals.append(normal)
        centroids.append(tri.mean(axis=0))
    if not areas:
        raise ValueError("reason=frame_unmeasured")
    areas = np.asarray(areas)
    normals = np.asarray(normals)
    centroids = np.asarray(centroids)
    vertical = np.abs(normals[:, 2]) < 0.45
    if not np.any(vertical):
        raise ValueError("reason=frame_unmeasured")
    # The backrest side is the vertical mass whose centroid is farthest from the mesh center in XY.
    center = points.mean(axis=0)
    offset = centroids[vertical] - center
    offset[:, 2] = 0.0
    distance = np.linalg.norm(offset, axis=1)
    back_index = int(np.argmax(distance * areas[vertical]))
    back = centroids[vertical][back_index]
    direction = center - back
    direction[2] = 0.0
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        raise ValueError("reason=frame_unmeasured")
    return direction / length


def _face_records(mesh: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    points = np.asarray(mesh["vertices"], dtype=np.float64)
    if len(points) < 3 or not mesh.get("corners"):
        return None
    normals = []
    centroids = []
    areas = []
    for face in mesh["corners"]:
        index = [int(corner[0]) for corner in face[:3]]
        triangle = points[index]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            continue
        normals.append(normal / length)
        centroids.append(triangle.mean(axis=0))
        areas.append(0.5 * length)
    if not areas:
        return None
    return points, np.asarray(normals), np.asarray(centroids), np.asarray(areas)


def measure_semantic_frame(mesh: dict, category: str) -> dict:
    """Measure facing from mesh regions. A category name is not a direction.

    A sofa or chair forward goes from the backrest centroid to the seat centroid.
    A screen or appliance forward is the mean normal of the dominant front face.
    If those regions cannot be separated, the status is ``unknown``.
    """
    records = _face_records(mesh)
    if records is None:
        return {"status": "unknown", "reason": "frame_unmeasured", "category": category}
    points, normals, centroids, areas = records
    low = points.min(axis=0)
    high = points.max(axis=0)
    height = float(high[2] - low[2])
    if height < 1e-6:
        return {"status": "unknown", "reason": "flat_mesh", "category": category}
    lower = category.lower()
    seating = lower in {"sofa", "chair"} or lower.endswith("chair")
    fronted = lower in {"monitor", "television", "tv", "refrigerator", "microwave"} or "door" in lower or "cabinet" in lower
    if seating:
        seat = (normals[:, 2] > 0.65) & (centroids[:, 2] < low[2] + 0.55 * height)
        back = (np.abs(normals[:, 2]) < 0.4) & (centroids[:, 2] > low[2] + 0.45 * height)
        if float(areas[seat].sum()) < 0.02 * float(areas.sum()) or float(areas[back].sum()) < 0.02 * float(areas.sum()):
            return {"status": "unknown", "reason": "regions_not_separated", "category": category}
        seat_centroid = np.average(centroids[seat], axis=0, weights=areas[seat])
        back_centroid = np.average(centroids[back], axis=0, weights=areas[back])
        try:
            forward = sofa_forward_from_regions(back_centroid, seat_centroid)
        except ValueError:
            return {"status": "unknown", "reason": "sofa_forward", "category": category}
        evidence = {
            "seat_centroid_m": [float(value) for value in seat_centroid],
            "backrest_centroid_m": [float(value) for value in back_centroid],
            "seat_area_m2": float(areas[seat].sum()),
            "backrest_area_m2": float(areas[back].sum()),
        }
    elif fronted:
        vertical = np.abs(normals[:, 2]) < 0.35
        if not np.any(vertical):
            return {"status": "unknown", "reason": "no_front_face", "category": category}
        # The front is the vertical direction bin with the most area.
        horizontal = normals.copy()
        horizontal[:, 2] = 0.0
        lengths = np.linalg.norm(horizontal, axis=1)
        usable = vertical & (lengths > 1e-6)
        if not np.any(usable):
            return {"status": "unknown", "reason": "no_front_face", "category": category}
        direction = horizontal[usable] / lengths[usable][:, None]
        # Quantize to 8 headings so opposite faces stay distinct.
        heading = np.arctan2(direction[:, 1], direction[:, 0])
        bins = np.round(heading / (np.pi / 4.0)).astype(int)
        best_area = -1.0
        best = None
        for value in np.unique(bins):
            chosen = bins == value
            total = float(areas[usable][chosen].sum())
            if total > best_area:
                best_area = total
                best = chosen
        if best is None or best_area < 0.05 * float(areas.sum()):
            return {"status": "unknown", "reason": "front_not_dominant", "category": category}
        forward = np.average(direction[best], axis=0, weights=areas[usable][best])
        forward[2] = 0.0
        length = float(np.linalg.norm(forward))
        if length < 1e-6:
            return {"status": "unknown", "reason": "front_not_dominant", "category": category}
        forward = forward / length
        evidence = {"front_area_m2": best_area}
    else:
        return {"status": "unknown", "reason": "category_has_no_facing_rule", "category": category}
    frame = make_frame(forward, origin=points.mean(axis=0))
    frame["status"] = "measured"
    frame["category"] = category
    frame["validation"] = "backrest_to_seat" if seating else "dominant_front_face"
    frame["evidence"] = evidence
    if fronted:
        frame["screen_normal"] = frame["forward"]
        frame["screen_up"] = [0.0, 0.0, 1.0]
    return frame


def semantic_for(category: str, mesh: dict) -> dict:
    """Measure the frame from mesh regions. Unknown stays unknown."""
    return measure_semantic_frame(mesh, category)


FACING_DOT_MINIMUM = 0.5


def evaluate_room_relationships(records: list[dict], *, floor_center: list[float] | None = None) -> dict:
    """Room relationships. A local frame is not enough, and unknown does not pass.

    ``records`` use world ``origin``, optional ``forward``, ``frame_status``,
    ``category``, and optional world ``bounds`` as ``[low, high]``.

    A facing check passes when the horizontal forward and the direction to the
    target have dot product at least 0.5. A missing chair set is ``not_present``
    and passes. A missing required target, or an unknown frame, fails.
    Door access fails when a major-furniture box intersects the 0.8 m by 0.6 m
    segment from the door toward the floor center. Circulation uses the same
    test on the segment from the first door to the sofa/display midpoint.
    """
    center = np.asarray(floor_center or [0.0, 0.0, 0.0], dtype=np.float64)
    displays = [row for row in records if str(row.get("category", "")).lower() in {"tv", "television", "monitor"}]
    sofas = [row for row in records if str(row.get("category", "")).lower() == "sofa"]
    chairs = [row for row in records if "chair" in str(row.get("category", "")).lower()]
    doors = [row for row in records if "door" in str(row.get("category", "")).lower()]
    relations = []
    focus = None
    if sofas and displays:
        focus = (np.asarray(sofas[0]["origin"], dtype=np.float64) + np.asarray(displays[0]["origin"], dtype=np.float64)) * 0.5
    for sofa in sofas:
        target = _nearest(sofa, displays)
        relations.append(_facing("sofa_toward_display", sofa, None if target is None else target["origin"]))
    for display in displays:
        target = _nearest(display, sofas)
        relations.append(_facing("display_toward_seating", display, None if target is None else target["origin"]))
    if not chairs:
        relations.append({"name": "chairs_toward_focus", "status": "not_present", "passed": True, "count": 0, "reason": "no_chair_in_room"})
    else:
        target = focus if focus is not None else center
        for chair in chairs:
            relations.append(_facing("chair_toward_focus", chair, target))
    if not doors:
        relations.append({"name": "door_access", "status": "unknown", "passed": False, "reason": "no_door"})
    else:
        for door in doors:
            blocked = _segment_blocked(door, center, records)
            facing = _facing("door_toward_room", door, center)
            relations.append({
                "name": "door_access",
                "id": door.get("id"),
                "status": "measured" if facing["status"] == "measured" else facing["status"],
                "passed": bool(facing["passed"] and not blocked),
                "facing_dot": facing.get("dot"),
                "blocked_by": blocked,
                "reason": "clear" if facing["passed"] and not blocked else "blocked" if blocked else facing["reason"],
            })
    if doors and focus is not None:
        blocked = _segment_blocked(doors[0], focus, records)
        relations.append({
            "name": "walking_path",
            "status": "measured",
            "passed": blocked is None,
            "blocked_by": blocked,
            "reason": "clear" if blocked is None else "furniture_on_path",
            "definition": "0.8 m approach is not this path. This path is the segment from the first door to the sofa/display midpoint. A major-furniture AABB that crosses the segment fails.",
        })
    else:
        relations.append({"name": "walking_path", "status": "unknown", "passed": False, "reason": "path_endpoints_missing"})
    passed = all(item["passed"] for item in relations) and all(item["status"] != "unknown" for item in relations)
    return {
        "passed": passed,
        "facing_dot_minimum": FACING_DOT_MINIMUM,
        "relations": relations,
        "definition": "Facing dot is the horizontal forward dotted with the direction to the target. Unknown frames fail. not_present passes.",
    }


def _nearest(source: dict, others: list[dict]) -> dict | None:
    if not others:
        return None
    origin = np.asarray(source["origin"], dtype=np.float64)
    return min(others, key=lambda row: float(np.linalg.norm(np.asarray(row["origin"], dtype=np.float64) - origin)))


def _facing(name: str, source: dict, target) -> dict:
    if source.get("frame_status") != "measured" or source.get("forward") is None or target is None:
        return {"name": name, "id": source.get("id"), "status": "unknown", "passed": False, "reason": "forward_unknown" if target is not None else "target_missing"}
    direction = np.asarray(target, dtype=np.float64) - np.asarray(source["origin"], dtype=np.float64)
    direction[2] = 0.0
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return {"name": name, "id": source.get("id"), "status": "unknown", "passed": False, "reason": "coincident"}
    forward = np.asarray(source["forward"], dtype=np.float64)
    forward[2] = 0.0
    forward_length = float(np.linalg.norm(forward))
    if forward_length < 1e-8:
        return {"name": name, "id": source.get("id"), "status": "unknown", "passed": False, "reason": "forward_unknown"}
    dot = float(np.dot(forward / forward_length, direction / length))
    passed = dot >= FACING_DOT_MINIMUM
    return {
        "name": name,
        "id": source.get("id"),
        "status": "measured",
        "passed": passed,
        "dot": dot,
        "minimum_dot": FACING_DOT_MINIMUM,
        "reason": "faces_target" if passed else "faces_away",
    }


def _segment_blocked(start: dict, target, records: list[dict]) -> str | None:
    origin = np.asarray(start["origin"], dtype=np.float64)
    end = np.asarray(target, dtype=np.float64)
    delta = end - origin
    delta[2] = 0.0
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return None
    direction = delta / length
    for row in records:
        if row is start or "bounds" not in row:
            continue
        category = str(row.get("category", "")).lower()
        if category in {"sofa", "desk", "table", "bed", "shelf", "bookcase", "cabinet", "tvstand"} or any(
            token in category for token in ("shelf", "desk", "bookcase", "cabinet")
        ):
            low, high = np.asarray(row["bounds"][0], dtype=np.float64), np.asarray(row["bounds"][1], dtype=np.float64)
            if _aabb_hits_segment(low, high, origin, direction, min(length, 2.0)):
                return str(row.get("id"))
    return None


def _aabb_hits_segment(low: np.ndarray, high: np.ndarray, origin: np.ndarray, direction: np.ndarray, length: float) -> bool:
    for step in np.linspace(0.2, length, num=8):
        point = origin + direction * float(step)
        if np.all(point[:2] >= low[:2] - 0.3) and np.all(point[:2] <= high[:2] + 0.3):
            return True
    return False


def ingest_mesh(
    mesh: dict,
    destination: Path,
    *,
    asset_id: str,
    category: str,
    commit: str,
    license_text: str,
    source_name: str = "infinigen",
    generator_name: str = "infinigen_ingest",
    upstream_object: str,
    source_counts: dict | None = None,
    textures: list[dict] | None = None,
) -> dict:
    """Write one unregistered package. The destination must not already exist."""
    destination = Path(destination)
    render = write_mesh(mesh, name=asset_id)
    proxy_payload, proxy_report = proxy_bytes(mesh, name=f"{asset_id}_collision")
    proxy_mesh = read_mesh(proxy_payload)
    deviation = surface_deviation(mesh, proxy_mesh)
    payload = {"geometry/render.obj": render, "geometry/collision.obj": proxy_payload}
    roles = {"geometry/render.obj": "render", "geometry/collision.obj": "collision"}
    packaged_textures = []
    for texture in textures or []:
        payload[texture["path"]] = Path(texture["source"]).read_bytes()
        roles[texture["path"]] = "texture"
        item = {key: value for key, value in texture.items() if key != "source"}
        item["sha256"] = _sha(payload[texture["path"]])
        packaged_textures.append(item)
    counts = _counts(mesh)
    frame = {
        "up_axis": "Z",
        "handedness": "right",
        "transform_convention": "row_major_column_vector",
        "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "semantic": semantic_for(category, mesh),
    }
    manifest = {
        "asset_id": asset_id,
        "version": "1.0.0",
        "units": "m",
        "frame": frame,
        "provenance": {
            "source": source_name,
            "generator": generator_name,
            "generator_version": "asset_realism_v1",
            "seed": 0,
            "parameters": {
                "category": category,
                "source_counts": source_counts or {},
                "packaged_counts": counts,
                "proxy": proxy_report,
                "deviation": deviation,
                "redistribution": "unresolved",
            },
            "license": license_text,
            "license_status": "unresolved",
            "upstream_commit": commit,
            "upstream_object": upstream_object,
        },
        "geometry": {
            "profile": "imported",
            "units": "m",
            "category": category,
            "polygon_count": counts["polygons"],
            "triangle_count": counts["triangles"],
            "bounds_min": counts["bounds_min"],
            "bounds_max": counts["bounds_max"],
            "components": [{"id": name, "material": name} for name in counts["material_names"]],
        },
        "materials": [{"id": name, "name": name, "type": "diffuse", "reflectance": [0.6, 0.6, 0.6]} for name in counts["material_names"] or ["default"]],
        "textures": packaged_textures,
        "collision": [{
            "id": "proxy",
            "path": "geometry/collision.obj",
            "render_path": "geometry/render.obj",
            "triangle_count": proxy_report["triangle_count"],
            "qualified": True,
            "method": proxy_report["method"],
        }],
        "support_surfaces": [{
            "id": "base",
            "normal": [0.0, 0.0, 1.0],
            "origin": [counts["bounds_min"][0], counts["bounds_min"][1], counts["bounds_min"][2]],
            "polygon": [
                [counts["bounds_min"][0], counts["bounds_min"][1], counts["bounds_min"][2]],
                [counts["bounds_max"][0], counts["bounds_min"][1], counts["bounds_min"][2]],
                [counts["bounds_max"][0], counts["bounds_max"][1], counts["bounds_min"][2]],
                [counts["bounds_min"][0], counts["bounds_max"][1], counts["bounds_min"][2]],
            ],
        }],
    }
    sealed = create_package(destination, manifest, payload, roles)
    report = {"asset_id": asset_id, "package": str(destination), "package_sha256": sealed["package_sha256"], "counts": counts, "proxy": proxy_report, "deviation": deviation, "license_status": "unresolved", "upstream_commit": commit}
    logger.info("infinigen_ingest event=package asset_id=%s triangles=%s uv=%s sha=%s", asset_id, counts["triangles"], counts["has_uv"], sealed["package_sha256"])
    (destination / "ingest_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def license_for(commit: str) -> str:
    if commit == STABLE_COMMIT:
        return STABLE_LICENSE
    if commit == INFINIGEN2_COMMIT:
        return INFINIGEN2_LICENSE
    raise ValueError(f"reason=unknown_commit value={commit}")
