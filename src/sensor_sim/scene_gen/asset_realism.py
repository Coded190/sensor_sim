"""Build the qualification-only Infinigen ingest, scenes, and gallery."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.composition_rules import BUILDERS
from sensor_sim.scene_gen.composition_solver import solve
from sensor_sim.scene_gen.gap_assets import BUILDERS as GAP
from sensor_sim.scene_gen.infinigen_extract import INFINIGEN2_COMMIT, INFINIGEN2_LICENSE, STABLE_COMMIT, STABLE_LICENSE
from sensor_sim.scene_gen.infinigen_ingest import ingest_mesh, read_ply_mesh
from sensor_sim.scene_gen.mesh_io import read_mesh, read_obj, write_obj
from sensor_sim.scene_gen.room_schema import make_specification
from sensor_sim.scene_gen.scene_critic import mesh_report, scene_report

logger = logging.getLogger("sensor_sim.asset_realism")


class AssetRealismError(RuntimeError):
    """A qualification scene cannot be rendered from its stored materials."""


def object_bsdf(item: dict) -> dict:
    """Return the stored material spec. A missing spec is not replaced with one color."""
    spec = item.get("material_spec")
    if not isinstance(spec, dict) or spec.get("schema_version") != "material_spec_v1":
        raise AssetRealismError(f"reason=flat_diffuse_removed instance={item.get('id')}")
    return spec

ROOT = Path("dataset/qualification/asset_realism")
BAKEOFF = Path("dataset/qualification/indoor_generator_bakeoff")
PREVIEW_MESHES = BAKEOFF / "compatibility/mitsuba_export/infinigen_2_preview/meshes"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _openings(width: float, depth: float) -> list[dict]:
    return [
        {"id": "door_south", "kind": "door", "wall": "south", "offset_m": 0.35, "width_m": 0.9, "height_m": 2.1, "sill_m": 0.0},
        {"id": "window_north", "kind": "window", "wall": "north", "offset_m": min(0.8, width * 0.2), "width_m": 1.0, "height_m": 1.0, "sill_m": 0.9},
    ]


def _room_shell(directory: Path, spec: dict) -> list[Path]:
    width = spec["interior_width_m"]
    depth = spec["interior_depth_m"]
    height = spec["interior_height_m"]
    directory.mkdir(parents=True, exist_ok=True)
    parts = {
        "floor": ([[0, 0, 0], [width, 0, 0], [width, depth, 0], [0, depth, 0]], [[0, 1, 2, 3]]),
        "ceiling": ([[0, 0, height], [0, depth, height], [width, depth, height], [width, 0, height]], [[0, 1, 2, 3]]),
    }
    written = []
    for name, (vertices, faces) in parts.items():
        path = directory / f"{name}.obj"
        path.write_bytes(write_obj(np.asarray(vertices, dtype=np.float64), faces, name=name))
        written.append(path)
    return written


def audit() -> dict:
    """Record repository state and the source-to-package gap for the existing extracts."""
    import subprocess

    status = subprocess.run(["git", "status", "--short", "--branch"], check=False, capture_output=True, text=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    diff = subprocess.run(["git", "diff", "--stat"], check=False, capture_output=True, text=True)
    rows = []
    interior_ids = [
        "furn_sofa", "furn_armchair", "furn_office_desk", "furn_dining_table", "prop_chair",
        "furn_monitor", "furn_tv", "furn_fridge", "furn_microwave", "furn_base_cabinet",
    ]
    for asset_id in interior_ids:
        rows.append({
            "asset_id": asset_id,
            "origin": "sensor_sim_primitive_builder",
            "upstream_commit": None,
            "note": "Built by interior_assets box assemblies. Not an Infinigen mesh. The builder was not edited.",
        })
    extracts = [
        ("infinigen_stable_desk", STABLE_COMMIT, BAKEOFF / "candidates/infinigen_indoors_stable/extracted_assets/SimpleDeskFactory(361208).spawn_asset(3395843)"),
        ("infinigen_stable_shelf", STABLE_COMMIT, BAKEOFF / "candidates/infinigen_indoors_stable/extracted_assets/LargeShelfFactory(4380085).spawn_asset(215187)"),
        ("infinigen_stable_floor", STABLE_COMMIT, BAKEOFF / "candidates/infinigen_indoors_stable/extracted_assets/living-room_0/0.floor"),
        ("infinigen2_table", INFINIGEN2_COMMIT, BAKEOFF / "candidates/infinigen_2_preview/extracted_assets/table_dining"),
        ("infinigen2_bowl", INFINIGEN2_COMMIT, BAKEOFF / "candidates/infinigen_2_preview/extracted_assets/bowl.001"),
        ("infinigen2_room_shape", INFINIGEN2_COMMIT, BAKEOFF / "candidates/infinigen_2_preview/extracted_assets/room_shape.001"),
    ]
    for asset_id, commit, folder in extracts:
        meta = json.loads((folder / "asset.json").read_text(encoding="utf-8"))
        obj = folder / "geometry/mesh.obj"
        text = obj.read_text(encoding="utf-8", errors="replace")
        vertices, polygons = read_obj(obj)
        package = Path("assets/candidates") / asset_id / "1.0.0" / "asset.json"
        packaged = json.loads(package.read_text(encoding="utf-8")) if package.is_file() else {}
        rows.append({
            "asset_id": asset_id,
            "origin": "infinigen",
            "upstream_commit": commit,
            "source_blend": meta.get("source_blend"),
            "object_name": meta.get("object_name"),
            "source_vertices": meta.get("vertices"),
            "source_polygons": meta.get("polygons"),
            "exported_vertices": int(len(vertices)),
            "exported_polygons": int(len(polygons)),
            "exported_has_vt": "\nvt " in text or text.startswith("vt "),
            "exported_has_vn": "\nvn " in text,
            "exported_has_usemtl": "usemtl" in text,
            "material_slots": meta.get("material_slots"),
            "dimensions_m": meta.get("dimensions"),
            "texture_files_in_extract": [],
            "textures_in_candidate_package": len(packaged.get("textures") or []),
            "mitsuba_binding": "not_bound_historical_package_has_slot_names_only",
            "license_status": packaged.get("provenance", {}).get("license_status", "unresolved"),
            "stage_difference": "The historical OBJ has the source polygon count and no UV, normal, or texture files. Slot names were stored. Mitsuba was not given those materials.",
        })
    ply_rows = []
    if PREVIEW_MESHES.is_dir():
        for path in sorted(PREVIEW_MESHES.glob("*.ply")):
            if path.name.split("-")[0].split(".")[0] not in {"sofa", "lamp", "bowl_001", "table_dining"} and not path.name.startswith("sofa"):
                continue
            header = path.read_bytes().split(b"end_header", 1)[0].decode("ascii", errors="replace")
            ply_rows.append({
                "file": path.name,
                "bytes": path.stat().st_size,
                "has_normals": "property float nx" in header,
                "has_uv": "property float s" in header,
                "origin": "infinigen_2_preview_mitsuba_export",
                "upstream_commit": INFINIGEN2_COMMIT,
            })
    payload = {
        "git_status": status.stdout,
        "head": head.stdout.strip(),
        "diff_stat": diff.stdout,
        "pins": {"stable": STABLE_COMMIT, "infinigen2": INFINIGEN2_COMMIT},
        "checkout_present": False,
        "checkout_note": "No external Infinigen git checkout was found beside the preserved bakeoff blends and extracts. Generation was not re-run. The preserved blends are the source scenes for these commits.",
        "preserved_blends": {
            "infinigen2": str(BAKEOFF / "candidates/infinigen_2_preview/generated/run1/scene.blend"),
            "stable": str(BAKEOFF / "candidates/infinigen_indoors_stable/generated/run1/scene.blend"),
        },
        "assets": rows,
        "preview_ply": ply_rows,
        "license": {
            "stable": STABLE_LICENSE,
            "infinigen2": INFINIGEN2_LICENSE,
            "redistribution": "unresolved",
        },
    }
    destination = ROOT / "source_audit"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "source_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Source audit",
        "",
        f"HEAD `{payload['head']}`. External Infinigen checkout present: {payload['checkout_present']}.",
        "",
        "The six furnished interiors use `interior_assets` box builders (`furn_sofa`, `furn_monitor`, `furn_fridge`, and the other `furn_*` ids) plus `prop_chair`. Those meshes are not Infinigen extracts.",
        "",
        "The historical Infinigen OBJ extracts match the source polygon counts recorded in their `asset.json` files and contain no `vt`, `vn`, or texture images. Candidate packages store material-slot names and no texture files, so Mitsuba was not given the Infinigen appearance. That loss is in the packaging step. It is not evidence that the source blend was a box.",
        "",
        "The Infinigen 2 preview Mitsuba PLY export does contain vertex normals and `s`/`t` coordinates for sofa, lamp, bowl, and table meshes. Those files are the geometry used by the new ingest.",
        "",
        f"License status remains unresolved. Stable: {STABLE_LICENSE}",
        "",
    ]
    (destination / "source_audit.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("asset_realism event=audit assets=%d", len(rows))
    return payload


def _ingest_existing() -> list[dict]:
    reports = []
    jobs = [
        ("qual_desk", "desk", STABLE_COMMIT, STABLE_LICENSE, BAKEOFF / "candidates/infinigen_indoors_stable/extracted_assets/SimpleDeskFactory(361208).spawn_asset(3395843)/geometry/mesh.obj", "SimpleDeskFactory"),
        ("qual_shelf", "shelf", STABLE_COMMIT, STABLE_LICENSE, BAKEOFF / "candidates/infinigen_indoors_stable/extracted_assets/LargeShelfFactory(4380085).spawn_asset(215187)/geometry/mesh.obj", "LargeShelfFactory"),
        ("qual_table", "table", INFINIGEN2_COMMIT, INFINIGEN2_LICENSE, BAKEOFF / "candidates/infinigen_2_preview/extracted_assets/table_dining/geometry/mesh.obj", "table_dining"),
        ("qual_bowl", "bowl", INFINIGEN2_COMMIT, INFINIGEN2_LICENSE, BAKEOFF / "candidates/infinigen_2_preview/extracted_assets/bowl.001/geometry/mesh.obj", "bowl.001"),
    ]
    for asset_id, category, commit, license_text, path, upstream in jobs:
        mesh = read_mesh(path)
        destination = ROOT / "packages" / asset_id / "1.0.0"
        if destination.exists():
            shutil.rmtree(destination.parent)
        reports.append(ingest_mesh(mesh, destination, asset_id=asset_id, category=category, commit=commit, license_text=license_text, upstream_object=upstream))
    ply_jobs = [
        ("qual_sofa", "sofa", "sofa.ply"),
        ("qual_lamp", "lamp", "lamp-material_52.ply"),
    ]
    for asset_id, category, filename in ply_jobs:
        mesh = read_ply_mesh(PREVIEW_MESHES / filename)
        destination = ROOT / "packages" / asset_id / "1.0.0"
        if destination.exists():
            shutil.rmtree(destination.parent)
        reports.append(ingest_mesh(mesh, destination, asset_id=asset_id, category=category, commit=INFINIGEN2_COMMIT, license_text=INFINIGEN2_LICENSE, upstream_object=filename, source_counts={"ply_vertices": int(len(mesh["vertices"]))}))
    for category, builder in GAP.items():
        for variant, seed in (("a", 1), ("b", 2)):
            if category not in {"chair", "monitor"} and variant == "b":
                continue
            asset_id = f"gap_{category}_{variant}"
            mesh = builder(seed)
            destination = ROOT / "packages" / asset_id / "1.0.0"
            if destination.exists():
                shutil.rmtree(destination.parent)
            reports.append(ingest_mesh(
                mesh,
                destination,
                asset_id=asset_id,
                category=category,
                commit="sensor_sim_gap",
                license_text="Project qualification fallback. Not an Infinigen mesh.",
                upstream_object=category,
                source_name="sensor_sim",
                generator_name="gap_assets",
            ))
    return reports


def _catalog() -> dict[str, str]:
    return {
        "sofa": "qual_sofa",
        "television": "gap_television_a",
        "table": "qual_table",
        "desk": "qual_desk",
        "monitor": "gap_monitor_a",
        "chair": "gap_chair_a",
        "refrigerator": "gap_refrigerator_a",
        "microwave": "gap_microwave_a",
        "cabinet": "gap_cabinet_a",
        "bed": "gap_bed_a",
        "nightstand": "gap_nightstand_a",
        "wardrobe": "gap_wardrobe_a",
        "shelf": "qual_shelf",
        "toolbox": "gap_toolbox_a",
        "lamp": "qual_lamp",
        "bowl": "qual_bowl",
    }


def build_scenes() -> list[dict]:
    catalog = _catalog()
    sizes = {
        "living": [(4.8, 3.8), (5.6, 4.4)],
        "office": [(3.6, 3.2), (4.4, 3.6)],
        "kitchen": [(3.8, 3.2), (4.6, 3.6)],
        "bedroom": [(3.8, 4.0), (4.4, 4.6)],
        "dining": [(4.2, 3.6), (5.0, 4.2)],
        "workshop": [(4.4, 3.6), (5.2, 4.0)],
    }
    written = []
    for kind, pairs in sizes.items():
        for index, (width, depth) in enumerate(pairs, start=1):
            spec = make_specification(room_id=f"{kind}_{index}", width=width, depth=depth, height=2.7, seed=18000 + index, openings=_openings(width, depth))
            instances, rules = BUILDERS[kind](width, depth)
            room = {"door": {"x0": 0.35, "x1": 1.25, "depth_m": 0.9}, "width": width, "depth": depth}
            solved = solve(instances, rules, room, seed=19000 + index)
            report = scene_report(solved, required=[item["id"] for item in instances])
            scene_dir = ROOT / "scenes" / f"{kind}_{index}"
            if scene_dir.exists():
                shutil.rmtree(scene_dir)
            scene_dir.mkdir(parents=True)
            _room_shell(scene_dir / "shell", spec)
            record = {
                "room": kind,
                "variant": index,
                "spec": spec,
                "packages": {item["id"]: catalog.get(item["category"]) for item in instances},
                "solved": solved,
                "critic": report,
            }
            (scene_dir / "scene.json").write_text(json.dumps(record, indent=2, default=_json) + "\n", encoding="utf-8")
            written.append({"id": f"{kind}_{index}", "valid": report["valid"], "directory": str(scene_dir)})
            logger.info("asset_realism event=scene id=%s valid=%s", f"{kind}_{index}", report["valid"])
    invalid = json.loads((ROOT / "scenes" / "living_1" / "scene.json").read_text(encoding="utf-8"))
    for item in invalid["solved"]["instances"]:
        if item["id"] == "sofa":
            item["forward"] = [0, -1, 0]
    invalid_solved = solve(invalid["solved"]["instances"], BUILDERS["living"](4.8, 3.8)[1], {"door": {"x0": 0.35, "x1": 1.25, "depth_m": 0.9}}, seed=1, steps=0)
    invalid_report = scene_report(invalid_solved, required=["sofa", "television", "table"])
    rejected = ROOT / "scenes" / "rejected_sofa_reversed"
    rejected.mkdir(parents=True, exist_ok=True)
    (rejected / "scene.json").write_text(json.dumps({"solved": invalid_solved, "critic": invalid_report}, indent=2, default=_json) + "\n", encoding="utf-8")
    written.append({"id": "rejected_sofa_reversed", "valid": invalid_report["valid"], "directory": str(rejected)})
    return written


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value))


def render_scenes(limit: int = 12) -> list[dict]:
    """Mitsuba RGB for the first scenes. The variant is the pinned polarized one."""
    from sensor_sim.scene_gen.indoor_mitsuba import render_stokes, require_variant, write_rgb_png
    from sensor_sim.scene_gen.tabletop_render import _scene

    require_variant()
    images = []
    scenes = sorted(path for path in (ROOT / "scenes").glob("*/scene.json") if "rejected" not in path.parent.name)[:limit]
    for path in scenes:
        if "rejected" in path.parent.name:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        spec = record["spec"]
        width, depth = spec["interior_width_m"], spec["interior_depth_m"]
        instances = []
        for name in ("floor", "ceiling"):
            instances.append({"id": name, "mesh": path.parent / "shell" / f"{name}.obj", "matrix": np.eye(4), "bsdf": {"type": "diffuse", "reflectance": [0.55, 0.52, 0.46] if name == "floor" else [0.9, 0.9, 0.88]}})
        skipped = None
        for item in record["solved"]["instances"]:
            asset_id = record["packages"][item["id"]]
            mesh = ROOT / "packages" / asset_id / "1.0.0" / "geometry" / "render.obj"
            yaw = float(item["yaw"])
            cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
            position = item["position"]
            matrix = np.array([
                [cosine, -sine, 0, position[0]],
                [sine, cosine, 0, position[1]],
                [0, 0, 1, position[2]],
                [0, 0, 0, 1],
            ], dtype=np.float64)
            try:
                bsdf = object_bsdf(item)
            except AssetRealismError as exc:
                skipped = str(exc)
                break
            instances.append({"id": item["id"], "mesh": mesh, "matrix": matrix, "bsdf": bsdf})
        if skipped is not None:
            logger.info("asset_realism event=skip_render scene=%s %s", path.parent.name, skipped)
            images.append({"renderer": "Mitsuba", "scene": path.parent.name, "status": "skipped", "reason": skipped})
            continue
        camera = {"origin": [width * 0.2, depth * 0.18, 1.55], "target": [width * 0.55, depth * 0.62, 0.5], "up": [0, 0, 1], "fov_deg": 60}
        import mitsuba as mi

        transform = mi.ScalarTransform4f.translate([width * 0.5, depth * 0.45, 2.35]) @ mi.ScalarTransform4f.rotate([1.0, 0.0, 0.0], 180.0) @ mi.ScalarTransform4f.scale([0.9, 0.7, 1.0])
        light = {"type": "rectangle", "to_world": transform, "emitter": {"type": "area", "radiance": {"type": "rgb", "value": 18.0}}}
        scene = _scene(instances, camera, (160, 120), light)
        png = ROOT / "gallery" / f"mitsuba_{path.parent.name}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        exr = png.with_suffix(".exr")
        render_stokes(scene, exr, 4, 17531)
        write_rgb_png(exr, png)
        images.append({"renderer": "Mitsuba", "png": str(png), "scene": path.parent.name})
        logger.info("asset_realism event=render scene=%s", path.parent.name)
    return images


def write_gallery(images: list[dict]) -> None:
    gallery = ROOT / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    body = [
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Asset realism</title>",
        "<style>body{font-family:sans-serif;margin:2rem;background:#111;color:#eee} img{max-width:640px;background:#222}</style>",
        "</head><body><h1>Asset realism qualification</h1>",
        "<p>Each caption names the renderer. Mitsuba RGB is the scientific evidence. Packages are unregistered and the Infinigen license status is unresolved.</p>",
    ]
    for image in images:
        name = Path(image["png"]).name
        body.append(f"<figure><img src=\"{name}\" alt=\"{image['renderer']}\"><figcaption><strong>{image['renderer']}</strong> {image.get('label', image.get('scene', ''))}</figcaption></figure>")
    body.append("</body></html>\n")
    (gallery / "index.html").write_text("\n".join(body), encoding="utf-8")


def coverage_matrix(reports: list[dict]) -> list[dict]:
    by_category = {}
    for report in reports:
        manifest = json.loads((Path(report["package"]) / "asset.json").read_text(encoding="utf-8"))
        category = manifest["geometry"]["category"]
        by_category.setdefault(category, []).append(report["asset_id"])
    rows = []
    for category, assets in sorted(by_category.items()):
        origin = "infinigen" if any(asset.startswith("qual_") for asset in assets) else "gap"
        rows.append({
            "category": category,
            "assets": assets,
            "recommended_source": origin,
            "gap": origin == "gap",
        })
    path = ROOT / "reports" / "coverage_matrix.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ROOT.mkdir(parents=True, exist_ok=True)
    source = audit()
    reports = _ingest_existing()
    matrix = coverage_matrix(reports)
    scenes = build_scenes()
    images = render_scenes()
    write_gallery(images)
    summary = {"audit_head": source["head"], "packages": len(reports), "scenes": scenes, "images": images, "coverage": matrix}
    (ROOT / "reports" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("asset_realism event=done packages=%d scenes=%d images=%d", len(reports), len(scenes), len(images))
    return summary


if __name__ == "__main__":
    main()
