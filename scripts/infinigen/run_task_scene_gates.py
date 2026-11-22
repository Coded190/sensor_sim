"""Compose the real table and can, then render and replay that manifest in Mitsuba."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import verify_package
from sensor_sim.scene_gen.indoor_mitsuba import render_stokes, write_rgb_png
from sensor_sim.scene_gen.mesh_io import read_mesh
from sensor_sim.scene_gen.mitsuba_scene_package import build_mitsuba_scene
from sensor_sim.scene_gen.scene_manifest import apply_transform, manifest_hash
from sensor_sim.scene_gen.task_scene_composer import compose_scene

ROOT = Path(__file__).resolve().parents[2] / "dataset" / "task_scene"


def _asset(asset_id: str, category: str) -> dict:
    package_root = ROOT / "library" / asset_id / "1.0.0"
    package = verify_package(package_root)
    extension = json.loads((ROOT / "library" / asset_id / "1.0.0.asset_extension.json").read_text(encoding="utf-8"))
    mesh = read_mesh(package_root / "geometry" / "render.obj")
    faces = []
    for corners in mesh["corners"]:
        indexes = [corner[0] for corner in corners]
        for offset in range(1, len(indexes) - 1):
            faces.append([indexes[0], indexes[offset], indexes[offset + 1]])
    polygon = package["support_surfaces"][0]["polygon_m"]
    return {
        "category": category,
        "package": package,
        "extension": extension,
        "render_vertices": np.asarray(mesh["vertices"], dtype=np.float64),
        "render_faces": np.asarray(faces, dtype=np.int64),
        "support_polygon_local": np.asarray(polygon, dtype=np.float64),
    }


def _project(point: np.ndarray, camera: dict) -> np.ndarray:
    pose = np.asarray(camera["world_from_camera"], dtype=np.float64)
    local = apply_transform(np.linalg.inv(pose), point)
    return np.array([camera["fx"] * local[0] / local[2] + camera["cx"], camera["fy"] * local[1] / local[2] + camera["cy"], local[2]])


def main() -> None:
    assets = [_asset("table_dining", "table"), _asset("can", "can")]
    request = {
        "schema_version": "task_scene_request_v1",
        "task_type": "pick_and_place",
        "support": {"asset_id": "table_dining", "count": {"minimum": 1, "maximum": 1}},
        "objects": [{"asset_id": "can", "count": {"minimum": 1, "maximum": 1}}],
        "target_region": {"frame": "support_surface", "origin_m": [0.15, 0.1, 0.0], "size_m": [0.2, 0.12, 0.04]},
        "camera_profile": {"name": "overview"},
        "light_profile": {"name": "relative_constant"},
        "seeds": {"selection": 1, "layout": 2, "appearance": 3, "camera": 4, "lighting": 5},
        "relations": [{"type": "supported_on"}],
        "clutter": 0.0,
    }
    resolved = compose_scene(request, assets, root="library")
    out = ROOT / "scenes" / "pick_can"
    if out.exists():
        raise SystemExit(f"reason=refusing_to_overwrite path={out}")
    out.mkdir(parents=True)
    (out / "scene_manifest.json").write_text(json.dumps(resolved["manifest"], indent=2) + "\n", encoding="utf-8")
    (out / "validation.json").write_text(json.dumps(resolved["validation"], indent=2) + "\n", encoding="utf-8")
    built = build_mitsuba_scene(resolved["manifest"], ROOT, {"manifest_sha256": resolved["manifest_sha256"]})
    (out / "mitsuba_conversion.json").write_text(json.dumps(built["report"], indent=2) + "\n", encoding="utf-8")
    exr = out / "render.exr"
    png = out / "render.png"
    stokes = render_stokes(built["scene"], exr, 8, 1)
    write_rgb_png(exr, png)
    omitted = json.loads(json.dumps(resolved["manifest"]))
    omitted["instances"][1]["omit"] = True
    empty = build_mitsuba_scene(omitted, ROOT)
    empty_png = out / "render_without_object.png"
    render_stokes(empty["scene"], empty_png.with_suffix(".exr"), 8, 1)
    write_rgb_png(empty_png.with_suffix(".exr"), empty_png)
    import mitsuba as mi

    with_object = np.array(mi.Bitmap(str(exr)))
    without = np.array(mi.Bitmap(str(empty_png.with_suffix(".exr"))))
    difference = float(np.mean(np.abs(with_object[..., :3] - without[..., :3])))
    camera = resolved["manifest"]["cameras"][0]
    object_world = np.asarray(resolved["manifest"]["instances"][1]["world_from_asset"], dtype=np.float64)
    can = assets[1]["render_vertices"]
    projected = np.vstack([_project(object_world[:3, :3] @ vertex + object_world[:3, 3], camera) for vertex in can[:: max(1, len(can) // 200)]])
    in_front = projected[:, 2] > 0
    inside = np.mean((projected[in_front, 0] >= 0) & (projected[in_front, 0] < camera["width"]) & (projected[in_front, 1] >= 0) & (projected[in_front, 1] < camera["height"]))
    copy_root = out / "portable"
    shutil.copytree(ROOT / "library", copy_root / "library")
    shutil.copy2(out / "scene_manifest.json", copy_root / "scene_manifest.json")
    copied = build_mitsuba_scene(resolved["manifest"], copy_root, {"manifest_sha256": manifest_hash(resolved["manifest"])})
    summary = {
        "manifest_sha256": resolved["manifest_sha256"],
        "validation_failures": [row for row in resolved["validation"] if row["status"] == "fail"],
        "stokes_shape": stokes["shape"],
        "variant": stokes["variant"],
        "png": str(png),
        "mean_abs_rgb_difference_against_omitted_object": difference,
        "projected_vertices_inside_image_fraction": float(inside),
        "portable_instance_map": copied["instance_map"],
        "object_visible": bool(difference > 1e-4 and inside > 0.5),
    }
    (out / "gate_c_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
