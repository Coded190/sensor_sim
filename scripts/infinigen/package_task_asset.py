"""Package one exported Infinigen asset as asset_package_v1 plus a sidecar.

The sidecar is written next to the package directory, not inside it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_extension import validate_extension
from sensor_sim.scene_gen.asset_package import create_package, verify_package
from sensor_sim.scene_gen.scene_manifest import matrix_lists, translation_matrix
from sensor_sim.scene_gen.task_scene_geometry import (
    bounds,
    convex_collision,
    load_exported_obj,
    surface_distance,
    table_collision,
    tabletop_mask,
    write_obj,
    write_textured_obj,
)

COMMIT = "b11700ebac77695d8f53538705cfdc0cd890c54d"
ROOT = Path(__file__).resolve().parents[2] / "dataset" / "task_scene"


def package_asset(source_obj: Path, blend: Path, asset_id: str, factory: str, seed: int, kind: str) -> dict:
    loaded = load_exported_obj(source_obj)
    vertices = loaded["vertices"] - np.array([loaded["vertices"][:, 0].mean(), loaded["vertices"][:, 1].mean(), loaded["vertices"][:, 2].min()])
    loaded["vertices"] = vertices
    if kind == "table":
        collision_v, collision_f, collision_note = table_collision(vertices)
        mask = tabletop_mask(vertices)
        top = vertices[mask]
        distance = {
            "render_to_collision": surface_distance(top, collision_v, collision_f),
            "collision_to_render": surface_distance(collision_v, vertices, loaded["faces"], sample_count=64),
        }
        distance["max_m"] = distance["render_to_collision"]["max_m"]
        support_z = float(top[:, 2].max())
        frame_name = "support_surface"
        polygon = _rectangle(top)
    else:
        collision_v, collision_f = convex_collision(vertices)
        collision_note = {"method": "convex_mesh"}
        distance = {
            "render_to_collision": surface_distance(vertices, collision_v, collision_f),
            "collision_to_render": surface_distance(collision_v, vertices, loaded["faces"], sample_count=64),
        }
        distance["max_m"] = max(distance["render_to_collision"]["max_m"], distance["collision_to_render"]["max_m"])
        support_z = 0.0
        frame_name = "base_contact"
        polygon = _rectangle(vertices)
    stage = ROOT / "staging" / asset_id
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    write_textured_obj(stage / "geometry" / "render.obj", vertices, loaded["corners"], loaded["texcoords"], loaded["normals"], loaded["face_materials"])
    _copy_materials(source_obj.parent, stage / "geometry")
    write_obj(stage / "geometry" / "collision.obj", collision_v, collision_f)
    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    support_transform = translation_matrix(0.0, 0.0, support_z)
    manifest = {
        "asset_id": asset_id,
        "version": "1.0.0",
        "provenance": {
            "source": "infinigen_indoors",
            "generator": factory,
            "generator_version": "indoors-stable",
            "seed": seed,
            "parameters": {"export": "obj", "export_texture_res": 1024, "no_ground": True, "coordinate_conversion": "exported_y_up_to_canonical_then_min_z_at_origin"},
            "license": "BSD-3-Clause upstream code; generated mesh redistribution unresolved",
            "license_status": "unresolved",
            "upstream_commit": COMMIT,
            "upstream_object": factory,
        },
        "geometry": {"render_mesh": "geometry/render.obj", "units": "m", "frame": "asset_root"},
        "collision": {"mesh": "geometry/collision.obj", "method": collision_note["method"]},
        "units": "m",
        "frame": {"up": [0, 0, 1], "handedness": "right"},
        "support_surfaces": [{"id": "top" if kind == "table" else "base", "frame": frame_name, "polygon_m": polygon}],
    }
    roles = {}
    payload = {}
    for path in stage.rglob("*"):
        if path.is_file():
            relative = path.relative_to(stage).as_posix()
            payload[relative] = path.read_bytes()
            roles[relative] = "collision" if relative.endswith("collision.obj") else "texture" if path.suffix.lower() == ".png" else "material" if path.suffix.lower() == ".mtl" else "render"
    destination = ROOT / "library" / asset_id / "1.0.0"
    if destination.exists():
        shutil.rmtree(destination)
    sealed = create_package(destination, manifest, payload, roles)
    verified = verify_package(destination)
    extension = _extension(verified, frame_name, support_transform, identity, collision_note["method"])
    validate_extension(extension, verified)
    sidecar = ROOT / "library" / asset_id / "1.0.0.asset_extension.json"
    sidecar.write_text(json.dumps(extension, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    blend_copy = ROOT / "source_blends" / f"{asset_id}.blend"
    blend_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(blend, blend_copy)
    qualification = {
        "asset_id": asset_id,
        "package_sha256": verified["package_sha256"],
        "bounds_m": bounds(vertices),
        "collision": collision_note,
        "surface_distance": distance,
        "blend_sha256": hashlib.sha256(blend_copy.read_bytes()).hexdigest(),
        "license_status": "unresolved",
        "tabletop_tolerance_m": 0.02,
        "object_tolerance_m": 0.01,
    }
    limit = qualification["tabletop_tolerance_m"] if kind == "table" else qualification["object_tolerance_m"]
    qualification["distance_within_tolerance"] = bool(distance["max_m"] <= limit)
    report_path = ROOT / "reports" / f"{asset_id}_qualification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(qualification, indent=2) + "\n", encoding="utf-8")
    return qualification


def _extension(package: dict, frame_name: str, support_transform: list, identity: list, method: str) -> dict:
    return {
        "schema_version": "asset_extension_v1",
        "asset_id": package["asset_id"],
        "package_version": package["version"],
        "package_sha256": package["package_sha256"],
        "frames": [
            {"name": "asset_root", "purpose": "packaged mesh origin", "origin_m": [0, 0, 0], "axes": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}, "transform_to_asset": identity, "units": "m", "provenance": "estimated"},
            {"name": frame_name, "purpose": "tabletop" if frame_name == "support_surface" else "base contact", "origin_m": [0, 0, support_transform[2][3]], "axes": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}, "transform_to_asset": support_transform, "units": "m", "provenance": "estimated"},
        ],
        "regions": [
            {
                "id": "support" if frame_name == "support_surface" else "base",
                "region_type": frame_name,
                "frame": frame_name,
                "geometry": "rectangle",
                "normal": [0, 0, 1],
                "units": "m",
                "provenance": "estimated",
                "method": "highest_z_band" if frame_name == "support_surface" else "minimum_z_after_conversion",
            }
        ],
        "geometries": [
            {"role": "render", "path": "geometry/render.obj", "sha256": _file_hash(package, "geometry/render.obj"), "frame": "asset_root", "units": "m", "method": "pinned_obj_export_converted_to_z_up", "validation": "uv_normal_material_checked"},
            {"role": "collision", "path": "geometry/collision.obj", "sha256": _file_hash(package, "geometry/collision.obj"), "frame": "asset_root", "units": "m", "method": method, "validation": "surface_distance_recorded"},
        ],
    }


def _file_hash(package: dict, relative: str) -> str:
    for item in package["files"]:
        if item["path"] == relative:
            return item["sha256"]
    raise RuntimeError(f"reason=missing_packaged_file path={relative}")


def _rectangle(points: np.ndarray) -> list[list[float]]:
    low = points.min(axis=0)
    high = points.max(axis=0)
    return [[float(low[0]), float(low[1]), 0.0], [float(high[0]), float(low[1]), 0.0], [float(high[0]), float(high[1]), 0.0], [float(low[0]), float(high[1]), 0.0]]


def _copy_materials(export_dir: Path, geometry_dir: Path) -> None:
    geometry_dir.mkdir(parents=True, exist_ok=True)
    for path in export_dir.iterdir():
        if path.suffix.lower() in {".mtl", ".png"}:
            shutil.copy2(path, geometry_dir / path.name)
    for path in (export_dir / "textures").glob("*.png") if (export_dir / "textures").is_dir() else []:
        target = geometry_dir / path.name
        if not target.exists():
            shutil.copy2(path, target)


def main() -> None:
    table = package_asset(
        ROOT / "generation" / "table_dining_seed7" / "export_007" / "export_007.obj",
        ROOT / "generation" / "table_dining_seed7" / "TableDiningFactory_000" / "scene.blend",
        "table_dining",
        "TableDiningFactory",
        7,
        "table",
    )
    can = package_asset(
        ROOT / "generation" / "can_seed11" / "export_011" / "export_011.obj",
        ROOT / "generation" / "can_seed11" / "CanFactory_000" / "scene.blend",
        "can",
        "CanFactory",
        11,
        "can",
    )
    print(json.dumps({"table": table, "can": can}, indent=2))


if __name__ == "__main__":
    main()
