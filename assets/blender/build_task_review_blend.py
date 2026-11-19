"""Import a task scene.json into a self-contained Blender file.

Meshes are embedded. Each instance matrix is the 16 row-major floats from
scene.json, applied as matrix_world. Review colors match the Scene A–C
render map used by the task renders. The sun and gray background are for
looking at the geometry in Blender. They are not the Mitsuba lighting.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
import mathutils

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "assets" / "candidates"
RENDER_BSDF = {
    "room_floor": {"type": "diffuse", "reflectance": [0.55, 0.52, 0.46]},
    "room_ceiling": {"type": "diffuse", "reflectance": [0.9, 0.9, 0.88]},
    "room_wall_south": {"type": "diffuse", "reflectance": [0.82, 0.82, 0.78]},
    "room_wall_north": {"type": "diffuse", "reflectance": [0.82, 0.82, 0.78]},
    "room_wall_west": {"type": "diffuse", "reflectance": [0.82, 0.82, 0.78]},
    "room_wall_east": {"type": "diffuse", "reflectance": [0.82, 0.82, 0.78]},
    "table_rect": {"type": "diffuse", "reflectance": [0.45, 0.28, 0.16]},
    "shelf_two": {"type": "diffuse", "reflectance": [0.36, 0.24, 0.14]},
    "prop_crate": {"type": "diffuse", "reflectance": [0.62, 0.24, 0.14]},
    "prop_book": {"type": "diffuse", "reflectance": [0.15, 0.35, 0.22]},
    "prop_cup": {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0},
}


def _args() -> tuple[Path, Path, Path | None]:
    argv = sys.argv
    if "--" not in argv or len(argv) < argv.index("--") + 3:
        raise SystemExit(
            "usage: blender --background --python build_task_review_blend.py -- scene.json out.blend [validation.json]"
        )
    start = argv.index("--") + 1
    validation = Path(argv[start + 2]) if len(argv) > start + 2 else None
    return Path(argv[start]), Path(argv[start + 1]), validation


def _matrix(values: list[float]) -> mathutils.Matrix:
    rows = [values[0:4], values[4:8], values[8:12], values[12:16]]
    return mathutils.Matrix(rows)


def _import_obj(path: Path):
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    created = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    if len(created) != 1:
        raise RuntimeError(f"{path} imported {len(created)} meshes")
    return created[0]


def _bsdf(asset_id: str, package_path: Path) -> dict:
    if asset_id in RENDER_BSDF:
        return dict(RENDER_BSDF[asset_id])
    package = json.loads(package_path.read_text(encoding="utf-8"))
    materials = package.get("materials") or []
    if not materials:
        raise RuntimeError(f"{asset_id} has no material")
    return dict(materials[0])


def _color(bsdf: dict) -> list[float]:
    if "reflectance" in bsdf:
        return [float(channel) for channel in bsdf["reflectance"]]
    if "diffuse_reflectance" in bsdf:
        return [float(channel) for channel in bsdf["diffuse_reflectance"]]
    metal = {"Al": [0.91, 0.92, 0.92], "Cu": [0.95, 0.64, 0.54], "Au": [1.0, 0.77, 0.34], "Ag": [0.97, 0.96, 0.91]}
    if bsdf.get("type") in {"conductor", "roughconductor"}:
        return metal.get(str(bsdf.get("material")), [0.8, 0.8, 0.8])
    return [0.7, 0.7, 0.7]


def _material(name: str, bsdf: dict):
    color = _color(bsdf)
    kind = bsdf.get("type")
    dielectric = kind == "dielectric"
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = (color[0], color[1], color[2], 1.0)
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    if dielectric:
        shader.inputs["Roughness"].default_value = 0.05
        shader.inputs["Transmission Weight"].default_value = 1.0
        shader.inputs["IOR"].default_value = float(bsdf.get("int_ior", 1.5))
    elif kind == "conductor":
        shader.inputs["Metallic"].default_value = 1.0
        shader.inputs["Roughness"].default_value = 0.05
    elif kind == "roughconductor":
        shader.inputs["Metallic"].default_value = 1.0
        shader.inputs["Roughness"].default_value = 0.35
    elif kind == "pplastic":
        shader.inputs["Roughness"].default_value = 0.15
    else:
        shader.inputs["Roughness"].default_value = 0.6
    return material


def _camera(name: str, origin: list[float], target: list[float], fov_deg: float):
    camera = bpy.data.cameras.new(name)
    camera.sensor_fit = "HORIZONTAL"
    camera.angle = math.radians(float(fov_deg))
    camera.clip_start = 0.01
    camera.clip_end = 100.0
    obj = bpy.data.objects.new(name, camera)
    direction = mathutils.Vector(target) - mathutils.Vector(origin)
    obj.matrix_world = direction.to_track_quat("-Z", "Y").to_matrix().to_4x4()
    obj.matrix_world.translation = mathutils.Vector(origin)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _world_bounds(obj) -> tuple[list[float], list[float]]:
    corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    low = [min(corner[axis] for corner in corners) for axis in range(3)]
    high = [max(corner[axis] for corner in corners) for axis in range(3)]
    return low, high


def main() -> None:
    scene_path, blend_path, validation_path = _args()
    manifest = json.loads(scene_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path else {}
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    world = bpy.data.worlds.new("review_world")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.55, 0.58, 0.62, 1.0)
    background.inputs["Strength"].default_value = 0.35
    scene.world = world
    sun_data = bpy.data.lights.new("review_sun", "SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("review_sun", sun_data)
    sun.rotation_euler = (math.radians(55.0), 0.0, math.radians(20.0))
    scene.collection.objects.link(sun)

    for instance in manifest["instances"]:
        package = CATALOG / instance["asset_id"] / instance["version"]
        obj = _import_obj(package / "geometry" / "render.obj")
        obj.name = instance["instance_id"]
        obj.data.name = instance["instance_id"]
        obj.matrix_world = _matrix(instance["matrix"])
        bsdf = _bsdf(instance["asset_id"], package / "asset.json")
        material = _material(instance["instance_id"], bsdf)
        obj.data.materials.append(material)
        color = _color(bsdf)
        obj.color = (color[0], color[1], color[2], 1.0)
        low, high = _world_bounds(obj)
        print(
            f"task_review_blend object={obj.name} asset={instance['asset_id']} "
            f"world_min={[round(value, 6) for value in low]} world_max={[round(value, 6) for value in high]}"
        )

    cameras = validation.get("cameras") or {}
    overview = validation.get("camera") or cameras.get("room") or {
        "origin": [2.2, -1.4, 1.6],
        "target": [0.6, 0.35, 0.4],
        "fov_deg": 45.0,
    }
    detail = validation.get("detail_camera") or cameras.get("tabletop") or overview
    _camera("overview", overview["origin"], overview["target"], overview["fov_deg"])
    active = _camera("detail", detail["origin"], detail["target"], detail["fov_deg"])
    scene.camera = active
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path.resolve()))
    print(f"task_review_blend wrote={blend_path.resolve()} task_id={manifest.get('task_id')}")


if __name__ == "__main__":
    main()
