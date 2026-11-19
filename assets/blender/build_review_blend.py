"""Import a qualification scene.json into a self-contained Blender file.

The meshes are copied into the .blend. Blender is Z-up. Each instance matrix
is the 16 row-major floats from the manifest, applied as matrix_world.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
import mathutils


def _args() -> tuple[Path, Path]:
    argv = sys.argv
    if "--" not in argv or len(argv) < argv.index("--") + 3:
        raise SystemExit("usage: blender --background --python build_review_blend.py -- scene.json out.blend")
    start = argv.index("--") + 1
    return Path(argv[start]), Path(argv[start + 1])


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


def _material(name: str, color: list[float], dielectric: bool):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    shader.inputs["Roughness"].default_value = 0.05 if dielectric else 0.6
    if dielectric:
        shader.inputs["Transmission Weight"].default_value = 1.0
        shader.inputs["IOR"].default_value = 1.5
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


def main() -> None:
    scene_path, blend_path = _args()
    manifest = json.loads(scene_path.read_text(encoding="utf-8"))
    catalog = Path("assets/candidates")
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
        obj_path = catalog / instance["asset_id"] / instance["version"] / "geometry" / "render.obj"
        obj = _import_obj(obj_path)
        obj.name = instance["instance_id"]
        obj.data.name = instance["instance_id"]
        obj.matrix_world = _matrix(instance["matrix"])
        material = _material(instance["instance_id"], instance["reflectance"], bool(instance["dielectric"]))
        obj.data.materials.append(material)
        low = [min(corner[axis] for corner in obj.bound_box) for axis in range(3)]
        high = [max(corner[axis] for corner in obj.bound_box) for axis in range(3)]
        world_low = obj.matrix_world @ mathutils.Vector(low)
        world_high = obj.matrix_world @ mathutils.Vector(high)
        print(
            f"review_blend object={obj.name} world_min={[round(world_low[i], 3) for i in range(3)]} "
            f"world_max={[round(world_high[i], 3) for i in range(3)]}"
        )

    overview = _camera("overview", [2.15, 0.55, 1.65], [2.05, 2.4, 1.05], 60.0)
    _camera("door", [0.35, 1.15, 1.35], [0.72, 0.12, 1.15], 48.0)
    _camera("window", [0.35, 2.15, 1.45], [0.85, 3.32, 1.4], 42.0)
    scene.camera = overview
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path.resolve()))
    print(f"review_blend wrote={blend_path.resolve()}")


if __name__ == "__main__":
    main()
