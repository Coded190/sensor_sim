"""Build one Z-up scene in Blender and export Mitsuba XML.

Run by Blender: ``blender --background --python create_dataset_scene.py -- params.json``.
The official add-on operator is ``bpy.ops.export_scene.mitsuba``. Axis ``Y``
forward and ``Z`` up keeps Blender's Z-up coordinates.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
import mathutils

ADDON = "bl_ext.user_default.mitsuba_blender"


def _params() -> dict:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("missing -- params.json")
    return json.loads(Path(argv[argv.index("--") + 1]).read_text())


def _clear() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    enabled = bpy.ops.preferences.addon_enable(module=ADDON)
    if enabled != {"FINISHED"}:
        raise RuntimeError(f"could not enable {ADDON}: {enabled}")


def _link(obj):
    bpy.context.collection.objects.link(obj)
    return obj


def _diffuse(name: str, color: list[float]):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    shader = tree.nodes.new("ShaderNodeBsdfDiffuse")
    shader.inputs["Color"].default_value = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def _glass(name: str, ior: float):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    shader = tree.nodes.new("ShaderNodeBsdfGlass")
    shader.inputs["IOR"].default_value = float(ior)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def _assign(obj, material) -> None:
    obj.data.materials.append(material)


def _add_object(spec: dict):
    kind = spec["kind"]
    if kind == "uv_sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=float(spec["radius"]),
            location=tuple(spec["location"]),
            segments=int(spec.get("segments", 128)),
            ring_count=int(spec.get("rings", 64)),
        )
        obj = bpy.context.active_object
    elif kind == "plane":
        bpy.ops.mesh.primitive_plane_add(size=float(spec["size"]), location=tuple(spec["location"]))
        obj = bpy.context.active_object
    elif kind == "obj":
        bpy.ops.wm.obj_import(filepath=str(spec["filename"]), forward_axis="Y", up_axis="Z")
        selected = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
        if len(selected) != 1:
            raise RuntimeError(f"{spec['filename']} imported {len(selected)} meshes")
        obj = selected[0]
    else:
        raise RuntimeError(f"unknown blender object kind {kind}")
    obj.name = str(spec["name"])
    source = spec.get("share_datablock_with")
    if source:
        obj.data = bpy.data.objects[str(source)].data
    # A shared datablock makes the Mitsuba exporter drop the second object.
    # Copy after any requested link so each logical object is its own shape.
    obj.data = obj.data.copy()
    obj.data.name = str(spec["name"])
    obj.data.materials.clear()
    if spec.get("object_location"):
        obj.location = mathutils.Vector([float(value) for value in spec["object_location"]])
    material_name = f"obj_{spec['name']}"
    if spec.get("blender_material") == "glass":
        _assign(obj, _glass(material_name, float(spec.get("ior", 1.5))))
    else:
        color = spec.get("color", [0.5, 0.5, 0.5])
        _assign(obj, _diffuse(material_name, color))
    return obj


def _camera(spec: dict, scene) -> bpy.types.Object:
    data = bpy.data.cameras.new("camera")
    data.sensor_fit = "HORIZONTAL"
    data.lens_unit = "FOV"
    data.angle = math.radians(float(spec["fov_deg"]))
    data.clip_start = float(spec.get("near_clip", 0.01))
    data.clip_end = float(spec.get("far_clip", 100.0))
    data.shift_x = 0.0
    data.shift_y = 0.0
    camera = _link(bpy.data.objects.new("camera", data))
    scene.camera = camera
    camera.location = mathutils.Vector(spec["position"])
    direction = mathutils.Vector(spec["target"]) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return camera


def _light(spec: dict) -> None:
    data = bpy.data.lights.new("light", "AREA")
    size_x, size_y = float(spec["size"][0]), float(spec["size"][1])
    data.size = size_x
    data.size_y = size_y
    data.energy = math.pi * size_x * size_y * float(spec["radiance"])
    light = _link(bpy.data.objects.new("light", data))
    light.location = mathutils.Vector(spec["position"])
    direction = mathutils.Vector(spec["target"]) - light.location
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _rays(camera, width: int, height: int, pixels: list) -> list[dict]:
    scale = math.tan(float(camera.data.angle) / 2.0)
    matrix = camera.matrix_world
    origin = [float(value) for value in camera.location]
    rays = []
    for col, row in pixels:
        ndc_x = ((float(col) + 0.5) / float(width)) * 2.0 - 1.0
        ndc_y = 1.0 - ((float(row) + 0.5) / float(height)) * 2.0
        local = mathutils.Vector((ndc_x * scale, ndc_y * scale, -1.0)).normalized()
        direction = matrix.to_3x3() @ local
        rays.append(
            {
                "pixel": [int(col), int(row)],
                "origin": origin,
                "direction": [float(direction.x), float(direction.y), float(direction.z)],
            }
        )
    return rays


def main() -> None:
    params = _params()
    _clear()
    scene = bpy.context.scene
    width, height = int(params["resolution"][0]), int(params["resolution"][1])
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.cycles.samples = int(params["spp"])
    scene.cycles.max_bounces = int(params["max_depth"])
    for spec in params["objects"]:
        _add_object(spec)
    camera = _camera(params["camera"], scene)
    _light(params["light"])
    xml = str(params["xml"])
    Path(xml).parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.export_scene.mitsuba(
        filepath=xml,
        export_ids=True,
        ignore_background=True,
        axis_forward="Y",
        axis_up="Z",
        check_existing=False,
    )
    if result != {"FINISHED"}:
        raise RuntimeError(f"export_scene.mitsuba returned {result}")
    payload = {
        "rays": _rays(camera, width, height, params["ray_pixels"]),
        "fov_x_deg": math.degrees(camera.data.angle_x),
        "fov_y_deg": math.degrees(camera.data.angle_y),
        "lens_mm": float(camera.data.lens),
        "sensor_width_mm": float(camera.data.sensor_width),
        "sensor_height_mm": float(camera.data.sensor_height),
        "sensor_fit": camera.data.sensor_fit,
        "shift_x": float(camera.data.shift_x),
        "shift_y": float(camera.data.shift_y),
        "pixel_aspect": 1.0,
        "row_0_is_top": True,
        "clip_start": float(camera.data.clip_start),
        "clip_end": float(camera.data.clip_end),
    }
    Path(params["rays"]).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"dataset_scene status=exported xml={xml}")


if __name__ == "__main__":
    main()
