"""Blender Cycles previews for the texture qualification.

This script does not import sensor_sim. The driver passes a JSON file after
``--``. Every image it writes is a Blender Cycles render.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _params() -> dict:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("missing -- params.json")
    return json.loads(Path(argv[argv.index("--") + 1]).read_text(encoding="utf-8"))


def _cycles(width: int, height: int, samples: int) -> str:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(samples)
    scene.view_settings.view_transform = "AgX"
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.image_settings.file_format = "PNG"
    device = "CPU"
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "CUDA"
        preferences.get_devices()
        for entry in preferences.devices:
            if entry.type == "CUDA":
                entry.use = True
                device = "GPU"
        scene.cycles.device = device
    except Exception as exc:
        scene.cycles.device = "CPU"
        print(f"BLENDER_CYCLES_DEVICE_FALLBACK error={exc}")
    return device


def _camera(camera: dict) -> None:
    data = bpy.data.cameras.new("qualification_camera")
    data.sensor_fit = "HORIZONTAL"
    data.angle = float(camera["fov_deg"]) * 3.141592653589793 / 180.0
    obj = bpy.data.objects.new("qualification_camera", data)
    obj.location = tuple(camera["origin"])
    direction = Vector(camera["target"]) - Vector(camera["origin"])
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.camera = obj


def _light(camera: dict) -> None:
    data = bpy.data.lights.new("qualification_key", "AREA")
    data.energy = 40.0
    data.size = 0.25
    obj = bpy.data.objects.new("qualification_key", data)
    origin = Vector(camera["origin"])
    target = Vector(camera["target"])
    obj.location = target + (origin - target) * 0.45 + Vector((0.0, 0.0, 0.12))
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.world = bpy.data.worlds.new("qualification_world")
    bpy.context.scene.world.color = (0.02, 0.02, 0.02)


def _import_obj(path: str):
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    imported = [item for item in bpy.data.objects if item not in before and item.type == "MESH"]
    if not imported:
        raise RuntimeError(f"no mesh imported from {path}")
    return imported[0]


def _textured(job: dict) -> None:
    obj = _import_obj(job["obj"])
    material = bpy.data.materials.new("qualification_material")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    uv = tree.nodes.new("ShaderNodeTexCoord")
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    shader.inputs["Metallic"].default_value = float(job.get("metallic", 0.0))
    shader.inputs["IOR"].default_value = float(job.get("ior", 1.45))
    if job.get("base_color"):
        image = tree.nodes.new("ShaderNodeTexImage")
        image.image = bpy.data.images.load(job["base_color"])
        image.image.colorspace_settings.name = "sRGB"
        tree.links.new(uv.outputs["UV"], image.inputs["Vector"])
        tree.links.new(image.outputs["Color"], shader.inputs["Base Color"])
    else:
        color = job.get("flat_color", [0.6, 0.1, 0.1])
        shader.inputs["Base Color"].default_value = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    if job.get("roughness"):
        image = tree.nodes.new("ShaderNodeTexImage")
        image.image = bpy.data.images.load(job["roughness"])
        image.image.colorspace_settings.name = "Non-Color"
        tree.links.new(uv.outputs["UV"], image.inputs["Vector"])
        tree.links.new(image.outputs["Color"], shader.inputs["Roughness"])
    else:
        shader.inputs["Roughness"].default_value = float(job.get("roughness_value", 0.4))
    if job.get("normal"):
        image = tree.nodes.new("ShaderNodeTexImage")
        image.image = bpy.data.images.load(job["normal"])
        image.image.colorspace_settings.name = "Non-Color"
        normal = tree.nodes.new("ShaderNodeNormalMap")
        normal.space = "TANGENT"
        normal.inputs["Strength"].default_value = 1.0
        tree.links.new(uv.outputs["UV"], image.inputs["Vector"])
        tree.links.new(image.outputs["Color"], normal.inputs["Color"])
        tree.links.new(normal.outputs["Normal"], shader.inputs["Normal"])
    obj.data.materials.append(material)
    print(
        "BLENDER_PREVIEW",
        json.dumps({
            "object": obj.name,
            "vertices": len(obj.data.vertices),
            "polygons": len(obj.data.polygons),
            "uv_layers": len(obj.data.uv_layers),
            "dimensions_m": [float(value) for value in obj.dimensions],
        }),
    )


def _match_polyhaven(job: dict) -> None:
    bpy.ops.wm.open_mainfile(filepath=job["blend"])
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("polyhaven blend has no mesh")
    apple = max(meshes, key=lambda obj: obj.dimensions.length)
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    height = float(apple.dimensions.z)
    if height <= 0.0:
        raise RuntimeError("polyhaven height is zero")
    factor = float(job["target_height_m"]) / height
    apple.scale = tuple(float(value) * factor for value in apple.scale)
    bpy.context.view_layer.update()
    corners = [apple.matrix_world @ Vector(corner) for corner in apple.bound_box]
    minimum = Vector((min(corner.x for corner in corners), min(corner.y for corner in corners), min(corner.z for corner in corners)))
    maximum = Vector((max(corner.x for corner in corners), max(corner.y for corner in corners), max(corner.z for corner in corners)))
    center = (minimum + maximum) * 0.5
    target = Vector(job["target_center_m"])
    apple.location += Vector((target.x - center.x, target.y - center.y, target.z - center.z))
    bpy.context.view_layer.update()
    print(
        "POLYHAVEN_PLACED",
        json.dumps({
            "object": apple.name,
            "scale_factor": factor,
            "source_height_m": height,
            "target_height_m": float(job["target_height_m"]),
            "license_note": "CC0 authored appearance benchmark, not a measured material",
        }),
    )


def main() -> None:
    params = _params()
    for job in params["jobs"]:
        if job["kind"] == "polyhaven":
            _match_polyhaven(job)
        else:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            _textured(job)
        device = _cycles(params["width"], params["height"], params["samples"])
        _camera(job["camera"])
        _light(job["camera"])
        bpy.context.scene.render.filepath = job["png"]
        Path(job["png"]).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.render.render(write_still=True)
        print(f"BLENDER_CYCLES_RENDER device={device} path={job['png']}")


if __name__ == "__main__":
    main()
