"""Export a small material-transfer scene with the pinned Mitsuba add-on.

Blender runs this file. It does not import sensor_sim. The driver writes
cases.json and reads the XML afterwards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy

ADDON = "bl_ext.user_default.mitsuba_blender"


def _params() -> dict:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("missing -- params.json")
    return json.loads(Path(argv[argv.index("--") + 1]).read_text(encoding="utf-8"))


def _clear() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    enabled = bpy.ops.preferences.addon_enable(module=ADDON)
    if enabled != {"FINISHED"}:
        raise RuntimeError(f"could not enable {ADDON}: {enabled}")


def _output(tree):
    node = tree.nodes.new("ShaderNodeOutputMaterial")
    return node


def _material_from_recipe(name: str, recipe: dict):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    kind = recipe["node"]
    node_type = {"BSDF_DIFFUSE": "ShaderNodeBsdfDiffuse", "BSDF_GLASS": "ShaderNodeBsdfGlass", "BSDF_GLOSSY": "ShaderNodeBsdfGlossy", "BSDF_PRINCIPLED": "ShaderNodeBsdfPrincipled"}[kind]
    shader = tree.nodes.new(node_type)
    output = _output(tree)
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    for socket_name, value in recipe["sockets"].items():
        socket = shader.inputs[socket_name]
        if isinstance(value, list):
            socket.default_value = (float(value[0]), float(value[1]), float(value[2]), 1.0)
        else:
            socket.default_value = float(value)
    return material, shader


def _cube(name: str, location, material):
    bpy.ops.mesh.primitive_cube_add(size=0.2, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def _two_slot_cube(location):
    bpy.ops.mesh.primitive_cube_add(size=0.2, location=location)
    obj = bpy.context.active_object
    obj.name = "two_slots"
    red = bpy.data.materials.new("slot_red")
    red.use_nodes = True
    red.node_tree.nodes.clear()
    shader = red.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
    shader.inputs["Color"].default_value = (0.8, 0.1, 0.1, 1.0)
    output = red.node_tree.nodes.new("ShaderNodeOutputMaterial")
    red.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    blue = bpy.data.materials.new("slot_blue")
    blue.use_nodes = True
    blue.node_tree.nodes.clear()
    shader = blue.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
    shader.inputs["Color"].default_value = (0.1, 0.2, 0.8, 1.0)
    output = blue.node_tree.nodes.new("ShaderNodeOutputMaterial")
    blue.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    obj.data.materials.append(red)
    obj.data.materials.append(blue)
    for polygon in obj.data.polygons:
        polygon.material_index = 0 if polygon.normal.z > 0.5 else 1
    return obj


def _image_material(path: Path):
    image = bpy.data.images.load(str(path))
    image.colorspace_settings.name = "sRGB"
    material = bpy.data.materials.new("image_texture")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    shader = tree.nodes.new("ShaderNodeBsdfDiffuse")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(texture.outputs["Color"], shader.inputs["Color"])
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def _normal_material(path: Path):
    image = bpy.data.images.load(str(path))
    image.colorspace_settings.name = "Non-Color"
    material = bpy.data.materials.new("normal_map")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    normal = tree.nodes.new("ShaderNodeNormalMap")
    normal.inputs["Strength"].default_value = 0.5
    shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = (0.4, 0.4, 0.4, 1.0)
    shader.inputs["Roughness"].default_value = 0.3
    shader.inputs["Metallic"].default_value = 0.0
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(texture.outputs["Color"], normal.inputs["Color"])
    tree.links.new(normal.outputs["Normal"], shader.inputs["Normal"])
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def main() -> None:
    params = _params()
    destination = Path(params["xml"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    _clear()
    x_value = 0.0
    for case in params["cases"]:
        if "recipe" not in case:
            continue
        material, _shader = _material_from_recipe(case["case_id"], case["recipe"])
        _cube(case["case_id"], (x_value, 0.0, 0.1), material)
        x_value += 0.4
    texture = Path(params["texture"])
    _cube("image_texture", (x_value, 0.0, 0.1), _image_material(texture))
    x_value += 0.4
    _cube("normal_map", (x_value, 0.0, 0.1), _normal_material(texture))
    x_value += 0.4
    _two_slot_cube((x_value, 0.0, 0.1))
    report = []
    for material in bpy.data.materials:
        if not material.use_nodes or material.node_tree is None:
            continue
        images = []
        sockets = []
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                images.append({
                    "node": node.name,
                    "path": bpy.path.abspath(node.image.filepath),
                    "color_space": node.image.colorspace_settings.name,
                })
            if node.type in {"BSDF_PRINCIPLED", "BSDF_DIFFUSE", "BSDF_GLOSSY", "BSDF_GLASS"}:
                sockets.append({"node": node.type, "inputs": [socket.name for socket in node.inputs if socket.is_linked or socket.name in {"Base Color", "Color", "Roughness", "Normal", "IOR", "Metallic"}]})
        report.append({"material": material.name, "images": images, "sockets": sockets})
    report_path = Path(params["node_report"]) if params.get("node_report") else destination.with_name("node_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    bpy.ops.export_scene.mitsuba(filepath=str(destination), export_ids=True, ignore_background=True, axis_forward="Y", axis_up="Z", check_existing=False)
    print(f"TRANSFER_EXPORT path={destination}")


if __name__ == "__main__":
    main()
