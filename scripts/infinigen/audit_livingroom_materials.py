"""Read material node graphs from the frozen LivingRoom blend. Does not save it."""

from __future__ import annotations

import json
import sys

import bpy


def _socket(node, *names):
    for name in names:
        if name in node.inputs:
            return node.inputs[name]
    return None


def _linked(socket):
    return socket is not None and len(socket.links) > 0


def _default(socket, fallback=0.0):
    if socket is None:
        return fallback
    value = socket.default_value
    try:
        return [float(channel) for channel in value]
    except TypeError:
        return float(value)


def _surface_node(material):
    if material is None or not material.use_nodes or material.node_tree is None:
        return None
    output = next((node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeOutputMaterial" and getattr(node, "is_active_output", True)), None)
    if output is None or not output.inputs[0].links:
        return None
    return output.inputs[0].links[0].from_node


def describe(material) -> dict:
    node = _surface_node(material)
    record = {
        "name": None if material is None else material.name,
        "uses_nodes": bool(material and material.use_nodes),
        "surface": None if node is None else node.bl_idname,
        "metallic": 0.0,
        "roughness": 0.5,
        "transmission": 0.0,
        "ior": 1.5,
        "emission": 0.0,
        "thin": False,
        "base_color_linked": False,
        "roughness_linked": False,
        "normal_linked": False,
        "bump_linked": False,
        "displacement_linked": False,
        "measured_pbrdf": False,
    }
    if node is None:
        return record
    if node.bl_idname == "ShaderNodeBsdfPrincipled":
        base = _socket(node, "Base Color")
        rough = _socket(node, "Roughness")
        metal = _socket(node, "Metallic")
        ior = _socket(node, "IOR")
        trans = _socket(node, "Transmission Weight", "Transmission")
        normal = _socket(node, "Normal")
        emit = _socket(node, "Emission Strength", "Emission")
        record["base_color_linked"] = _linked(base)
        record["roughness_linked"] = _linked(rough)
        record["metallic"] = _default(metal, 0.0) if not _linked(metal) else 0.0
        record["roughness"] = _default(rough, 0.5) if not _linked(rough) else 0.5
        record["ior"] = _default(ior, 1.45)
        record["transmission"] = _default(trans, 0.0) if not _linked(trans) else 1.0
        record["emission"] = _default(emit, 0.0) if not _linked(emit) else 1.0
        record["normal_linked"] = _linked(normal)
        if _linked(normal):
            source = normal.links[0].from_node
            record["bump_linked"] = source.bl_idname == "ShaderNodeBump"
            record["normal_linked"] = source.bl_idname == "ShaderNodeNormalMap" or record["bump_linked"]
    output = next((item for item in material.node_tree.nodes if item.bl_idname == "ShaderNodeOutputMaterial"), None)
    if output is not None and "Displacement" in output.inputs:
        record["displacement_linked"] = _linked(output.inputs["Displacement"])
    record["thin"] = "glass" in (material.name or "").lower() and record["transmission"] > 0.0 and record["roughness"] < 0.05
    return record


def main() -> None:
    blend = sys.argv[sys.argv.index("--") + 1]
    output = sys.argv[sys.argv.index("--") + 2]
    bpy.ops.wm.open_mainfile(filepath=blend)
    materials = [describe(material) for material in bpy.data.materials]
    objects = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        slots = [None if slot.material is None else slot.material.name for slot in obj.material_slots]
        objects.append({"name": obj.name, "slots": slots, "hide_render": bool(obj.hide_render)})
    with open(output, "w", encoding="utf-8") as handle:
        json.dump({"materials": materials, "objects": objects}, handle)
        handle.write("\n")
    print(f"event=material_audit materials={len(materials)} objects={len(objects)}")


if __name__ == "__main__":
    main()
