"""Read one blend and write cameras, lights, and object matrices. Does not save the blend."""

from __future__ import annotations

import json
import sys

import bpy


def _matrix(value) -> list[list[float]]:
    return [[float(value[row][column]) for column in range(4)] for row in range(4)]


def _vector(value) -> list[float]:
    return [float(item) for item in value]


def main() -> None:
    blend = sys.argv[sys.argv.index("--") + 1]
    output = sys.argv[sys.argv.index("--") + 2]
    bpy.ops.wm.open_mainfile(filepath=blend)
    scene = bpy.context.scene
    bake = scene.render.bake
    objects = []
    for obj in bpy.data.objects:
        materials = []
        if obj.type == "MESH":
            for slot in obj.material_slots:
                material = slot.material
                cycles = getattr(material, "cycles", None) if material is not None else None
                displacement = None
                if cycles is not None and hasattr(cycles, "displacement_method"):
                    displacement = str(cycles.displacement_method)
                elif material is not None and hasattr(material, "displacement_method"):
                    displacement = str(material.displacement_method)
                materials.append({"name": None if material is None else material.name, "displacement_method": displacement})
        objects.append({
            "name": obj.name,
            "type": obj.type,
            "matrix_world": _matrix(obj.matrix_world),
            "location": _vector(obj.location),
            "rotation_euler": _vector(obj.rotation_euler),
            "scale": _vector(obj.scale),
            "dimensions": _vector(obj.dimensions) if obj.type == "MESH" else None,
            "hide_render": bool(obj.hide_render),
            "parent": None if obj.parent is None else obj.parent.name,
            "materials": materials,
            "vertices": int(len(obj.data.vertices)) if obj.type == "MESH" and obj.data is not None else 0,
        })
    cameras = []
    for obj in bpy.data.objects:
        if obj.type != "CAMERA":
            continue
        camera = obj.data
        cameras.append({
            "name": obj.name,
            "matrix_world": _matrix(obj.matrix_world),
            "location": _vector(obj.location),
            "lens_mm": float(camera.lens),
            "sensor_width_mm": float(camera.sensor_width),
            "sensor_height_mm": float(camera.sensor_height),
            "sensor_fit": str(camera.sensor_fit),
            "clip_start_m": float(camera.clip_start),
            "clip_end_m": float(camera.clip_end),
            "shift_x": float(camera.shift_x),
            "shift_y": float(camera.shift_y),
        })
    lights = []
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        light = obj.data
        lights.append({
            "name": obj.name,
            "light_type": str(light.type),
            "matrix_world": _matrix(obj.matrix_world),
            "color": _vector(light.color),
            "energy": float(light.energy),
            "size": float(getattr(light, "size", 0.0)),
        })
    world = None
    if scene.world is not None:
        world = {"name": scene.world.name, "use_nodes": bool(scene.world.use_nodes)}
    payload = {
        "blend": blend,
        "saved": False,
        "render_resolution": [int(scene.render.resolution_x), int(scene.render.resolution_y)],
        "normal_space": str(getattr(bake, "normal_space", "")),
        "normal_r": str(getattr(bake, "normal_r", "")),
        "normal_g": str(getattr(bake, "normal_g", "")),
        "normal_b": str(getattr(bake, "normal_b", "")),
        "objects": objects,
        "cameras": cameras,
        "lights": lights,
        "world": world,
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")
    print(f"event=dump_blend objects={len(objects)} cameras={len(cameras)} lights={len(lights)} output={output}")


if __name__ == "__main__":
    main()
