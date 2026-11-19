"""Blender side of the indoor transform contract.

Run with:

    blender --background --python assets/blender/create_indoor_scene.py -- report.json

The default mode writes Blender ``matrix_world`` values and world-space vertex
positions for known local vertices. A JSON input with ``mode`` = ``build`` creates
one object per supplied mesh and writes the same world-space record. Blender's
matrix is row-major, multiplies column vectors, and keeps translation in the
last column. Scene generation itself lives in ``sensor_sim.scene_gen.structural``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import mathutils


def _output_path() -> Path:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("missing -- report.json")
    return Path(argv[argv.index("--") + 1])


def _matrix_rows(matrix: mathutils.Matrix) -> list[list[float]]:
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def _world_vertices(obj) -> list[list[float]]:
    matrix = obj.matrix_world
    return [[float(value) for value in matrix @ vertex.co] for vertex in obj.data.vertices]


def _new_mesh(name: str, vertices, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _case(obj, name: str, expect: str) -> dict:
    bpy.context.view_layer.update()
    return {
        "name": name,
        "expect": expect,
        "matrix": _matrix_rows(obj.matrix_world),
        "world_vertices": _world_vertices(obj),
    }


def write_probe(output: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    local = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0)]
    obj = _new_mesh("triangle", local, [(0, 1, 2)])
    cases = []
    obj.location = (1.0, -2.0, 0.5)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    cases.append(_case(obj, "translation", "accept"))
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_euler = (0.4, 0.0, 0.0)
    cases.append(_case(obj, "rotation_x", "accept"))
    obj.rotation_euler = (0.0, -0.3, 0.0)
    cases.append(_case(obj, "rotation_y", "accept"))
    obj.rotation_euler = (0.0, 0.0, 0.8)
    cases.append(_case(obj, "rotation_z", "accept"))
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.5, 0.75, 2.0)
    cases.append(_case(obj, "nonuniform_scale", "accept"))
    obj.scale = (-1.0, 1.0, 1.0)
    cases.append(_case(obj, "negative_scale", "accept"))
    parent = bpy.data.objects.new("parent", None)
    bpy.context.collection.objects.link(parent)
    parent.scale = (2.0, 1.0, 1.0)
    obj.parent = parent
    obj.scale = (1.0, 1.0, 1.0)
    obj.rotation_euler = (0.0, 0.0, 0.7853981633974483)
    cases.append(_case(obj, "parent_nonuniform_scale_local_rotation", "reject_shear"))
    output.write_text(json.dumps({"local_vertices": local, "cases": cases}) + "\n", encoding="utf-8")


def build_meshes(payload: dict, output: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    records = []
    for mesh in payload["meshes"]:
        obj = _new_mesh(mesh["name"], mesh["vertices"], mesh["faces"])
        matrix = mesh.get("matrix")
        if matrix is not None:
            obj.matrix_world = mathutils.Matrix(matrix)
        records.append(_case(obj, mesh["name"], "accept"))
    output.write_text(json.dumps({"records": records}) + "\n", encoding="utf-8")


def main() -> None:
    output = _output_path()
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("mode") == "build":
            build_meshes(payload, output)
            return
    output.parent.mkdir(parents=True, exist_ok=True)
    write_probe(output)


if __name__ == "__main__":
    main()
