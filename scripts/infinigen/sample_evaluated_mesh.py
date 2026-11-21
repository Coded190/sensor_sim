"""Sample evaluated world positions. Does not save the blend."""

from __future__ import annotations

import json
import sys

import bpy
import numpy as np


def main() -> None:
    blend = sys.argv[sys.argv.index("--") + 1]
    output = sys.argv[sys.argv.index("--") + 2]
    bpy.ops.wm.open_mainfile(filepath=blend)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    records = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.hide_render or len(obj.data.vertices) < 3:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        if mesh is None or len(mesh.vertices) < 3 or len(mesh.polygons) < 1:
            evaluated.to_mesh_clear()
            continue
        matrix = np.array(evaluated.matrix_world, dtype=np.float64)
        count = len(mesh.vertices)
        step = max(1, count // 24)
        indices = list(range(0, count, step))[:24]
        local = np.array([mesh.vertices[index].co for index in indices], dtype=np.float64)
        ones = np.ones((len(local), 1))
        world = np.concatenate([local, ones], axis=1) @ matrix.T
        step_face = max(1, int(len(mesh.polygons) // 8))
        triangles = []
        for polygon in list(mesh.polygons)[::step_face][:8]:
            corners = [int(vertex) for vertex in polygon.vertices[:3]]
            if len(corners) < 3:
                continue
            local = np.array([mesh.vertices[index].co for index in corners], dtype=np.float64)
            world_triangle = np.concatenate([local, np.ones((3, 1))], axis=1) @ matrix.T
            geometric = np.cross(world_triangle[1, :3] - world_triangle[0, :3], world_triangle[2, :3] - world_triangle[0, :3])
            length = float(np.linalg.norm(geometric))
            if length < 1e-12:
                continue
            triangles.append({
                "world": world_triangle[:, :3].round(6).tolist(),
                "normal": (geometric / length).round(6).tolist(),
            })
        records.append({
            "name": obj.name,
            "vertices": int(count),
            "polygons": int(len(mesh.polygons)),
            "slots": [None if slot.material is None else slot.material.name for slot in obj.material_slots],
            "world_samples": world[:, :3].round(6).tolist(),
            "triangles": triangles,
        })
        evaluated.to_mesh_clear()
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(records, handle)
        handle.write("\n")
    print(f"event=sampled_meshes count={len(records)}")


if __name__ == "__main__":
    main()
