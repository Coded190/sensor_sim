"""Dump evaluated world bounds. Does not save the blend."""

from __future__ import annotations

import json
import sys

import bpy
import numpy as np


def main() -> None:
    blend = sys.argv[sys.argv.index("--") + 1]
    output = sys.argv[sys.argv.index("--") + 2]
    names = sys.argv[sys.argv.index("--") + 3 :]
    bpy.ops.wm.open_mainfile(filepath=blend)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    records = []
    for obj in bpy.data.objects:
        if names and not any(token in obj.name for token in names):
            continue
        if obj.type == "LIGHT":
            evaluated = obj.evaluated_get(depsgraph)
            matrix = np.array(evaluated.matrix_world, dtype=np.float64)
            records.append({
                "name": obj.name,
                "type": "LIGHT",
                "light_type": obj.data.type,
                "energy": float(obj.data.energy),
                "size": float(getattr(obj.data, "size", 0.0)),
                "size_y": float(getattr(obj.data, "size_y", getattr(obj.data, "size", 0.0))),
                "parent": None if obj.parent is None else obj.parent.name,
                "matrix_world": matrix.tolist(),
                "unevaluated_translation": [float(value) for value in obj.matrix_world.translation],
                "evaluated_translation": matrix[:3, 3].tolist(),
            })
            continue
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        if mesh is None or len(mesh.vertices) == 0:
            continue
        matrix = np.array(evaluated.matrix_world, dtype=np.float64)
        points = np.array([vertex.co for vertex in mesh.vertices], dtype=np.float64)
        world = np.concatenate([points, np.ones((len(points), 1))], axis=1) @ matrix.T
        world = world[:, :3]
        records.append({
            "name": obj.name,
            "vertices": int(len(points)),
            "parent": None if obj.parent is None else obj.parent.name,
            "matrix_world": matrix.tolist(),
            "parent_inverse": np.array(obj.matrix_parent_inverse, dtype=np.float64).tolist() if obj.parent else None,
            "scale": [float(value) for value in obj.scale],
            "world_min": world.min(axis=0).tolist(),
            "world_max": world.max(axis=0).tolist(),
            "local_min": points.min(axis=0).tolist(),
            "local_max": points.max(axis=0).tolist(),
            "landmark_local": points[0].tolist(),
            "landmark_world": world[0].tolist(),
        })
        evaluated.to_mesh_clear()
    Path = __import__("pathlib").Path
    Path(output).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"event=evaluated_bounds count={len(records)} output={output}")


if __name__ == "__main__":
    main()
