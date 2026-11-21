"""Dump evaluated light pose, shape, radius, spread, and sides. Does not save the blend."""

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
        if obj.type != "LIGHT":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        matrix = np.array(evaluated.matrix_world, dtype=np.float64)
        light = obj.data
        records.append({
            "name": obj.name,
            "light_type": str(light.type),
            "energy": float(light.energy),
            "color": [float(channel) for channel in light.color],
            "color_space": "linear",
            "size": float(getattr(light, "size", 0.0)),
            "size_y": float(getattr(light, "size_y", getattr(light, "size", 0.0))),
            "shape": str(getattr(light, "shape", "")),
            "spread_rad": float(getattr(light, "spread", 3.141592653589793)),
            "shadow_soft_size": float(getattr(light, "shadow_soft_size", 0.0)),
            "two_sided": False,
            "emitting_side": "local_negative_z" if light.type == "AREA" else "omnidirectional",
            "parent": None if obj.parent is None else obj.parent.name,
            "object_scale": [float(value) for value in obj.scale],
            "evaluated_matrix": matrix.tolist(),
            "evaluated_translation": matrix[:3, 3].tolist(),
            "evaluated_axis_scale": [
                float(np.linalg.norm(matrix[0, :3])),
                float(np.linalg.norm(matrix[1, :3])),
                float(np.linalg.norm(matrix[2, :3])),
            ],
        })
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")
    print(f"event=light_evaluation count={len(records)}")


if __name__ == "__main__":
    main()
