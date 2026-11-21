"""Weld a mesh on a grid so a render copy fits in GPU memory.

Each vertex moves to the mean of the vertices in its cell. One vertex at a
corner and the rest at the opposite corner can move by almost the full cell
diagonal, ``cell * sqrt(3)``. Callers set the cell from the export tolerance
and then measure the actual move. This does not replace the archived mesh.
"""

from __future__ import annotations

import numpy as np


def cluster_surface(vertices: np.ndarray, faces: np.ndarray, cell_m: float) -> dict:
    """Return a welded mesh and the maximum vertex move in metres."""
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    cell = float(cell_m)
    if cell <= 0.0 or not np.isfinite(cell):
        raise ValueError(f"reason=cell value={cell_m}")
    if points.ndim != 2 or points.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("reason=mesh_shape")
    shift = points.min(axis=0)
    quantized = np.floor((points - shift) / cell).astype(np.int64)
    span = quantized.max(axis=0) + 1
    code = quantized[:, 0] + span[0] * (quantized[:, 1] + span[1] * quantized[:, 2])
    _unique, inverse = np.unique(code, return_inverse=True)
    count = np.bincount(inverse)
    summed = np.zeros((_unique.shape[0], 3), dtype=np.float64)
    np.add.at(summed, inverse, points)
    welded = summed / count[:, None]
    move = np.linalg.norm(welded[inverse] - points, axis=1)
    bound = float(cell * np.sqrt(3.0))
    if len(move) and float(move.max()) > bound + 1e-8:
        raise ValueError(f"reason=cluster_move move_m={float(move.max())} bound_m={bound}")
    mapped = inverse[triangles]
    keep = (mapped[:, 0] != mapped[:, 1]) & (mapped[:, 0] != mapped[:, 2]) & (mapped[:, 1] != mapped[:, 2])
    return {
        "vertices": welded,
        "faces": mapped[keep],
        "max_move_m": float(move.max()) if len(move) else 0.0,
        "mean_move_m": float(move.mean()) if len(move) else 0.0,
        "input_vertices": int(len(points)),
        "output_vertices": int(len(welded)),
        "input_faces": int(len(triangles)),
        "output_faces": int(keep.sum()),
        "cell_m": cell,
        "bound_m": bound,
    }
