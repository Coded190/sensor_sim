"""Deterministic OBJ read and write for indoor assets.

Coordinates are meters. Floats are written with nine digits after the decimal.
The writer emits no timestamps and no random identifiers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OBJ_DECIMALS = 9


def quantize(values: np.ndarray, decimals: int = OBJ_DECIMALS) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.round(array, decimals)


def write_obj(vertices: np.ndarray, polygons: list[list[int]], *, name: str = "mesh") -> bytes:
    """Serialize one mesh. Vertex order, face order, and winding are preserved."""
    vertices = quantize(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    lines = [f"# sensor_sim indoor obj 1 name={name}", f"o {name}"]
    for vertex in vertices:
        lines.append(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}")
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError("a face needs at least three vertices")
        indexes = " ".join(str(int(index) + 1) for index in polygon)
        lines.append(f"f {indexes}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def read_obj(path: Path | bytes | str) -> tuple[np.ndarray, list[list[int]]]:
    """Read vertices and polygonal faces. Texture and normal indices are ignored."""
    if isinstance(path, bytes):
        text = path.decode("utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    vertices: list[list[float]] = []
    polygons: list[list[int]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "v":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif kind == "f":
            polygon = []
            for token in parts[1:]:
                index = int(token.split("/")[0])
                polygon.append(index - 1 if index > 0 else len(vertices) + index)
            polygons.append(polygon)
    if not vertices:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.asarray(vertices, dtype=np.float64), polygons
