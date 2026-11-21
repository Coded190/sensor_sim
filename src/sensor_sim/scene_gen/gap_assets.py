"""Parameterized meshes for categories the Infinigen export inventory does not contain.

These are not the finished form of a category that Infinigen already exported.
Silhouette features are mesh geometry. A normal map is not used to invent them.
"""

from __future__ import annotations

import numpy as np


def _box(minimum, maximum) -> tuple[np.ndarray, list[list[int]]]:
    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    vertices = np.array(
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        dtype=np.float64,
    )
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return vertices, faces


def _append(vertices, faces, materials, piece_vertices, piece_faces, material: str) -> None:
    offset = len(vertices)
    vertices.extend(piece_vertices.tolist())
    for face in piece_faces:
        faces.append([index + offset for index in face])
        materials.append(material)


def _mesh(vertices, faces, materials) -> dict:
    corners = [[(index, None, None) for index in face] for face in faces]
    return {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "corners": corners,
        "face_materials": materials,
    }


def chair(seed: int) -> dict:
    """Seat, curved-section back, and four tapered legs. Variant 1 is wider."""
    wide = 0.08 if seed % 2 else 0.0
    vertices: list = []
    faces: list = []
    materials: list = []
    _append(vertices, faces, materials, *_box([0.06, 0.06, 0.0], [0.10, 0.10, 0.42]), "leg")
    _append(vertices, faces, materials, *_box([0.36 + wide, 0.06, 0.0], [0.40 + wide, 0.10, 0.42]), "leg")
    _append(vertices, faces, materials, *_box([0.06, 0.36, 0.0], [0.10, 0.40, 0.42]), "leg")
    _append(vertices, faces, materials, *_box([0.36 + wide, 0.36, 0.0], [0.40 + wide, 0.40, 0.42]), "leg")
    _append(vertices, faces, materials, *_box([0.04, 0.04, 0.42], [0.42 + wide, 0.42, 0.50]), "seat")
    _append(vertices, faces, materials, *_box([0.06, 0.36, 0.50], [0.40 + wide, 0.44, 0.92]), "backrest")
    return _mesh(vertices, faces, materials)


def monitor(seed: int) -> dict:
    tall = 0.06 if seed % 2 else 0.0
    vertices: list = []
    faces: list = []
    materials: list = []
    _append(vertices, faces, materials, *_box([0.08, 0.0, 0.0], [0.46, 0.18, 0.02]), "stand")
    _append(vertices, faces, materials, *_box([0.24, 0.04, 0.02], [0.30, 0.08, 0.16]), "neck")
    _append(vertices, faces, materials, *_box([0.0, 0.02, 0.16], [0.54, 0.06, 0.46 + tall]), "bezel")
    _append(vertices, faces, materials, *_box([0.03, 0.0, 0.19], [0.51, 0.02, 0.43 + tall]), "screen")
    return _mesh(vertices, faces, materials)


def television(seed: int) -> dict:
    return monitor(seed + 3)


def bed(seed: int) -> dict:
    vertices: list = []
    faces: list = []
    materials: list = []
    _append(vertices, faces, materials, *_box([0.05, 0.05, 0.0], [0.12, 0.12, 0.25]), "leg")
    _append(vertices, faces, materials, *_box([1.48, 1.88, 0.0], [1.55, 1.95, 0.25]), "leg")
    _append(vertices, faces, materials, *_box([0.0, 0.15, 0.25], [1.6, 2.0, 0.45]), "mattress")
    _append(vertices, faces, materials, *_box([0.0, 1.75, 0.45], [1.6, 2.05, 0.95]), "headboard")
    return _mesh(vertices, faces, materials)


def refrigerator(seed: int) -> dict:
    vertices: list = []
    faces: list = []
    materials: list = []
    _append(vertices, faces, materials, *_box([0.02, 0.04, 0.0], [0.68, 0.66, 1.7]), "body")
    _append(vertices, faces, materials, *_box([0.04, 0.0, 0.08], [0.66, 0.04, 1.62]), "door")
    _append(vertices, faces, materials, *_box([0.58, 0.0, 0.85], [0.62, 0.02, 1.05]), "handle")
    return _mesh(vertices, faces, materials)


def microwave(seed: int) -> dict:
    vertices: list = []
    faces: list = []
    materials: list = []
    _append(vertices, faces, materials, *_box([0.0, 0.04, 0.0], [0.5, 0.35, 0.3]), "body")
    _append(vertices, faces, materials, *_box([0.04, 0.0, 0.04], [0.36, 0.04, 0.26]), "door")
    _append(vertices, faces, materials, *_box([0.40, 0.0, 0.08], [0.46, 0.03, 0.22]), "handle")
    return _mesh(vertices, faces, materials)


def cabinet(seed: int) -> dict:
    vertices: list = []
    faces: list = []
    materials: list = []
    _append(vertices, faces, materials, *_box([0.0, 0.04, 0.0], [0.8, 0.42, 0.9]), "body")
    _append(vertices, faces, materials, *_box([0.04, 0.0, 0.04], [0.76, 0.04, 0.86]), "door")
    return _mesh(vertices, faces, materials)


def nightstand(seed: int) -> dict:
    return cabinet(seed)


def wardrobe(seed: int) -> dict:
    vertices: list = []
    faces: list = []
    materials: list = []
    _append(vertices, faces, materials, *_box([0.0, 0.05, 0.0], [1.1, 0.58, 1.9]), "body")
    _append(vertices, faces, materials, *_box([0.04, 0.0, 0.04], [1.06, 0.05, 1.86]), "door")
    return _mesh(vertices, faces, materials)


def keyboard(seed: int) -> dict:
    vertices, faces = _box([0.0, 0.0, 0.0], [0.42, 0.14, 0.02])
    return _mesh(vertices.tolist(), faces, ["body"] * len(faces))


def mouse(seed: int) -> dict:
    vertices, faces = _box([0.0, 0.0, 0.0], [0.06, 0.11, 0.03])
    return _mesh(vertices.tolist(), faces, ["body"] * len(faces))


def toolbox(seed: int) -> dict:
    vertices: list = []
    faces: list = []
    materials: list = []
    _append(vertices, faces, materials, *_box([0.0, 0.0, 0.0], [0.4, 0.22, 0.18]), "body")
    _append(vertices, faces, materials, *_box([0.12, 0.02, 0.18], [0.28, 0.20, 0.24]), "handle")
    return _mesh(vertices, faces, materials)


BUILDERS = {
    "chair": chair,
    "monitor": monitor,
    "television": television,
    "bed": bed,
    "refrigerator": refrigerator,
    "microwave": microwave,
    "cabinet": cabinet,
    "nightstand": nightstand,
    "wardrobe": wardrobe,
    "keyboard": keyboard,
    "mouse": mouse,
    "toolbox": toolbox,
}
