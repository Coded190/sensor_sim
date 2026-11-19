"""Associate exported meshes with the shape-index AOV of one render."""

from __future__ import annotations

import numpy as np

from sensor_sim.scene_gen.checks import integer_index
from sensor_sim.scene_gen.convention import SHAPE_INDEX_TOL


def _project(sensor, point: np.ndarray, width: int, height: int) -> tuple[int, int] | None:
    import mitsuba as mi

    transformed = sensor.world_transform().inverse() @ mi.Point3f(float(point[0]), float(point[1]), float(point[2]))
    ndc = np.array(sensor.projection_transform() @ transformed).reshape(-1)
    if not np.isfinite(ndc).all():
        return None
    column = int(np.floor(float(ndc[0]) * width))
    row = int(np.floor(float(ndc[1]) * height))
    if not (0 <= row < height and 0 <= column < width):
        return None
    return row, column


def _samples(shape, limit: int = 2000) -> np.ndarray:
    count = int(shape.vertex_count())
    step = max(1, count // limit)
    return np.stack([np.array(shape.vertex_position(i)).reshape(-1)[:3] for i in range(0, count, step)])


def _triangle_distance(point: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> float:
    """Closest distance from one point to a small set of triangles."""
    if faces.size == 0:
        return float("inf")
    step = max(1, faces.shape[0] // 400)
    triangles = vertices[faces[::step]]
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    unit = normal / np.maximum(length, 1e-15)
    signed = np.sum((point - a) * unit, axis=1)
    projected = point - signed[:, None] * unit
    # Reject projections outside the triangle and use the nearest vertex instead.
    v0 = ab
    v1 = ac
    v2 = projected - a
    dot00 = np.sum(v0 * v0, axis=1)
    dot01 = np.sum(v0 * v1, axis=1)
    dot11 = np.sum(v1 * v1, axis=1)
    dot02 = np.sum(v0 * v2, axis=1)
    dot12 = np.sum(v1 * v2, axis=1)
    denom = np.maximum(dot00 * dot11 - dot01 * dot01, 1e-15)
    bary_u = (dot11 * dot02 - dot01 * dot12) / denom
    bary_v = (dot00 * dot12 - dot01 * dot02) / denom
    inside = (bary_u >= 0.0) & (bary_v >= 0.0) & (bary_u + bary_v <= 1.0)
    distances = np.linalg.norm(np.stack([a, b, c], axis=1) - point, axis=-1).min(axis=1)
    distances[inside] = np.abs(signed[inside])
    return float(np.min(distances))


def indices_for_shape(
    shape,
    index_image: np.ndarray,
    position: np.ndarray,
    sensor,
    width: int,
    height: int,
    others: list[dict] | None = None,
) -> list[int]:
    """The integer shape index painted on this mesh's own visible vertices.

    A vertex counts only when the pixel's hit is closer to that vertex than to
    any other mesh. Two strong answers are rejected instead of guessed.
    """
    points = _samples(shape)
    others = [] if others is None else others
    votes: dict[int, int] = {}
    for point in points:
        pixel = _project(sensor, point, width, height)
        if pixel is None:
            continue
        row, column = pixel
        hit = position[row, column]
        distance = float(np.linalg.norm(hit - point))
        if distance > 0.03:
            continue
        other_distance = min(
            (_triangle_distance(hit, mesh["vertices"], mesh["faces"]) for mesh in others),
            default=np.inf,
        )
        # The hit is on another surface, or the two surfaces cannot be separated.
        if other_distance <= distance + 0.001:
            continue
        value = float(index_image[row, column])
        if abs(value - round(value)) >= SHAPE_INDEX_TOL:
            continue
        key = int(round(value))
        votes[key] = votes.get(key, 0) + 1
    if not votes:
        return []
    ordered = sorted(votes.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) > 1 and ordered[1][1] > 0.25 * ordered[0][1]:
        raise RuntimeError(f"ambiguous shape index for {shape.id()}: {ordered[:4]}")
    return [ordered[0][0]]


def paint_ids(index_image: np.ndarray, mapping: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Object and material images from resolved indices. Unresolved pixels stay 0."""
    object_ids = np.zeros(index_image.shape, dtype=np.float32)
    material_ids = np.zeros(index_image.shape, dtype=np.float32)
    claimed: dict[int, dict] = {}
    for item in mapping:
        for shape_index in item["shape_indices"]:
            if shape_index in claimed and claimed[shape_index]["object_id"] != item["object_id"]:
                raise RuntimeError(f"shape index {shape_index} belongs to two objects")
            claimed[shape_index] = item
    for shape_index, item in claimed.items():
        match = integer_index(index_image, shape_index)
        object_ids[match] = float(item["object_index"])
        material_ids[match] = float(item["material_id"])
    return object_ids, material_ids


def ray_triangle_ids(origins: np.ndarray, directions: np.ndarray, meshes: list[dict]) -> np.ndarray:
    """Nearest triangle hit per ray. This is the reference, not the AOV."""
    ids = np.zeros(origins.shape[0], dtype=np.int32)
    best = np.full(origins.shape[0], np.inf)
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    for mesh in meshes:
        vertices = np.asarray(mesh["vertices"], dtype=np.float64)
        faces = np.asarray(mesh["faces"], dtype=np.int64)
        triangles = vertices[faces]
        edges_u = triangles[:, 1] - triangles[:, 0]
        edges_v = triangles[:, 2] - triangles[:, 0]
        for ray in range(origins.shape[0]):
            direction = directions[ray]
            origin = origins[ray]
            pvec = np.cross(direction, edges_v)
            det = np.sum(edges_u * pvec, axis=1)
            good = np.abs(det) > 1e-12
            inv = np.zeros_like(det)
            inv[good] = 1.0 / det[good]
            tvec = origin - triangles[:, 0]
            u = np.sum(tvec * pvec, axis=1) * inv
            qvec = np.cross(tvec, edges_u)
            v = np.sum(direction * qvec, axis=1) * inv
            t = np.sum(edges_v * qvec, axis=1) * inv
            hit = good & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-5) & (t < best[ray])
            if np.any(hit):
                best[ray] = float(np.min(t[hit]))
                ids[ray] = int(mesh["object_index"])
    return ids
