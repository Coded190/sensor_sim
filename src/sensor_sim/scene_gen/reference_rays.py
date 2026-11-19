"""Ray-triangle ids from exported meshes only.

This module does not read shape-index AOVs and does not call the identity
voter. It is the scoring reference for the shared-geometry scenes.
"""

from __future__ import annotations

import numpy as np


def _hit_distance(origins: np.ndarray, directions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Nearest positive hit distance for each ray. Misses are infinite."""
    edge_u = triangles[:, 1] - triangles[:, 0]
    edge_v = triangles[:, 2] - triangles[:, 0]
    nearest = np.full(origins.shape[0], np.inf)
    for start in range(0, origins.shape[0], 64):
        stop = min(start + 64, origins.shape[0])
        origin = origins[start:stop]
        direction = directions[start:stop]
        pvec = np.cross(direction[:, None, :], edge_v[None, :, :])
        det = np.sum(edge_u[None, :, :] * pvec, axis=-1)
        good = np.abs(det) > 1e-12
        inv = np.zeros_like(det)
        inv[good] = 1.0 / det[good]
        tvec = origin[:, None, :] - triangles[None, :, 0, :]
        bary_u = np.sum(tvec * pvec, axis=-1) * inv
        qvec = np.cross(tvec, edge_u[None, :, :])
        bary_v = np.sum(direction[:, None, :] * qvec, axis=-1) * inv
        distance = np.sum(edge_v[None, :, :] * qvec, axis=-1) * inv
        hit = good & (bary_u >= 0.0) & (bary_v >= 0.0) & (bary_u + bary_v <= 1.0) & (distance > 1e-5)
        distance = np.where(hit, distance, np.inf)
        nearest[start:stop] = np.min(distance, axis=1)
    return nearest


def reference_ray_ids(origins: np.ndarray, directions: np.ndarray, meshes: list[dict]) -> np.ndarray:
    """Nearest exported triangle per ray. ``object_index`` comes from the mesh record."""
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    ids = np.zeros(origins.shape[0], dtype=np.int32)
    best = np.full(origins.shape[0], np.inf)
    for mesh in meshes:
        triangles = np.asarray(mesh["vertices"], dtype=np.float64)[np.asarray(mesh["faces"], dtype=np.int64)]
        if triangles.size == 0:
            continue
        distance = _hit_distance(origins, directions, triangles)
        closer = distance < best
        ids[closer] = int(mesh["object_index"])
        best[closer] = distance[closer]
    return ids
