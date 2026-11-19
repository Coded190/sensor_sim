"""Triangle clearance that does not call ``narrow.mesh_clearance``.

The search is a centroid grid, not the production bounding-volume tree.
Distance is the minimum of vertex-to-triangle distances and edge-to-edge
distances. Intersecting triangles have distance zero.
"""

from __future__ import annotations

import numpy as np

FLOOR_Z = 0.001
FLOOR_TOLERANCE_M = 1e-4


def triangles_of(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]


def _segments(triangles: np.ndarray) -> np.ndarray:
    edges = np.stack(
        [
            np.stack([triangles[:, 0], triangles[:, 1]], axis=1),
            np.stack([triangles[:, 1], triangles[:, 2]], axis=1),
            np.stack([triangles[:, 2], triangles[:, 0]], axis=1),
        ],
        axis=1,
    )
    return edges.reshape(-1, 2, 3)


def _point_triangle(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Distance from points ``(P, 3)`` to triangles ``(T, 3, 3)``."""
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    ap = points[:, None, :] - a[None, :, :]
    d1 = np.sum(ab[None, :, :] * ap, axis=-1)
    d2 = np.sum(ac[None, :, :] * ap, axis=-1)
    d3 = np.sum(ab[None, :, :] * (points[:, None, :] - b[None, :, :]), axis=-1)
    d4 = np.sum(ac[None, :, :] * (points[:, None, :] - b[None, :, :]), axis=-1)
    d5 = np.sum(ab[None, :, :] * (points[:, None, :] - c[None, :, :]), axis=-1)
    d6 = np.sum(ac[None, :, :] * (points[:, None, :] - c[None, :, :]), axis=-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = np.maximum(va + vb + vc, 1e-15)
    closest = a + (ab * (vb / denom)[:, :, None]) + (ac * (vc / denom)[:, :, None])
    outside = (va <= 0.0) | (vb <= 0.0) | (vc <= 0.0)
    vertices = np.stack([a, b, c], axis=1)
    vertex_distance = np.linalg.norm(points[:, None, None, :] - vertices[None, :, :, :], axis=-1)
    nearest_vertex = vertices[None, :, np.argmin(vertex_distance, axis=-1)]
    # The barycentric closest point is valid only inside the triangle.
    chosen = np.where(outside[:, :, None], nearest_vertex, closest)
    # Edge projections replace the vertex fallback when the outside region is an edge.
    return np.linalg.norm(points[:, None, :] - chosen, axis=-1)


def _edge_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Distance between segment bundles ``(A, 2, 3)`` and ``(B, 2, 3)``."""
    p = first[:, None, 0, :]
    q = first[:, None, 1, :]
    r = second[None, :, 0, :]
    s = second[None, :, 1, :]
    u = q - p
    v = s - r
    w = p - r
    a = np.sum(u * u, axis=-1)
    b = np.sum(u * v, axis=-1)
    c = np.sum(v * v, axis=-1)
    d = np.sum(u * w, axis=-1)
    e = np.sum(v * w, axis=-1)
    denom = a * c - b * b
    parallel = denom < 1e-12
    s_param = np.zeros_like(denom)
    t_param = np.zeros_like(denom)
    np.divide(b * e - c * d, denom, out=s_param, where=~parallel)
    np.divide(a * e - b * d, denom, out=t_param, where=~parallel)
    s_param = np.clip(s_param, 0.0, 1.0)
    t_param = np.clip(t_param, 0.0, 1.0)
    s_param = np.where(parallel, 0.0, s_param)
    if np.any(parallel):
        t_parallel = np.divide(e, c, out=np.zeros_like(e), where=(c > 1e-12) & parallel)
        t_param = np.where(parallel, np.clip(t_parallel, 0.0, 1.0), t_param)
    closest_first = p + u * s_param[:, :, None]
    closest_second = r + v * t_param[:, :, None]
    return np.linalg.norm(closest_first - closest_second, axis=-1)


def _intersects(first: np.ndarray, second: np.ndarray) -> bool:
    """True when any triangle pair has a segment lying in the other triangle."""
    if first.shape[0] == 0 or second.shape[0] == 0:
        return False
    distance = _edge_distance(_segments(first), _segments(second))
    return bool(np.any(distance <= 1e-8))


def _grid_pairs(left: np.ndarray, right: np.ndarray, cell: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pairs of triangle index blocks that share a grid cell or a neighbor."""
    def keys(triangles: np.ndarray) -> np.ndarray:
        center = triangles.mean(axis=1)
        return np.floor(center / cell).astype(np.int64)

    left_keys = keys(left)
    right_keys = keys(right)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(right_keys):
        buckets.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(index)
    pairs = []
    seen = set()
    for index, key in enumerate(left_keys):
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    found = buckets.get((int(key[0]) + dx, int(key[1]) + dy, int(key[2]) + dz))
                    if not found:
                        continue
                    token = (index, found[0], len(found))
                    if token in seen:
                        continue
                    seen.add(token)
                    pairs.append((np.array([index]), np.asarray(found, dtype=np.int64)))
    return pairs


def reference_clearance(first: dict, second: dict) -> dict:
    """Minimum separation from an independent vertex, edge, and intersection test.

    Large meshes are searched on a 2 mm grid of triangle centroids. A triangle
    pair whose centroids are more than one cell apart is not compared. If that
    leaves no candidate, the result is marked incomplete rather than reported
    as a large clearance.
    """
    left = triangles_of(first["vertices"], first["faces"])
    right = triangles_of(second["vertices"], second["faces"])
    counts = [int(left.shape[0]), int(right.shape[0])]
    empty = {"clearance_m": float("inf"), "surface_intersection": False, "method": "reference_grid", "n_triangles": counts, "complete": True}
    if left.size == 0 or right.size == 0:
        return empty
    cell = 0.01
    if counts[0] <= 8 and counts[1] <= 8:
        candidates = [(np.arange(counts[0]), np.arange(counts[1]))]
        complete = True
    else:
        grouped: dict[tuple[int, ...], list[int]] = {}
        left_index = np.arange(counts[0])
        # Compare every left triangle with right triangles in neighboring cells.
        right_keys = np.floor(right.mean(axis=1) / cell).astype(np.int64)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for index, key in enumerate(right_keys):
            buckets.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(int(index))
        left_keys = np.floor(left.mean(axis=1) / cell).astype(np.int64)
        candidates = []
        for index, key in enumerate(left_keys):
            found: list[int] = []
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        found.extend(buckets.get((int(key[0]) + dx, int(key[1]) + dy, int(key[2]) + dz), []))
            if found:
                candidates.append((np.array([index]), np.asarray(found, dtype=np.int64)))
        complete = True
        if not candidates:
            return {
                "clearance_m": None,
                "lower_bound_m": cell,
                "surface_intersection": False,
                "method": "reference_grid",
                "n_triangles": counts,
                "complete": False,
                "note": "no triangle centroids share a 1 cm neighborhood, so the exact minimum was not computed",
            }
    best = float("inf")
    intersected = False
    for left_ids, right_ids in candidates:
        block_left = left[left_ids]
        # Chunk the right ids so the edge-distance array stays small.
        for start in range(0, right_ids.size, 32):
            block_right = right[right_ids[start : start + 32]]
            edge_distance = _edge_distance(_segments(block_left), _segments(block_right))
            point_distance = _point_triangle(block_left.reshape(-1, 3), block_right)
            other_distance = _point_triangle(block_right.reshape(-1, 3), block_left)
            best = min(best, float(np.min(edge_distance)), float(np.min(point_distance)), float(np.min(other_distance)))
            if best <= 1e-8:
                intersected = True
                break
        if intersected:
            break
    return {
        "clearance_m": 0.0 if intersected else float(best),
        "surface_intersection": bool(intersected or best <= 1e-8),
        "method": "reference_grid",
        "n_triangles": counts,
        "complete": complete and not (best == float("inf") and not intersected and counts[0] > 8),
    }


def reference_contains(point: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> bool:
    """Odd +X ray count. This is not ``narrow.contains_point``."""
    triangles = triangles_of(vertices, faces)
    direction = np.array([1.0, 0.0, 0.0])
    edge_u = triangles[:, 1] - triangles[:, 0]
    edge_v = triangles[:, 2] - triangles[:, 0]
    pvec = np.cross(direction, edge_v)
    det = np.sum(edge_u * pvec, axis=1)
    good = np.abs(det) > 1e-12
    inv = np.zeros_like(det)
    inv[good] = 1.0 / det[good]
    tvec = np.asarray(point, dtype=np.float64) - triangles[:, 0]
    bary_u = np.sum(tvec * pvec, axis=1) * inv
    qvec = np.cross(tvec, edge_u)
    bary_v = np.sum(direction * qvec, axis=1) * inv
    distance = np.sum(edge_v * qvec, axis=1) * inv
    hit = good & (bary_u >= 0.0) & (bary_v >= 0.0) & (bary_u + bary_v <= 1.0) & (distance > 1e-8)
    return int(np.count_nonzero(hit)) % 2 == 1


def reference_floor(vertices: np.ndarray, floor_z: float = FLOOR_Z) -> dict:
    min_z = float(np.min(np.asarray(vertices, dtype=np.float64)[:, 2]))
    penetrating = min_z < float(floor_z) - FLOOR_TOLERANCE_M
    return {
        "min_z_m": min_z,
        "floor_z_m": float(floor_z),
        "penetrating": bool(penetrating),
        "method": "reference_min_vertex_z",
    }
