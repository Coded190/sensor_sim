"""Triangle clearance, containment, and floor contact.

AABB overlap is not mesh penetration. A sampled vertex distance is not the
minimum separation of two triangle meshes. ``narrow_phase`` reports the
minimum triangle-to-triangle distance from a bounding-volume search.
"""

from __future__ import annotations

import numpy as np

FLOOR_Z = 0.001
FLOOR_TOLERANCE_M = 1e-4
_LEAF = 8


def _triangles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]


def _ray_hits_mesh(origin: np.ndarray, direction: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> int:
    triangles = _triangles(vertices, faces)
    edge_u = triangles[:, 1] - triangles[:, 0]
    edge_v = triangles[:, 2] - triangles[:, 0]
    pvec = np.cross(direction, edge_v)
    det = np.sum(edge_u * pvec, axis=1)
    good = np.abs(det) > 1e-12
    inv = np.zeros_like(det)
    inv[good] = 1.0 / det[good]
    tvec = origin - triangles[:, 0]
    bary_u = np.sum(tvec * pvec, axis=1) * inv
    qvec = np.cross(tvec, edge_u)
    bary_v = np.sum(direction * qvec, axis=1) * inv
    distance = np.sum(edge_v * qvec, axis=1) * inv
    hit = good & (bary_u >= 0.0) & (bary_v >= 0.0) & (bary_u + bary_v <= 1.0) & (distance > 1e-8)
    return int(np.count_nonzero(hit))


def contains_point(point: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> bool:
    """Odd intersection count along +X. Closed meshes only."""
    return _ray_hits_mesh(np.asarray(point, dtype=np.float64), np.array([1.0, 0.0, 0.0]), vertices, faces) % 2 == 1


def _point_triangles(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Distance from each point ``(P, 3)`` to each triangle ``(T, 3, 3)``."""
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    ap = points[:, None, :] - a[None, :, :]
    bp = points[:, None, :] - b[None, :, :]
    cp = points[:, None, :] - c[None, :, :]
    ab_b = ab[None, :, :]
    ac_b = ac[None, :, :]
    d1 = np.sum(ab_b * ap, axis=-1)
    d2 = np.sum(ac_b * ap, axis=-1)
    d3 = np.sum(ab_b * bp, axis=-1)
    d4 = np.sum(ac_b * bp, axis=-1)
    d5 = np.sum(ab_b * cp, axis=-1)
    d6 = np.sum(ac_b * cp, axis=-1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = np.maximum(va + vb + vc, 1e-15)
    closest = a[None, :, :] + ab_b * (vb / denom)[:, :, None] + ac_b * (vc / denom)[:, :, None]
    region_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    region_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    region_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    region_a = (d1 <= 0.0) & (d2 <= 0.0)
    region_b = (d3 >= 0.0) & (d4 <= d3)
    region_c = (d6 >= 0.0) & (d5 <= d6)
    ab_t = (d1 / np.maximum(d1 - d3, 1e-15))[:, :, None]
    ac_t = (d2 / np.maximum(d2 - d6, 1e-15))[:, :, None]
    bc_t = ((d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-15))[:, :, None]
    closest = np.where(region_ab[:, :, None], a[None, :, :] + ab_b * ab_t, closest)
    closest = np.where(region_ac[:, :, None], a[None, :, :] + ac_b * ac_t, closest)
    closest = np.where(region_bc[:, :, None], b[None, :, :] + (c - b)[None, :, :] * bc_t, closest)
    closest = np.where(region_a[:, :, None], a[None, :, :], closest)
    closest = np.where(region_b[:, :, None], b[None, :, :], closest)
    closest = np.where(region_c[:, :, None], c[None, :, :], closest)
    return np.linalg.norm(points[:, None, :] - closest, axis=-1)


def _segment_distances(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Distance between segments ``(A, 3)`` and segments ``(B, 3)``, shape ``(A, B)``."""
    d1 = (q1 - p1)[:, None, :]
    d2 = (q2 - p2)[None, :, :]
    r = p1[:, None, :] - p2[None, :, :]
    shape = (p1.shape[0], p2.shape[0])
    a = np.broadcast_to(np.sum(d1 * d1, axis=-1), shape).copy()
    e = np.broadcast_to(np.sum(d2 * d2, axis=-1), shape).copy()
    f = np.sum(d2 * r, axis=-1)
    b = np.sum(d1 * d2, axis=-1)
    c = np.sum(d1 * r, axis=-1)
    eps = 1e-14
    denom = a * e - b * b
    s = np.zeros_like(denom)
    usable = np.abs(denom) > eps
    s[usable] = np.clip((b[usable] * f[usable] - c[usable] * e[usable]) / denom[usable], 0.0, 1.0)
    t = np.zeros_like(denom)
    nonzero_e = e > eps
    t[nonzero_e] = (b[nonzero_e] * s[nonzero_e] + f[nonzero_e]) / e[nonzero_e]
    low = t < 0.0
    high = t > 1.0
    t = np.clip(t, 0.0, 1.0)
    s = np.where(low & (a > eps), np.clip(-c / np.maximum(a, eps), 0.0, 1.0), s)
    s = np.where(high & (a > eps), np.clip((b - c) / np.maximum(a, eps), 0.0, 1.0), s)
    s = np.where(a <= eps, 0.0, s)
    t = np.where(e <= eps, 0.0, t)
    t = np.where((a <= eps) & (e > eps), np.clip(f / np.maximum(e, eps), 0.0, 1.0), t)
    closest_a = p1[:, None, :] + d1 * s[:, :, None]
    closest_b = p2[None, :, :] + d2 * t[:, :, None]
    return np.linalg.norm(closest_a - closest_b, axis=-1)


def _edges_hit_triangles(origins: np.ndarray, ends: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """True where the segment from ``origins`` to ``ends`` meets a triangle."""
    direction = ends - origins
    base = triangles[:, 0]
    edge_u = triangles[:, 1] - base
    edge_v = triangles[:, 2] - base
    pvec = np.cross(direction[:, None, :], edge_v[None, :, :])
    det = np.sum(edge_u[None, :, :] * pvec, axis=-1)
    good = np.abs(det) > 1e-12
    inv = np.zeros_like(det)
    inv[good] = 1.0 / det[good]
    tvec = origins[:, None, :] - base[None, :, :]
    bary_u = np.sum(tvec * pvec, axis=-1) * inv
    qvec = np.cross(tvec, edge_u[None, :, :])
    bary_v = np.sum(direction[:, None, :] * qvec, axis=-1) * inv
    distance = np.sum(edge_v[None, :, :] * qvec, axis=-1) * inv
    return good & (bary_u >= -1e-8) & (bary_v >= -1e-8) & (bary_u + bary_v <= 1.0 + 1e-8) & (distance >= -1e-8) & (distance <= 1.0 + 1e-8)


def _pair_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Minimum distance for every triangle pair. ``left`` and ``right`` are ``(N, 3, 3)``."""
    distance = np.min(_point_triangles(left.reshape(-1, 3), right).reshape(left.shape[0], 3, right.shape[0]), axis=1)
    other = np.min(_point_triangles(right.reshape(-1, 3), left).reshape(right.shape[0], 3, left.shape[0]), axis=1)
    distance = np.minimum(distance, other.T)
    for start, end in ((0, 1), (1, 2), (2, 0)):
        distance = np.where(_edges_hit_triangles(left[:, start], left[:, end], right), 0.0, distance)
        distance = np.where(_edges_hit_triangles(right[:, start], right[:, end], left).T, 0.0, distance)
        for other_start, other_end in ((0, 1), (1, 2), (2, 0)):
            distance = np.minimum(
                distance,
                _segment_distances(left[:, start], left[:, end], right[:, other_start], right[:, other_end]),
            )
    return distance


def triangle_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Minimum distance between two triangles, including edge crossings."""
    return float(_pair_distances(first.reshape(1, 3, 3), second.reshape(1, 3, 3))[0, 0])


def _bounds(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))


def _box_distance(low_a, high_a, low_b, high_b) -> float:
    gap = np.maximum(np.maximum(low_a - high_b, low_b - high_a), 0.0)
    return float(np.linalg.norm(gap))


def _build(triangles: np.ndarray, indices: np.ndarray) -> dict:
    chosen = triangles[indices]
    low, high = _bounds(chosen)
    node = {"low": low, "high": high, "indices": None, "left": None, "right": None}
    if indices.size <= _LEAF:
        node["indices"] = indices
        return node
    centroids = chosen.mean(axis=1)
    axis = int(np.argmax(high - low))
    order = indices[np.argsort(centroids[:, axis], kind="mergesort")]
    mid = order.size // 2
    node["left"] = _build(triangles, order[:mid])
    node["right"] = _build(triangles, order[mid:])
    return node


def _search(left_node: dict, right_node: dict, left: np.ndarray, right: np.ndarray, best: list[float]) -> None:
    if _box_distance(left_node["low"], left_node["high"], right_node["low"], right_node["high"]) >= best[0]:
        return
    if left_node["indices"] is not None and right_node["indices"] is not None:
        distance = float(np.min(_pair_distances(left[left_node["indices"]], right[right_node["indices"]])))
        if distance < best[0]:
            best[0] = distance
        return
    if left_node["indices"] is None and right_node["indices"] is None:
        pairs = (
            (left_node["left"], right_node["left"]),
            (left_node["left"], right_node["right"]),
            (left_node["right"], right_node["left"]),
            (left_node["right"], right_node["right"]),
        )
        pairs = sorted(pairs, key=lambda item: _box_distance(item[0]["low"], item[0]["high"], item[1]["low"], item[1]["high"]))
        for child_left, child_right in pairs:
            if best[0] <= 1e-12:
                return
            _search(child_left, child_right, left, right, best)
        return
    if left_node["indices"] is None:
        _search(left_node["left"], right_node, left, right, best)
        _search(left_node["right"], right_node, left, right, best)
        return
    _search(left_node, right_node["left"], left, right, best)
    _search(left_node, right_node["right"], left, right, best)


def _surfaces_intersect(left: np.ndarray, right: np.ndarray, left_node: dict, right_node: dict) -> bool:
    """True when any triangle pair is within 1e-8 m. Far boxes are not tested."""
    stack = [(left_node, right_node)]
    while stack:
        first_node, second_node = stack.pop()
        if _box_distance(first_node["low"], first_node["high"], second_node["low"], second_node["high"]) > 1e-8:
            continue
        both_leaves = first_node["indices"] is not None and second_node["indices"] is not None
        if both_leaves:
            if float(np.min(_pair_distances(left[first_node["indices"]], right[second_node["indices"]]))) <= 1e-8:
                return True
            continue
        if first_node["indices"] is None and second_node["indices"] is None:
            stack.extend(
                (
                    (first_node["left"], second_node["left"]),
                    (first_node["left"], second_node["right"]),
                    (first_node["right"], second_node["left"]),
                    (first_node["right"], second_node["right"]),
                )
            )
        elif first_node["indices"] is None:
            stack.append((first_node["left"], second_node))
            stack.append((first_node["right"], second_node))
        else:
            stack.append((first_node, second_node["left"]))
            stack.append((first_node, second_node["right"]))
    return False


def mesh_clearance(first: dict, second: dict) -> dict:
    """Exact minimum triangle distance for the meshes that reach a leaf."""
    left = _triangles(first["vertices"], first["faces"])
    right = _triangles(second["vertices"], second["faces"])
    if left.size == 0 or right.size == 0:
        return {"clearance_m": float("inf"), "surface_intersection": False, "method": "bvh_triangle_distance"}
    left_node = _build(left, np.arange(left.shape[0]))
    right_node = _build(right, np.arange(right.shape[0]))
    stride = max(1, left.reshape(-1, 3).shape[0] // 64)
    samples = left.reshape(-1, 3)[::stride]
    upper = float(np.min(np.min(_point_triangles(samples, right), axis=1)))
    best = [upper]
    _search(left_node, right_node, left, right, best)
    clearance = float(best[0])
    return {
        "clearance_m": clearance,
        "surface_intersection": bool(clearance <= 1e-8),
        "method": "bvh_triangle_distance",
        "n_triangles": [int(left.shape[0]), int(right.shape[0])],
    }


def surface_gap(first: dict, second: dict) -> float:
    """Minimum triangle-to-triangle distance. This is not a vertex sample."""
    return float(mesh_clearance(first, second)["clearance_m"])


def floor_status(vertices: np.ndarray, floor_z: float = FLOOR_Z) -> dict:
    """Contact at ``z = 0.001`` is allowed. A vertex below that tolerance penetrates."""
    min_z = float(np.min(np.asarray(vertices, dtype=np.float64)[:, 2]))
    penetrating = min_z < float(floor_z) - FLOOR_TOLERANCE_M
    return {
        "min_z_m": min_z,
        "floor_z_m": float(floor_z),
        "tolerance_m": FLOOR_TOLERANCE_M,
        "penetrating": bool(penetrating),
        "permitted_contact": bool(not penetrating and min_z <= float(floor_z) + FLOOR_TOLERANCE_M),
    }


def narrow_phase(first: dict, second: dict, gap: float) -> dict:
    """Intersection, containment, and continuous clearance.

    A vertex sample is not the separation decision. Boxes that are already at
    least ``gap`` apart are a lower bound: the surfaces are at least that far.
    Overlapping boxes still run the triangle query.
    """
    left = _triangles(first["vertices"], first["faces"])
    right = _triangles(second["vertices"], second["faces"])
    centroid = np.asarray(first["vertices"], dtype=np.float64).mean(axis=0)
    other = np.asarray(second["vertices"], dtype=np.float64).mean(axis=0)
    inside = contains_point(centroid, second["vertices"], second["faces"])
    inside_back = contains_point(other, first["vertices"], first["faces"])
    contained = bool(inside or inside_back)
    low_a, high_a = _bounds(left)
    low_b, high_b = _bounds(right)
    overlap = bool(np.all(low_a <= high_b) and np.all(low_b <= high_a))
    box_gap = _box_distance(low_a, high_a, low_b, high_b)
    left_node = _build(left, np.arange(left.shape[0]))
    right_node = _build(right, np.arange(right.shape[0]))
    intersecting = _surfaces_intersect(left, right, left_node, right_node)
    if intersecting:
        clearance = 0.0
        exact = True
        method = "bvh_triangle_distance"
    elif contained:
        clearance = box_gap
        exact = False
        method = "contained_without_surface_intersection"
    elif box_gap + 1e-12 >= float(gap):
        clearance = box_gap
        exact = False
        method = "aabb_lower_bound"
    else:
        found = mesh_clearance(first, second)
        clearance = found["clearance_m"]
        exact = True
        method = found["method"]
    separated = (not contained) and (not intersecting) and float(clearance) + 1e-6 >= float(gap)
    return {
        "contained": contained,
        "clearance_m": float(clearance),
        "clearance_exact": exact,
        "required_gap_m": float(gap),
        "separated": bool(separated),
        "surface_intersection": bool(intersecting),
        "aabb_overlap": overlap,
        "method": method,
    }


def aabb_overlap(first: dict, second: dict) -> bool:
    low_a, high_a = _bounds(_triangles(first["vertices"], first["faces"]))
    low_b, high_b = _bounds(_triangles(second["vertices"], second["faces"]))
    return bool(np.all(low_a <= high_b) and np.all(low_b <= high_a))
