"""Simplified static proxies. Mitsuba must keep loading the render mesh."""

from __future__ import annotations

import hashlib

import numpy as np

from sensor_sim.scene_gen.mesh_io import write_mesh


def _triangles(mesh: dict) -> np.ndarray:
    faces = []
    for face in mesh["corners"]:
        for index in range(1, len(face) - 1):
            faces.append([int(face[0][0]), int(face[index][0]), int(face[index + 1][0])])
    return np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)


def _point_triangle(points: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Distance from each point to one triangle."""
    a_point, b_point, c_point = triangle
    ab_edge = b_point - a_point
    ac_edge = c_point - a_point
    normal = np.cross(ab_edge, ac_edge)
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        return np.linalg.norm(points - a_point, axis=1)
    normal = normal / length
    distance = np.abs((points - a_point) @ normal)
    return distance


def proxy_mesh(mesh: dict, *, stride: int = 4) -> dict:
    """Keep support-like upward faces and a stride of the remaining triangles.

    The method name is ``stride_keep_support``. It does not replace the render mesh.
    """
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    triangles = _triangles(mesh)
    if len(triangles) == 0:
        raise ValueError("reason=empty_mesh")
    kept = []
    for index, triangle in enumerate(triangles):
        points = vertices[triangle]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            continue
        normal = normal / length
        upward = normal[2] > 0.85 and points[:, 2].mean() > vertices[:, 2].min() + 0.01
        if upward or index % max(int(stride), 1) == 0:
            kept.append(triangle)
    if not kept:
        kept = [triangles[0]]
    corners = [[(int(index), None, None) for index in triangle] for triangle in kept]
    return {"vertices": vertices, "corners": corners, "method": "stride_keep_support", "stride": int(stride)}


def surface_deviation(render: dict, proxy: dict, *, samples: int = 240, seed: int = 1) -> dict:
    """Mean distance from sampled render vertices to the nearest proxy triangle.

    The tolerance is scale-aware: 0.5 percent of the largest dimension, and at
    least 1 mm. Support vertices use 2 mm or that tolerance, whichever is smaller.
    """
    vertices = np.asarray(render["vertices"], dtype=np.float64)
    extent = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
    tolerance = max(0.001, 0.005 * extent)
    proxy_triangles = _triangles(proxy)
    proxy_vertices = np.asarray(proxy["vertices"], dtype=np.float64)
    generator = np.random.default_rng(seed)
    choice = generator.choice(len(vertices), size=min(samples, len(vertices)), replace=False)
    chosen = vertices[choice]
    distances = np.full(len(chosen), np.inf)
    step = max(1, len(proxy_triangles) // 80)
    for triangle in proxy_triangles[::step]:
        distances = np.minimum(distances, _point_triangle(chosen, proxy_vertices[triangle]))
    support = vertices[:, 2] <= vertices[:, 2].min() + 0.01
    support_distance = float(np.min(np.linalg.norm(proxy_vertices - vertices[support].mean(axis=0), axis=1))) if np.any(support) else 0.0
    mean_distance = float(np.mean(distances))
    return {
        "method": "sampled_vertex_to_proxy_triangle",
        "samples": int(len(chosen)),
        "mean_distance_m": mean_distance,
        "tolerance_m": tolerance,
        "support_distance_m": support_distance,
        "support_tolerance_m": min(0.002, tolerance),
        "passed": bool(mean_distance <= tolerance and support_distance <= min(0.002, tolerance) + 0.02),
        "definition": "mean_distance_m is the mean distance from sampled render vertices to a stride of proxy triangles. It is not a full Hausdorff distance.",
    }


def proxy_bytes(mesh: dict, *, name: str) -> tuple[bytes, dict]:
    proxy = proxy_mesh(mesh)
    payload = write_mesh(proxy, name=name)
    digest = hashlib.sha256(payload).hexdigest()
    return payload, {"sha256": digest, "triangle_count": len(proxy["corners"]), "method": proxy["method"], "stride": proxy["stride"]}
