"""Static support and collision checks for indoor placement.

Contact is the signed distance from collision vertices to a named support
polygon, not an axis-aligned box test. The support normal and origin come from
that polygon. A vertex projects into the polygon's plane, and containment is a
2D even-odd test in an orthonormal basis of the plane. Boundary points within
``1e-6 m`` of an edge count as inside.

The reported gap is the positive part of the minimum signed distance among
vertices that land inside the polygon. Penetration is the negative part. The
allowed gap is 1 mm and the allowed penetration is 0.1 mm. This is static
geometry. It is not a claim of dynamic stability.

Broad-phase AABB overlap can reject distant pairs. It is never the evidence
that a contact succeeded. Unintended intersections use triangle-to-triangle
distance from ``mesh_clearance`` against meshes that are not the named support.
"""

from __future__ import annotations

import logging

import numpy as np

from sensor_sim.scene_gen.asset_package import PackageError
from sensor_sim.scene_gen.indoor_transform import apply_points
from sensor_sim.scene_gen.narrow import mesh_clearance
from sensor_sim.scene_gen.structural import CLEARANCE_TRIANGLE_LIMIT

logger = logging.getLogger("sensor_sim.scene_gen.indoor_contact")

MAX_SUPPORT_GAP_M = 0.001
MAX_PENETRATION_M = 0.0001
MEASUREMENT_TOLERANCE_M = 1e-6
BOUNDARY_TOLERANCE_M = 1e-6


def measure_support(
    vertices: np.ndarray,
    faces: np.ndarray,
    support_polygon: np.ndarray,
    *,
    gap_limit_m: float = MAX_SUPPORT_GAP_M,
    penetration_limit_m: float = MAX_PENETRATION_M,
) -> dict:
    """Measure planar contact of one mesh against one support polygon."""
    polygon = np.asarray(support_polygon, dtype=np.float64)
    normal = _polygon_normal(polygon)
    origin = polygon[0]
    points = np.asarray(vertices, dtype=np.float64)
    signed = (points - origin) @ normal
    inside = _inside_polygon(points, polygon, normal, origin)
    if not np.any(inside):
        if _support_overlaps_object(polygon, points, normal, origin):
            inside = np.ones(len(points), dtype=bool)
        else:
            nearest = int(np.argmin(np.abs(signed))) if len(signed) else 0
            return _contact_record(
                gap=float("inf"),
                penetration=0.0,
                normal=normal,
                point=points[nearest] if len(points) else origin,
                inside=False,
                passed=False,
                reason="outside_support_polygon",
                gap_limit_m=gap_limit_m,
                penetration_limit_m=penetration_limit_m,
            )
    distances = signed[inside]
    nearest = int(np.flatnonzero(inside)[int(np.argmin(distances))])
    minimum = float(signed[nearest])
    gap = max(minimum, 0.0)
    penetration = max(-minimum, 0.0)
    passed = bool(gap <= gap_limit_m + MEASUREMENT_TOLERANCE_M and penetration <= penetration_limit_m + MEASUREMENT_TOLERANCE_M)
    reason = "contact" if passed else "gap" if gap > gap_limit_m else "penetration"
    return _contact_record(
        gap=gap,
        penetration=penetration,
        normal=normal,
        point=points[nearest],
        inside=True,
        passed=passed,
        reason=reason,
        gap_limit_m=gap_limit_m,
        penetration_limit_m=penetration_limit_m,
        triangle_count=int(len(faces)),
    )


def require_contacts(records: list[dict]) -> None:
    failed = [record for record in records if not record["passed"]]
    if failed:
        raise PackageError(f"reason=support_contact count={len(failed)} first={failed[0]['reason']}")


def support_graph_ok(edges: list[tuple[str, str]]) -> bool:
    """True when the directed support graph has no cycle. An edge is supporter -> supported."""
    adjacency: dict[str, list[str]] = {}
    for supporter, supported in edges:
        adjacency.setdefault(supporter, []).append(supported)
        adjacency.setdefault(supported, [])
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(node: str) -> bool:
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        for child in adjacency.get(node, []):
            if not walk(child):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    return all(walk(node) for node in adjacency)


def unintended_intersection(first: dict, second: dict) -> dict:
    """Triangle distance against a mesh that is not the intended support surface."""
    found = mesh_clearance(first, second)
    intersecting = bool(found["surface_intersection"] or found["clearance_m"] <= MAX_PENETRATION_M)
    return {
        "clearance_m": found["clearance_m"],
        "surface_intersection": bool(found["surface_intersection"]),
        "unintended": intersecting,
        "method": found["method"],
        "triangle_count": found["n_triangles"],
    }


def require_clearance_resource(triangle_count: int, proxy_qualified: bool, proxy_hash: str, expected_hash: str) -> None:
    if proxy_hash != expected_hash:
        raise PackageError("reason=collision_hash")
    if triangle_count > CLEARANCE_TRIANGLE_LIMIT and not proxy_qualified:
        raise PackageError("reason=clearance_proxy_required")


def _contact_record(*, gap, penetration, normal, point, inside, passed, reason, gap_limit_m, penetration_limit_m, triangle_count=0) -> dict:
    logger.info(
        "indoor_contact event=measure reason=%s passed=%s gap_m=%.6g penetration_m=%.6g",
        reason,
        passed,
        gap,
        penetration,
    )
    return {
        "gap_m": float(gap),
        "penetration_m": float(penetration),
        "normal": [float(value) for value in normal],
        "contact_point_m": [float(value) for value in point],
        "inside_polygon": bool(inside),
        "passed": bool(passed),
        "reason": reason,
        "gap_limit_m": float(gap_limit_m),
        "penetration_limit_m": float(penetration_limit_m),
        "measurement_tolerance_m": MEASUREMENT_TOLERANCE_M,
        "method": "signed_distance_and_polygon_containment",
        "triangle_count": int(triangle_count),
        "units": "m",
    }


def _support_overlaps_object(polygon: np.ndarray, points: np.ndarray, normal: np.ndarray, origin: np.ndarray) -> bool:
    """True when a smaller support polygon lands on a larger object footprint.

    A ceiling vertex does not have to fall inside a thin wall-top polygon. The
    wall-top polygon has to land on the ceiling. The test is the support
    polygon against the projected bounds of the object vertices.
    """
    axis_u, axis_v = _plane_basis(normal)
    object_2d = np.stack([(points - origin) @ axis_u, (points - origin) @ axis_v], axis=1)
    polygon_2d = np.stack([(polygon - origin) @ axis_u, (polygon - origin) @ axis_v], axis=1)
    low = object_2d.min(axis=0) - BOUNDARY_TOLERANCE_M
    high = object_2d.max(axis=0) + BOUNDARY_TOLERANCE_M
    inside_object = np.all((polygon_2d >= low) & (polygon_2d <= high), axis=1)
    return bool(np.any(inside_object))


def _polygon_normal(polygon: np.ndarray) -> np.ndarray:
    normal = np.zeros(3, dtype=np.float64)
    for index, start in enumerate(polygon):
        stop = polygon[(index + 1) % len(polygon)]
        normal[0] += (start[1] - stop[1]) * (start[2] + stop[2])
        normal[1] += (start[2] - stop[2]) * (start[0] + stop[0])
        normal[2] += (start[0] - stop[0]) * (start[1] + stop[1])
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        raise PackageError("reason=support_normal")
    return normal / length


def _inside_polygon(points: np.ndarray, polygon: np.ndarray, normal: np.ndarray, origin: np.ndarray) -> np.ndarray:
    axis_u, axis_v = _plane_basis(normal)
    polygon_2d = np.stack([(polygon - origin) @ axis_u, (polygon - origin) @ axis_v], axis=1)
    point_2d = np.stack([(points - origin) @ axis_u, (points - origin) @ axis_v], axis=1)
    return np.asarray([_point_inside(point, polygon_2d) for point in point_2d], dtype=bool)


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(float(normal[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis_u = np.cross(helper, normal)
    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    return axis_u, axis_v


def _point_inside(point: np.ndarray, polygon: np.ndarray) -> bool:
    if _near_edge(point, polygon):
        return True
    count = 0
    for index, start in enumerate(polygon):
        stop = polygon[(index + 1) % len(polygon)]
        crosses = (start[1] > point[1]) != (stop[1] > point[1])
        if not crosses:
            continue
        x_hit = start[0] + (point[1] - start[1]) * (stop[0] - start[0]) / (stop[1] - start[1])
        if point[0] < x_hit:
            count += 1
    return count % 2 == 1


def _near_edge(point: np.ndarray, polygon: np.ndarray) -> bool:
    for index, start in enumerate(polygon):
        stop = polygon[(index + 1) % len(polygon)]
        segment = stop - start
        length = float(np.dot(segment, segment))
        if length < 1e-20:
            continue
        weight = float(np.clip(np.dot(point - start, segment) / length, 0.0, 1.0))
        closest = start + weight * segment
        if float(np.linalg.norm(point - closest)) <= BOUNDARY_TOLERANCE_M:
            return True
    return False


def world_vertices(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return apply_points(matrix, vertices)
