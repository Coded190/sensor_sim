"""Indoor package and mesh validation.

Geometry profiles do not call Dataset v1 ``mesh_issues`` and do not change its
2 m extent rule. A mesh does not pass because its bounds or vertex count look plausible.

Numerical rules:

- A triangle is degenerate when its area is below ``1e-14 m²``.
- An undirected edge with one incident triangle is a boundary. Two is a manifold
  interior edge. Any other count is a non-manifold edge.
- A vertex is non-manifold when the opposite edges of its incident triangles do
  not form one path or one cycle. That is the vertex-link test.
- An interior edge is wound consistently when its two triangles traverse it in
  opposite directions.
- Closed solids use outward right-handed winding. The signed volume is positive.
  A negative volume is an inverted solid and is rejected. Validation does not
  flip faces by itself.
- Face normals must be finite and nonzero. After normalization, their dot product
  with the geometric polygon normal must be at least ``FACE_NORMAL_MIN_DOT``.
  A unit-length normal that points the wrong way fails.
- ``structural_solid`` adds a closed-solid check, thickness of at least 5 mm
  along the unit axis stored as ``thickness_axis``, and a maximum axis extent
  of 20 m. Thickness is ``max(dot(vertex, axis)) - min(dot(vertex, axis))``.
- ``multi_component`` requires every edge-connected component to have its own
  stable id and to pass its declared profile.
- Loose vertices that belong to no face are rejected. No profile permits them.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import PackageError, verify_package

DEGENERATE_AREA_M2 = 1e-14
FACE_NORMAL_MIN_DOT = 1.0 - 1e-4
VERTEX_NORMAL_MIN_DOT = 0.5
MIN_THICKNESS_M = 0.005
MAX_STRUCTURAL_EXTENT_M = 20.0
PROFILES = frozenset({"closed_solid", "multi_component", "structural_solid", "open_surface"})
CLOSED_PROFILES = frozenset({"closed_solid", "structural_solid"})


def validate_package(root: Path) -> dict:
    """Require a canonical, hash-valid package whose paths stay inside the package root."""
    manifest = verify_package(root)
    for item in manifest["files"]:
        path = Path(root, *item["path"].split("/"))
        if path.is_symlink():
            raise PackageError(f"reason=symlink path={item['path']}")
        if not path.is_file():
            raise PackageError(f"reason=missing_file path={item['path']}")
    return manifest


def triangulate(polygons: list[list[int]]) -> np.ndarray:
    """Fan-triangulate polygons. The triangle count is not the polygon count."""
    triangles: list[list[int]] = []
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        anchor = int(polygon[0])
        for index in range(1, len(polygon) - 1):
            triangles.append([anchor, int(polygon[index]), int(polygon[index + 1])])
    if not triangles:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(triangles, dtype=np.int64)


def polygon_normal(vertices: np.ndarray, polygon: list[int]) -> np.ndarray:
    """Newell normal. The zero vector is returned for a degenerate polygon."""
    normal = np.zeros(3, dtype=np.float64)
    for index, current in enumerate(polygon):
        start = vertices[int(current)]
        stop = vertices[int(polygon[(index + 1) % len(polygon)])]
        normal[0] += (start[1] - stop[1]) * (start[2] + stop[2])
        normal[1] += (start[2] - stop[2]) * (start[0] + stop[0])
        normal[2] += (start[0] - stop[0]) * (start[1] + stop[1])
    length = float(np.linalg.norm(normal))
    if length < 1e-15:
        return normal
    return normal / length


def signed_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Signed volume in cubic meters. Outward right-handed solids are positive."""
    if len(triangles) == 0:
        return 0.0
    corners = vertices[triangles]
    return float(np.sum(np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2]))) / 6.0)


def triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    if len(triangles) == 0:
        return np.zeros(0, dtype=np.float64)
    corners = vertices[triangles]
    return 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)


def thickness_along(vertices: np.ndarray, axis: np.ndarray) -> float:
    direction = np.asarray(axis, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if abs(length - 1.0) > 1e-8:
        raise PackageError("reason=thickness_axis_not_unit")
    direction = direction / length
    projection = vertices @ direction
    return float(np.max(projection) - np.min(projection))


def validate_mesh(
    vertices: np.ndarray,
    polygons: list[list[int]],
    profile: str,
    *,
    components: list[dict] | None = None,
    normals: np.ndarray | None = None,
    thickness_axis: list[float] | None = None,
) -> dict:
    """Return polygon and triangle counts plus profile failures. An empty issue list passes."""
    if profile not in PROFILES:
        raise PackageError(f"reason=profile profile={profile}")
    vertices = np.asarray(vertices, dtype=np.float64)
    issues: list[str] = []
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(polygons) == 0 or len(vertices) == 0:
        return _report(vertices, polygons, profile, ["empty"], 0.0, None)
    if not np.isfinite(vertices).all():
        issues.append("non_finite")
    referenced = {int(index) for polygon in polygons for index in polygon}
    if any(index < 0 or index >= len(vertices) for index in referenced):
        issues.append("face_index")
    if len(referenced) != len(vertices):
        issues.append("loose_vertex")
    triangles = triangulate(polygons)
    areas = triangle_areas(vertices, triangles)
    if np.any(areas < DEGENERATE_AREA_M2):
        issues.append("degenerate")
    issues.extend(_edge_issues(triangles, profile))
    if _nonmanifold_vertices(triangles):
        issues.append("non_manifold_vertex")
    volume = signed_volume(vertices, triangles)
    if profile in CLOSED_PROFILES and "boundary_edge" not in issues and "non_manifold_edge" not in issues:
        if volume < 0.0:
            issues.append("inverted_volume")
        elif abs(volume) <= DEGENERATE_AREA_M2:
            issues.append("zero_volume")
    if normals is not None:
        issues.extend(_normal_issues(vertices, polygons, triangles, normals))
    thickness = None
    if profile == "structural_solid":
        if thickness_axis is None:
            issues.append("thickness_axis")
        else:
            try:
                thickness = thickness_along(vertices, np.asarray(thickness_axis, dtype=np.float64))
            except PackageError:
                issues.append("thickness_axis")
            else:
                if thickness < MIN_THICKNESS_M:
                    issues.append("thickness")
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        if len(vertices) and float(np.max(extent)) > MAX_STRUCTURAL_EXTENT_M:
            issues.append("extent")
    if profile == "multi_component":
        issues.extend(_component_issues(vertices, polygons, components))
    return _report(vertices, polygons, profile, _unique(issues), volume, thickness)


def orient_outward(vertices: np.ndarray, polygons: list[list[int]]) -> list[list[int]]:
    """Return faces reversed when the signed volume is negative. Callers must ask for this."""
    triangles = triangulate(polygons)
    if signed_volume(np.asarray(vertices, dtype=np.float64), triangles) >= 0.0:
        return [list(polygon) for polygon in polygons]
    return [list(reversed(polygon)) for polygon in polygons]


def _report(vertices, polygons, profile, issues, volume, thickness) -> dict:
    triangles = triangulate(polygons) if len(polygons) else np.zeros((0, 3), dtype=np.int64)
    if len(vertices) == 0:
        bounds_min = [0.0, 0.0, 0.0]
        bounds_max = [0.0, 0.0, 0.0]
    else:
        bounds_min = vertices.min(axis=0).tolist()
        bounds_max = vertices.max(axis=0).tolist()
    return {
        "profile": profile,
        "passed": not issues,
        "issues": issues,
        "polygon_count": len(polygons),
        "triangle_count": int(len(triangles)),
        "signed_volume_m3": volume,
        "thickness_m": thickness,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "degenerate_area_m2": DEGENERATE_AREA_M2,
        "face_normal_min_dot": FACE_NORMAL_MIN_DOT,
        "vertex_normal_min_dot": VERTEX_NORMAL_MIN_DOT,
        "min_thickness_m": MIN_THICKNESS_M,
        "max_extent_m": MAX_STRUCTURAL_EXTENT_M,
    }


def _edge_issues(triangles: np.ndarray, profile: str) -> list[str]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    directed: dict[tuple[int, int], int] = defaultdict(int)
    for face in triangles:
        indexes = [int(value) for value in face]
        for start, stop in ((0, 1), (1, 2), (2, 0)):
            first, second = indexes[start], indexes[stop]
            directed[(first, second)] += 1
            key = (first, second) if first < second else (second, first)
            counts[key] += 1
    issues = []
    boundary = False
    nonmanifold = False
    inconsistent = False
    for first, second in counts:
        count = counts[(first, second)]
        forward = directed[(first, second)]
        backward = directed[(second, first)]
        if count == 1:
            boundary = True
        elif count != 2:
            nonmanifold = True
        elif forward != 1 or backward != 1:
            inconsistent = True
    if profile in CLOSED_PROFILES and boundary:
        issues.append("boundary_edge")
    if profile == "open_surface" and boundary:
        pass
    if nonmanifold:
        issues.append("non_manifold_edge")
    if inconsistent:
        issues.append("inconsistent_winding")
    if profile == "multi_component" and (nonmanifold or inconsistent):
        pass
    return issues


def _nonmanifold_vertices(triangles: np.ndarray) -> bool:
    links: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for face in triangles:
        indexes = [int(value) for value in face]
        links[indexes[0]].append((indexes[1], indexes[2]))
        links[indexes[1]].append((indexes[2], indexes[0]))
        links[indexes[2]].append((indexes[0], indexes[1]))
    for edges in links.values():
        adjacency: dict[int, list[int]] = defaultdict(list)
        for start, stop in edges:
            adjacency[start].append(stop)
            adjacency[stop].append(start)
        if any(len(neighbors) > 2 for neighbors in adjacency.values()):
            return True
        seen: set[int] = set()
        components = 0
        for node in adjacency:
            if node in seen:
                continue
            components += 1
            stack = [node]
            seen.add(node)
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
        if components != 1:
            return True
    return False


def _normal_issues(vertices, polygons, triangles, normals) -> list[str]:
    array = np.asarray(normals, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        return ["normal"]
    lengths = np.linalg.norm(array, axis=1)
    if np.any(lengths < 1e-8):
        return ["normal"]
    unit = array / lengths[:, None]
    if len(array) == len(polygons):
        for polygon, normal in zip(polygons, unit):
            geometric = polygon_normal(vertices, polygon)
            if float(np.linalg.norm(geometric)) < 1e-8 or float(np.dot(normal, geometric)) < FACE_NORMAL_MIN_DOT:
                return ["normal"]
        return []
    if len(array) == len(vertices):
        incident: dict[int, list[np.ndarray]] = defaultdict(list)
        for polygon in polygons:
            geometric = polygon_normal(vertices, polygon)
            for index in polygon:
                incident[int(index)].append(geometric)
        for index, normal in enumerate(unit):
            for geometric in incident.get(index, []):
                if float(np.dot(normal, geometric)) < VERTEX_NORMAL_MIN_DOT:
                    return ["normal"]
        return []
    return ["normal"]


def _component_issues(vertices, polygons, components) -> list[str]:
    if not components:
        return ["unnamed_component"]
    assigned: dict[int, str] = {}
    issues: list[str] = []
    groups = _geometric_components(polygons)
    declared_sets = []
    for component in components:
        identifier = component.get("id")
        profile = component.get("profile")
        indexes = component.get("polygon_indexes")
        if not isinstance(identifier, str) or identifier == "" or profile not in PROFILES or profile == "multi_component":
            issues.append("component_profile")
            continue
        if not isinstance(indexes, list) or not indexes:
            issues.append("unnamed_component")
            continue
        for index in indexes:
            if index in assigned:
                issues.append("component_overlap")
            assigned[int(index)] = identifier
        declared_sets.append((identifier, profile, set(int(value) for value in indexes)))
    if set(assigned) != set(range(len(polygons))):
        issues.append("unnamed_component")
    geometric = [set(group) for group in groups]
    for identifier, profile, indexes in declared_sets:
        matches = [group for group in geometric if group == indexes]
        if len(matches) != 1:
            issues.append("component_identity")
            continue
        subset = [polygons[index] for index in sorted(indexes)]
        used = sorted({int(index) for polygon in subset for index in polygon})
        remap = {old: new for new, old in enumerate(used)}
        local_vertices = vertices[used]
        local_polygons = [[remap[int(index)] for index in polygon] for polygon in subset]
        axis = next(
            (item.get("thickness_axis") for item in components if item.get("id") == identifier),
            None,
        )
        report = validate_mesh(
            local_vertices,
            local_polygons,
            profile,
            thickness_axis=axis if profile == "structural_solid" else None,
        )
        if not report["passed"]:
            issues.append("component_profile")
    return issues


def _geometric_components(polygons: list[list[int]]) -> list[list[int]]:
    parent = list(range(len(polygons)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    owners: dict[tuple[int, int], int] = {}
    for face_index, polygon in enumerate(polygons):
        count = len(polygon)
        for index in range(count):
            first, second = int(polygon[index]), int(polygon[(index + 1) % count])
            key = (first, second) if first < second else (second, first)
            if key in owners:
                union(face_index, owners[key])
            else:
                owners[key] = face_index
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(polygons)):
        groups[find(index)].append(index)
    return list(groups.values())


def _unique(issues: list[str]) -> list[str]:
    ordered: list[str] = []
    for issue in issues:
        if issue not in ordered:
            ordered.append(issue)
    return ordered
