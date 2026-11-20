"""Place heterogeneous objects on one tabletop.

This packer does not call ``task_layout.resolve``. Support and intersection
decisions go through ``measure_support`` and ``mesh_clearance``. A failure is
returned. Objects are not scaled and overlaps are not slid along +X.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import verify_package
from sensor_sim.scene_gen.asset_validate import triangulate
from sensor_sim.scene_gen.indoor_contact import measure_support, world_vertices
from sensor_sim.scene_gen.indoor_transform import apply_points, compose, rotation_matrix, serialize_matrix, translation_matrix
from sensor_sim.scene_gen.mesh_io import read_obj
from sensor_sim.scene_gen.narrow import mesh_clearance

logger = logging.getLogger("sensor_sim.scene_gen.tabletop_layout")

MAX_TRIES = 80
INSET_M = 0.015
SEPARATION = {"controlled": 0.03, "mixed": 0.015, "clutter": 0.008}


def load_mesh(catalog: Path, asset_id: str, version: str = "1.0.0") -> dict:
    manifest = verify_package(Path(catalog) / asset_id / version)
    vertices, polygons = read_obj(Path(catalog) / asset_id / version / "geometry" / "render.obj")
    return {
        "vertices": vertices,
        "polygons": polygons,
        "faces": triangulate(polygons),
        "manifest": manifest,
    }


def place_tabletop(
    catalog: Path,
    table_matrix: np.ndarray,
    table_support_local: np.ndarray,
    requests: list[dict],
    *,
    seed: int,
    minimum_separation_m: float,
) -> dict:
    """Return poses for ``requests``. ``valid`` is false when a request cannot be placed."""
    catalog = Path(catalog)
    support = apply_points(table_matrix, np.asarray(table_support_local, dtype=np.float64))
    inset = _inset_rect(support, INSET_M)
    support_z = float(support[0, 2])
    rng = np.random.default_rng(int(seed))
    loaded = []
    for request in requests:
        mesh = load_mesh(catalog, request["asset_id"], request.get("version", "1.0.0"))
        area = _footprint_area(mesh["vertices"])
        loaded.append({**request, "mesh": mesh, "area": area})
    loaded.sort(key=lambda item: (-item["area"], item["instance_id"]))
    placed: list[dict] = []
    relations = []
    attempts = []
    for item in loaded:
        found = _place_one(item, placed, inset, support, support_z, rng, minimum_separation_m)
        attempts.append({"instance_id": item["instance_id"], "asset_id": item["asset_id"], "tries": found["tries"], "placed": found["pose"] is not None})
        if found["pose"] is None:
            logger.info(
                "tabletop_layout event=placement_failed instance=%s asset=%s tries=%d seed=%d separation_m=%.4f",
                item["instance_id"],
                item["asset_id"],
                found["tries"],
                seed,
                minimum_separation_m,
            )
            return {
                "valid": False,
                "poses": {row["instance_id"]: row["matrix"] for row in placed},
                "relations": relations,
                "collision_log": {
                    "termination": "placement_failed",
                    "failed_instance": item["instance_id"],
                    "attempts": attempts,
                    "minimum_separation_m": float(minimum_separation_m),
                    "inset_m": INSET_M,
                    "seed": int(seed),
                },
            }
        placed.append(found["pose"])
        relations.append(found["relation"])
        clearance = found["pose"]["nearest_clearance_m"]
        logger.info(
            "tabletop_layout event=placed instance=%s asset=%s yaw_rad=%.4f translation=%s tries=%d clearance_m=%s",
            item["instance_id"],
            item["asset_id"],
            found["pose"]["yaw_rad"],
            [round(value, 4) for value in found["pose"]["translation"]],
            found["tries"],
            "none" if clearance is None else f"{clearance:.4f}",
        )
    from sensor_sim.scene_gen.placement_separation import PlacementSeparationError, require_outside

    try:
        require_outside(
            [{"instance_id": row["instance_id"], "vertices": row["world"], "faces": row["faces"]} for row in placed],
            minimum_m=float(minimum_separation_m),
        )
    except PlacementSeparationError as exc:
        logger.info("tabletop_layout event=object_inside reason=%s", exc)
        return {
            "valid": False,
            "poses": {row["instance_id"]: row["matrix"] for row in placed},
            "relations": relations,
            "collision_log": {"termination": "object_inside", "reason": str(exc), "attempts": attempts, "seed": int(seed)},
        }
    poses = {row["instance_id"]: row["matrix"] for row in placed}
    occupancy = _occupancy(placed, support)
    logger.info(
        "tabletop_layout event=occupancy seed=%d instances=%d footprint_sum_m2=%.4f union_m2=%.4f fraction=%.4f nearest_m=%.4f",
        seed,
        len(placed),
        occupancy["footprint_sum_m2"],
        occupancy["union_m2"],
        occupancy["occupied_fraction"],
        occupancy["nearest_clearance_m"],
    )
    return {
        "valid": True,
        "poses": poses,
        "placed": placed,
        "relations": relations,
        "collision_log": {
            "termination": "placed",
            "attempts": attempts,
            "minimum_separation_m": float(minimum_separation_m),
            "inset_m": INSET_M,
            "seed": int(seed),
            "occupancy": occupancy,
        },
    }


def _place_one(item, placed, inset, support, support_z, rng, minimum_separation_m) -> dict:
    vertices = item["mesh"]["vertices"]
    cluster = item.get("cluster")
    for attempt in range(1, MAX_TRIES + 1):
        yaw = float(rng.uniform(0.0, 2.0 * np.pi))
        bounds = _yaw_bounds(vertices, yaw)
        region = cluster if attempt <= 40 and cluster is not None else inset
        span_x = bounds[2] - bounds[0]
        span_y = bounds[3] - bounds[1]
        lo_x = region[0] - bounds[0]
        hi_x = region[2] - bounds[2]
        lo_y = region[1] - bounds[1]
        hi_y = region[3] - bounds[3]
        if hi_x < lo_x or hi_y < lo_y or span_x > (region[2] - region[0]) or span_y > (region[3] - region[1]):
            continue
        translation = [float(rng.uniform(lo_x, hi_x)), float(rng.uniform(lo_y, hi_y)), support_z]
        matrix = compose(translation_matrix(translation), rotation_matrix("z", yaw))
        world = world_vertices(vertices, matrix)
        if not _inside_rect(world[:, :2], inset):
            continue
        contact = measure_support(world, item["mesh"]["faces"], support)
        if not contact["passed"]:
            continue
        nearest = float("inf")
        blocked = False
        candidate = {"vertices": world, "faces": item["mesh"]["faces"]}
        for other in placed:
            clearance = float(mesh_clearance(candidate, {"vertices": other["world"], "faces": other["faces"]})["clearance_m"])
            nearest = min(nearest, clearance)
            if clearance < minimum_separation_m:
                blocked = True
                break
        if blocked:
            continue
        return {
            "tries": attempt,
            "pose": {
                "instance_id": item["instance_id"],
                "asset_id": item["asset_id"],
                "version": item.get("version", "1.0.0"),
                "matrix": serialize_matrix(matrix),
                "yaw_rad": yaw,
                "translation": translation,
                "world": world,
                "faces": item["mesh"]["faces"],
                "vertices": vertices,
                "nearest_clearance_m": None if nearest == float("inf") else nearest,
                "cluster": item.get("cluster_name", ""),
            },
            "relation": _relation(item["instance_id"], item["support_parent"], item["support_surface"], contact),
        }
    return {"tries": MAX_TRIES, "pose": None, "relation": None}


def support_pose(parent_matrix: np.ndarray, support_local: np.ndarray, yaw: float, origin_xy: tuple[float, float]) -> np.ndarray:
    """Place a contact-at-zero mesh so its origin sits on a support point."""
    support = apply_points(parent_matrix, np.asarray(support_local, dtype=np.float64))
    height = float(support[:, 2].mean())
    translation = [float(origin_xy[0]), float(origin_xy[1]), height]
    return compose(translation_matrix(translation), rotation_matrix("z", float(yaw)))


def _relation(child: str, parent: str, surface: str, contact: dict) -> dict:
    return {
        "type": "supported_on",
        "instances": [child, parent],
        "frame": "support_surface_local",
        "frame_instance": parent,
        "surface": surface,
        "priority": "required",
        "constraints": {},
        "passed": True,
        "reason": contact["reason"],
        "measured_m": float(contact["gap_m"]),
        "gap_limit_m": float(contact["gap_limit_m"]),
        "penetration_limit_m": float(contact["penetration_limit_m"]),
    }


def _inset_rect(polygon: np.ndarray, margin: float) -> tuple[float, float, float, float]:
    return (
        float(polygon[:, 0].min()) + margin,
        float(polygon[:, 1].min()) + margin,
        float(polygon[:, 0].max()) - margin,
        float(polygon[:, 1].max()) - margin,
    )


def _inside_rect(points: np.ndarray, rect: tuple[float, float, float, float]) -> bool:
    return bool(
        np.all(points[:, 0] >= rect[0] - 1e-8)
        and np.all(points[:, 0] <= rect[2] + 1e-8)
        and np.all(points[:, 1] >= rect[1] - 1e-8)
        and np.all(points[:, 1] <= rect[3] + 1e-8)
    )


def _yaw_bounds(vertices: np.ndarray, yaw: float) -> tuple[float, float, float, float]:
    cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
    x_values = cosine * vertices[:, 0] - sine * vertices[:, 1]
    y_values = sine * vertices[:, 0] + cosine * vertices[:, 1]
    return float(x_values.min()), float(y_values.min()), float(x_values.max()), float(y_values.max())


def _footprint_area(vertices: np.ndarray) -> float:
    return float((vertices[:, 0].max() - vertices[:, 0].min()) * (vertices[:, 1].max() - vertices[:, 1].min()))


def _occupancy(placed: list[dict], support: np.ndarray) -> dict:
    table_area = float((support[:, 0].max() - support[:, 0].min()) * (support[:, 1].max() - support[:, 1].min()))
    hulls = [_hull_area(row["world"][:, :2]) for row in placed]
    union = _union_area(placed, support)
    clearances = [float(row["nearest_clearance_m"]) for row in placed if row["nearest_clearance_m"] is not None]
    return {
        "instance_count": len(placed),
        "footprint_sum_m2": float(sum(hulls)),
        "union_m2": union,
        "table_area_m2": table_area,
        "occupied_fraction": float(union / table_area) if table_area else 0.0,
        "nearest_clearance_m": float(min(clearances)) if clearances else None,
        "clearance_median_m": float(np.median(clearances)) if clearances else None,
        "clearance_max_m": float(max(clearances)) if clearances else None,
        "footprint_method": "xy_convex_hull_area",
        "union_method": "4mm_raster_of_xy_projected_triangles",
    }


def _hull_area(points: np.ndarray) -> float:
    hull = _hull(points)
    if len(hull) < 3:
        return 0.0
    x_values, y_values = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1))))


def _hull(points: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(points, 6), axis=0)
    if len(unique) <= 2:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def cross(origin, left, right) -> float:
        return float((left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0]))

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _union_area(placed: list[dict], support: np.ndarray) -> float:
    resolution = 0.004
    min_x, min_y = float(support[:, 0].min()), float(support[:, 1].min())
    max_x, max_y = float(support[:, 0].max()), float(support[:, 1].max())
    width = max(1, int(np.ceil((max_x - min_x) / resolution)))
    height = max(1, int(np.ceil((max_y - min_y) / resolution)))
    grid = np.zeros((height, width), dtype=bool)
    for row in placed:
        for face in row["faces"]:
            triangle = row["world"][face][:, :2]
            _stamp_triangle(grid, triangle, min_x, min_y, resolution, width, height)
    return float(grid.sum() * resolution * resolution)


def _stamp_triangle(grid, triangle, min_x, min_y, resolution, width, height) -> None:
    lo = triangle.min(axis=0)
    hi = triangle.max(axis=0)
    c0 = max(0, int(np.floor((lo[0] - min_x) / resolution)))
    c1 = min(width - 1, int(np.floor((hi[0] - min_x) / resolution)))
    r0 = max(0, int(np.floor((lo[1] - min_y) / resolution)))
    r1 = min(height - 1, int(np.floor((hi[1] - min_y) / resolution)))
    if c1 < c0 or r1 < r0:
        return
    columns = min_x + (np.arange(c0, c1 + 1) + 0.5) * resolution
    rows = min_y + (np.arange(r0, r1 + 1) + 0.5) * resolution
    xx, yy = np.meshgrid(columns, rows)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    inside = _points_in_triangle(points, triangle)
    grid[r0 : r1 + 1, c0 : c1 + 1] |= inside.reshape(xx.shape)


def _points_in_triangle(points: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a, b, c = triangle
    v0, v1 = c - a, b - a
    denom = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(float(denom)) < 1e-14:
        return np.zeros(len(points), dtype=bool)
    v2 = points - a
    u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / denom
    v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / denom
    return (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1.0 + 1e-8)
