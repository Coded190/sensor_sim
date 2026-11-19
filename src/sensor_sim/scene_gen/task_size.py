"""Instance scale checks. The packaged mesh is never rewritten."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.mesh_io import read_obj
from sensor_sim.scene_gen.task_schema import TaskError


def mesh_bounds(catalog: Path, asset_id: str, version: str) -> dict:
    vertices, _polygons = read_obj(Path(catalog) / asset_id / version / "geometry" / "render.obj")
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    return {
        "minimum_m": low.tolist(),
        "maximum_m": high.tolist(),
        "size_m": (high - low).tolist(),
        "prior_status": "asset_declared",
        "prior_source": "package_geometry",
    }


def accept_scale(bounds: dict, scale: list[float], *, nonuniform_allowed: bool = False) -> dict:
    if len(scale) != 3 or any(float(value) <= 0.0 for value in scale):
        raise TaskError("reason=scale")
    values = [float(value) for value in scale]
    nonuniform = max(values) - min(values) > 1e-8
    if nonuniform and not nonuniform_allowed:
        raise TaskError("reason=nonuniform_scale")
    original = np.asarray(bounds["size_m"], dtype=np.float64)
    accepted = original * np.asarray(values)
    return {
        "original_size_m": bounds["size_m"],
        "original_bounds_m": {"minimum_m": bounds["minimum_m"], "maximum_m": bounds["maximum_m"]},
        "prior_status": bounds["prior_status"],
        "prior_source": bounds["prior_source"],
        "proposed_size_m": accepted.tolist(),
        "accepted_size_m": accepted.tolist(),
        "scale": values,
        "real_world_plausible": False if bounds["prior_status"] == "unknown" else None,
    }


def fits_interior(object_size: list[float], interior: dict) -> dict:
    """Compare an object with a named interior, including the margin."""
    margin = float(interior["margin_m"])
    usable = np.asarray(interior["size_m"], dtype=np.float64) - np.array([2.0 * margin, 2.0 * margin, margin])
    ordered = sorted(float(value) for value in object_size)
    height = ordered[0]
    footprint = sorted(ordered[1:])
    opening = sorted(float(value) for value in usable[:2])
    fits = bool(footprint[0] <= opening[0] + 1e-9 and footprint[1] <= opening[1] + 1e-9 and height <= float(usable[2]) + 1e-9)
    return {
        "fits": fits,
        "usable_m": usable.tolist(),
        "object_size_m": [float(value) for value in object_size],
        "margin_m": margin,
        "reason": "fit" if fits else "interior",
    }


def fits_support(object_size_xy: list[float], support_polygon: np.ndarray) -> bool:
    span = support_polygon.max(axis=0) - support_polygon.min(axis=0)
    return bool(float(object_size_xy[0]) <= float(span[0]) + 1e-9 and float(object_size_xy[1]) <= float(span[1]) + 1e-9)
