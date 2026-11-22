"""Pad and contact-threshold measurements for the SO-ARM101 gripper.

Point sets are meters. The contact threshold is an application rule. It is not
``ContactSensorCfg.force_threshold``, which Isaac Lab 2.3.0 uses only for
air-time tracking.
"""

from __future__ import annotations

import numpy as np


def noise_threshold(force_norms: np.ndarray) -> dict:
    """Threshold = p99.9(norm) + 3 * MAD(norm). A zero baseline uses any positive force.

    MAD is the median absolute deviation from the median, with no consistency scale.
    The result is in the same unit as ``force_norms``, newtons when that is the input.
    """
    values = np.asarray(force_norms, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("reason=force_norms_invalid")
    if np.all(values == 0.0):
        return {"method": "strict_positive_zero_baseline", "threshold_n": 0.0, "p999_n": 0.0, "mad_n": 0.0, "sample_count": int(values.size)}
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    p999 = float(np.quantile(values, 0.999))
    return {
        "method": "p99.9_plus_3_mad",
        "threshold_n": p999 + 3.0 * mad,
        "p999_n": p999,
        "mad_n": mad,
        "sample_count": int(values.size),
    }


def closest_indices(points: np.ndarray, other: np.ndarray, band_m: float = 0.004, sample: int = 1500) -> np.ndarray:
    """Indices of points within ``band_m`` of the minimum distance to ``other``.

    ``other`` is subsampled so the full jaw meshes stay cheap. The seed is fixed.
    """
    cloud = np.asarray(points, dtype=np.float64)
    reference = np.asarray(other, dtype=np.float64)
    rng = np.random.default_rng(0)
    choice = rng.choice(len(reference), size=min(sample, len(reference)), replace=False)
    distance = np.linalg.norm(cloud[:, None, :] - reference[choice][None, :, :], axis=2).min(axis=1)
    return np.flatnonzero(distance <= float(distance.min()) + band_m)


def facing_indices(points: np.ndarray, toward: np.ndarray, fraction: float = 0.15) -> np.ndarray:
    """Indices of points on the side that faces ``toward``.

    The side is the highest ``fraction`` of projections onto the unit direction
    from the point-set centroid to ``toward``.
    """
    cloud = np.asarray(points, dtype=np.float64)
    target = np.asarray(toward, dtype=np.float64)
    center = cloud.mean(axis=0)
    direction = target - center
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("reason=facing_direction_degenerate")
    direction = direction / norm
    projection = (cloud - center) @ direction
    cutoff = float(np.quantile(projection, 1.0 - fraction))
    return np.flatnonzero(projection >= cutoff)


def pinch_measurement(fixed_world: np.ndarray, moving_world: np.ndarray, finger_direction: np.ndarray) -> dict:
    """Opening between the jaws in the plane perpendicular to the finger.

    The closing axis is the component of the centroid offset that is orthogonal
    to the finger. The gap is the distance between the inner surfaces along that
    axis. It is positive when the surfaces do not overlap.
    """
    fixed = np.asarray(fixed_world, dtype=np.float64)
    moving = np.asarray(moving_world, dtype=np.float64)
    finger = np.asarray(finger_direction, dtype=np.float64)
    finger = finger / max(float(np.linalg.norm(finger)), 1e-12)
    offset = moving.mean(axis=0) - fixed.mean(axis=0)
    lateral = offset - finger * float(np.dot(offset, finger))
    norm = float(np.linalg.norm(lateral))
    if norm < 1e-9:
        raise ValueError("reason=pinch_axis_degenerate")
    axis = lateral / norm
    fixed_projection = fixed @ axis
    moving_projection = moving @ axis
    fixed_limit = float(fixed_projection.max())
    moving_limit = float(moving_projection.min())
    fixed_index = np.flatnonzero(fixed_projection >= fixed_limit - 0.012)
    moving_index = np.flatnonzero(moving_projection <= moving_limit + 0.012)
    fixed_center = fixed[fixed_index].mean(axis=0)
    moving_center = moving[moving_index].mean(axis=0)
    return {
        "closing_axis_world": axis.tolist(),
        "fixed_face_center_m": fixed_center.tolist(),
        "moving_face_center_m": moving_center.tolist(),
        "separation_m": moving_limit - fixed_limit,
        "fixed_face_index": fixed_index,
        "moving_face_index": moving_index,
    }


def pad_measurement(fixed_world: np.ndarray, moving_world: np.ndarray) -> dict:
    """Separation and closing axis from the two jaw collision clouds.

    Separation is the distance between the mean of the facing vertices along the
    closing axis. It is positive when the clouds do not overlap along that axis.
    """
    fixed = np.asarray(fixed_world, dtype=np.float64)
    moving = np.asarray(moving_world, dtype=np.float64)
    fixed_index = closest_indices(fixed, moving)
    moving_index = closest_indices(moving, fixed)
    fixed_face = fixed[fixed_index]
    moving_face = moving[moving_index]
    fixed_center = fixed_face.mean(axis=0)
    moving_center = moving_face.mean(axis=0)
    axis = moving_center - fixed_center
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("reason=pad_centers_coincident")
    axis = axis / norm
    return {
        "closing_axis_world": axis.tolist(),
        "fixed_face_center_m": fixed_center.tolist(),
        "moving_face_center_m": moving_center.tolist(),
        "separation_m": float((moving_center - fixed_center) @ axis),
        "fixed_face_index": fixed_index,
        "moving_face_index": moving_index,
    }


def axis_aligned_bounds(points: np.ndarray, margin_m: float = 0.002) -> dict:
    cloud = np.asarray(points, dtype=np.float64)
    return {
        "min_m": (cloud.min(axis=0) - margin_m).tolist(),
        "max_m": (cloud.max(axis=0) + margin_m).tolist(),
        "margin_m": margin_m,
    }


def point_in_bounds(point: np.ndarray, bounds: dict) -> bool:
    value = np.asarray(point, dtype=np.float64)
    lower = np.asarray(bounds["min_m"], dtype=np.float64)
    upper = np.asarray(bounds["max_m"], dtype=np.float64)
    return bool(np.all(value >= lower) and np.all(value <= upper))


def principal_axis(points: np.ndarray) -> dict:
    """Largest-eigenvector axis of the vertex covariance. The axis is a unit vector."""
    cloud = np.asarray(points, dtype=np.float64)
    center = cloud.mean(axis=0)
    covariance = np.cov((cloud - center).T)
    values, vectors = np.linalg.eigh(covariance)
    order = int(np.argmax(values))
    axis = vectors[:, order]
    if axis[2] < 0.0:
        axis = -axis
    return {"center_m": center.tolist(), "axis": axis.tolist(), "eigenvalues": values.tolist()}
