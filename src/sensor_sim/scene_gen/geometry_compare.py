"""Compare a source mesh with the mesh reconstructed for Mitsuba.

Distances are metres. A matching bounding box is not a pass by itself.
The reported tolerance is the larger of 1 mm and 0.1 percent of the source
diagonal. That follows OBJ text written with six decimal places, which is
about a micrometre, plus a margin for the evaluated-mesh and instance-matrix
path. It is not raised to hide a centimetre-scale placement error.
"""

from __future__ import annotations

import numpy as np

TOLERANCE_FLOOR_M = 0.001
TOLERANCE_FRACTION = 0.001


def surface_tolerance_m(source: np.ndarray) -> float:
    points = np.asarray(source, dtype=np.float64)
    if len(points) == 0:
        return TOLERANCE_FLOOR_M
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return max(TOLERANCE_FLOOR_M, TOLERANCE_FRACTION * diagonal)


def nearest_distances(source: np.ndarray, exported: np.ndarray) -> np.ndarray:
    """Distance from each source sample to the nearest exported sample, in metres."""
    left = np.asarray(source, dtype=np.float64)
    right = np.asarray(exported, dtype=np.float64)
    if len(left) == 0 or len(right) == 0:
        raise ValueError("reason=empty_comparison")
    distances = np.empty(len(left), dtype=np.float64)
    block = 4096
    for start in range(0, len(left), block):
        chunk = left[start : start + block]
        delta = chunk[:, None, :] - right[None, :, :]
        distances[start : start + block] = np.linalg.norm(delta, axis=2).min(axis=1)
    return distances


def comparison_report(source: np.ndarray, exported: np.ndarray, *, normals_source: np.ndarray | None = None, normals_exported: np.ndarray | None = None) -> dict:
    distances = nearest_distances(source, exported)
    tolerance = surface_tolerance_m(source)
    report = {
        "samples": int(len(distances)),
        "max_m": float(distances.max()),
        "mean_m": float(distances.mean()),
        "p95_m": float(np.percentile(distances, 95)),
        "p99_m": float(np.percentile(distances, 99)),
        "tolerance_m": float(tolerance),
        "tolerance_definition": "max(0.001 m, 0.001 * source bounding diagonal). Export integrity, not shape-from-polarization ground truth.",
        "passed": bool(float(distances.max()) <= tolerance),
        "bounds_source_m": [source.min(axis=0).tolist(), source.max(axis=0).tolist()],
        "bounds_exported_m": [exported.min(axis=0).tolist(), exported.max(axis=0).tolist()],
    }
    if normals_source is not None and normals_exported is not None:
        left = np.asarray(normals_source, dtype=np.float64)
        right = np.asarray(normals_exported, dtype=np.float64)
        dots = np.clip(np.sum(left * right, axis=1), -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))
        report["normal_dot_mean"] = float(np.mean(dots))
        report["normal_angle_mean_deg"] = float(np.mean(angles))
        report["normal_angle_max_deg"] = float(np.max(angles))
        report["normal_orientation_agreement"] = float(np.mean(dots > 0.0))
        report["normal_passed"] = bool(float(np.min(dots)) > 0.95)
    return report
