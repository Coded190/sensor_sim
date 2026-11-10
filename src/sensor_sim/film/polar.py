"""Ideal linear analyzers from camera-frame Stokes vectors.

``I(theta) = 0.5 * (S0 + S1 cos(2 theta) + S2 sin(2 theta))`` with
``theta`` in radians. Four linear analyzers recover S0, S1, and S2.
They do not measure S3.
"""

from __future__ import annotations

import numpy as np

from sensor_sim.stokes import S0_MIN, polarization_maps

ANALYZER_ANGLES = {
    "analyzer_000": 0.0,
    "analyzer_045": np.pi / 4.0,
    "analyzer_090": np.pi / 2.0,
    "analyzer_135": 3.0 * np.pi / 4.0,
}


def analyzer_intensity(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, theta: float) -> np.ndarray:
    """One ideal analyzer image. ``theta`` is radians."""
    angle = float(theta)
    return 0.5 * (
        np.asarray(s0, dtype=np.float64)
        + np.asarray(s1, dtype=np.float64) * np.cos(2.0 * angle)
        + np.asarray(s2, dtype=np.float64) * np.sin(2.0 * angle)
    )


def analyzer_images(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict[str, np.ndarray]:
    """I0, I45, I90, and I135 from the closed forms of the analyzer equation."""
    s0 = np.asarray(s0, dtype=np.float64)
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    return {
        "analyzer_000": 0.5 * (s0 + s1),
        "analyzer_045": 0.5 * (s0 + s2),
        "analyzer_090": 0.5 * (s0 - s1),
        "analyzer_135": 0.5 * (s0 - s2),
    }


def reconstruct_stokes(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """S0 = I0 + I90, S1 = I0 - I90, S2 = I45 - I135. S3 is not recovered."""
    i0 = images["analyzer_000"]
    i45 = images["analyzer_045"]
    i90 = images["analyzer_090"]
    i135 = images["analyzer_135"]
    return {"s0": i0 + i90, "s1": i0 - i90, "s2": i45 - i135}


def reconstruction_error(s0, s1, s2, images: dict[str, np.ndarray]) -> dict[str, float]:
    """Maximum absolute error of the three reconstructed components."""
    recovered = reconstruct_stokes(images)
    errors = {}
    for name, source in (("s0", s0), ("s1", s1), ("s2", s2)):
        errors[name] = float(np.max(np.abs(recovered[name] - np.asarray(source, dtype=np.float64))))
    errors["max"] = max(errors.values())
    return errors


def stokes_bundle(s0, s1, s2, s3) -> dict[str, np.ndarray]:
    """Averaged Stokes, analyzers, DoLP, and AoLP. DoLP and AoLP follow stokes.py."""
    maps = polarization_maps(s0, s1, s2, s3)
    images = analyzer_images(maps["s0"], maps["s1"], maps["s2"])
    bundle = {**maps, **images}
    bundle["reconstruction_error"] = reconstruction_error(maps["s0"], maps["s1"], maps["s2"], images)
    bundle["s0_min"] = S0_MIN
    return bundle
