"""Stokes maps from a Mitsuba ``stokes`` integrator image.

The integrator stores each Stokes component as three spectral channels.
On a constant-spectrum scene those three copies match. These helpers
average them, then apply the definitions below.

``S0`` is intensity. ``S1`` is horizontal versus vertical linear
polarization. ``S2`` is +45 degree versus -45 degree linear polarization.
``S3`` is right versus left circular polarization.

- Degree of polarization: ``DoP = sqrt(S1^2 + S2^2 + S3^2) / S0``
  where ``S0 > S0_MIN``, else 0. Noise-free light has ``DoP <= 1``.
- Degree of linear polarization: ``DoLP = sqrt(S1^2 + S2^2) / S0``
  where ``S0 > S0_MIN``, else 0. This is the linear part divided by
  total intensity.
- Angle of linear polarization: ``AoLP = 0.5 * arctan2(S2, S1)``
  in radians, range ``(-pi/2, pi/2]``. The sign follows Mitsuba's
  sensor Stokes frame. Pixels with ``DoLP < AOLP_DOLP_MIN`` are masked
  because the angle is unstable there.

``mean_dolp``, ``max_dolp``, ``max_dop``, ``frac_dop_gt_1_05``, and
``mean_abs_s3`` are computed on pixels with ``S0 > S0_MIN``. ``mean_abs_s3``
is the mean of ``|S3|`` in radiance units, the same units as ``S0``.
``mean_abs_s3_over_s0`` is the mean of ``|S3| / S0`` on those pixels.
``max_dolp`` is the check that some pixel is linearly polarized; ``mean_dolp``
is diluted by unpolarized walls.
"""

from __future__ import annotations

import numpy as np

S0_MIN = 1e-4
AOLP_DOLP_MIN = 0.05
DOP_EXCESS = 1.05

DEFINITIONS = {
    "s0_min": S0_MIN,
    "aolp_dolp_min": AOLP_DOLP_MIN,
    "dop_excess": DOP_EXCESS,
    "dop": "sqrt(S1^2 + S2^2 + S3^2) / S0 where S0 > s0_min, else 0",
    "dolp": "sqrt(S1^2 + S2^2) / S0 where S0 > s0_min, else 0",
    "aolp_rad": "0.5 * arctan2(S2, S1); masked where DoLP < aolp_dolp_min or S0 <= s0_min",
    "mean_abs_s3": "mean(|S3|) over pixels with S0 > s0_min",
    "mean_abs_s3_over_s0": "mean(|S3| / S0) over pixels with S0 > s0_min",
    "mean_dolp": "mean(DoLP) over pixels with S0 > s0_min",
    "max_dolp": "max(DoLP) over pixels with S0 > s0_min",
    "max_dop": "max(DoP) over pixels with S0 > s0_min",
    "frac_dop_gt_1_05": "fraction of S0 > s0_min pixels with DoP > dop_excess",
}


def average_spectrum(image: np.ndarray) -> np.ndarray:
    """Average spectral copies. ``HxWx3`` becomes ``HxW``; ``HxW`` is unchanged."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] >= 1:
        return array.mean(axis=2)
    raise ValueError(f"expected HxW or HxWxC image, got shape {array.shape}")


def polarization_maps(
    s0: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return averaged Stokes components and DoP, DoLP, and AoLP."""
    s0_avg = average_spectrum(s0)
    s1_avg = average_spectrum(s1)
    s2_avg = average_spectrum(s2)
    s3_avg = average_spectrum(s3)
    if not (s0_avg.shape == s1_avg.shape == s2_avg.shape == s3_avg.shape):
        raise ValueError(
            "Stokes components disagree after averaging: "
            f"S0 {s0_avg.shape}, S1 {s1_avg.shape}, S2 {s2_avg.shape}, S3 {s3_avg.shape}"
        )

    valid = s0_avg > S0_MIN
    linear = np.sqrt(s1_avg**2 + s2_avg**2)
    polarized = np.sqrt(s1_avg**2 + s2_avg**2 + s3_avg**2)
    dolp = np.zeros_like(s0_avg)
    dop = np.zeros_like(s0_avg)
    np.divide(linear, s0_avg, out=dolp, where=valid)
    np.divide(polarized, s0_avg, out=dop, where=valid)
    aolp = 0.5 * np.arctan2(s2_avg, s1_avg)
    aolp_masked = (~valid) | (dolp < AOLP_DOLP_MIN)
    return {
        "s0": s0_avg,
        "s1": s1_avg,
        "s2": s2_avg,
        "s3": s3_avg,
        "valid": valid,
        "dolp": dolp,
        "dop": dop,
        "aolp": aolp,
        "aolp_masked": aolp_masked,
    }


def summarize_polarization(maps: dict[str, np.ndarray]) -> dict[str, float | int]:
    """Reduce polarization maps to the scalars written in the Phase 0 log line."""
    valid = maps["valid"]
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return {
            "n_valid": 0,
            "mean_dolp": 0.0,
            "max_dolp": 0.0,
            "max_dop": 0.0,
            "frac_dop_gt_1_05": 0.0,
            "mean_abs_s3": 0.0,
            "mean_abs_s3_over_s0": 0.0,
        }
    dolp = maps["dolp"][valid]
    dop = maps["dop"][valid]
    s0 = maps["s0"][valid]
    s3 = maps["s3"][valid]
    return {
        "n_valid": n_valid,
        "mean_dolp": float(dolp.mean()),
        "max_dolp": float(dolp.max()),
        "max_dop": float(dop.max()),
        "frac_dop_gt_1_05": float(np.mean(dop > DOP_EXCESS)),
        "mean_abs_s3": float(np.abs(s3).mean()),
        "mean_abs_s3_over_s0": float(np.mean(np.abs(s3) / s0)),
    }
