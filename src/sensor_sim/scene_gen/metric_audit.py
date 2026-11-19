"""Independent convergence metrics.

The historical helper divided each Stokes component by ``np.maximum`` of that
same component and ``1e-4``. The corrected metrics divide by reference S0.
This module does not call ``pipeline_converge._relative``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CHANNEL_NAMES = (
    "R",
    "G",
    "B",
    "S0.R",
    "S0.G",
    "S0.B",
    "S1.R",
    "S1.G",
    "S1.B",
    "S2.R",
    "S2.G",
    "S2.B",
    "S3.R",
    "S3.G",
    "S3.B",
)
S0_MIN = 1e-4
DOLP_AOLP_MIN = 0.05
# Exact expression used by pipeline_converge._relative. Do not rewrite it.
ORIGINAL_DENOMINATOR_EXPRESSION = "np.maximum(reference, 1e-4)"
# Fixed before the reconstructed summaries are compared with the historical JSON.
RECONSTRUCTION_ABSOLUTE_TOLERANCE = 1e-5
BOOTSTRAP_REPLICATES = 200
BOOTSTRAP_SEED = 20261117
BOOTSTRAP_UPPER_QUANTILE = 0.95
HISTORICAL_REFERENCE_LIMITS = {
    "s0_p95": 0.15 * 0.25,
    "s1_p95": 0.08 * 0.25,
}
PROSPECTIVE_LIMITS = {
    "s0_relative": {"median": 0.0125, "p95": 0.0375, "units": "fraction of reference S0"},
    "s1_relative": {"median": 0.005, "p95": 0.02, "units": "fraction of reference S0"},
    "s2_relative": {"median": 0.005, "p95": 0.02, "units": "fraction of reference S0"},
    "s3_relative": {"median": 0.005, "p95": 0.02, "units": "fraction of reference S0"},
    "dolp_absolute": {"median": 0.005, "p95": 0.02, "units": "DoLP fraction"},
    "aolp_angular": {"median": 0.5, "p95": 2.5, "units": "degrees"},
}
CANDIDATE_LIMITS = {
    "s0_relative": {"median": 0.05, "p95": 0.15},
    "s1_relative": {"median": 0.02, "p95": 0.08},
    "s2_relative": {"median": 0.02, "p95": 0.08},
    "s3_relative": {"median": 0.02, "p95": 0.08},
    "dolp_absolute": {"median": 0.02, "p95": 0.08},
    "aolp_angular": {"median": 2.0, "p95": 10.0},
}

PROTOCOL_TEXT = """# Convergence v2 reference protocol

This protocol is the prospective rule for reference qualification. It was written before any reference render other than the historical reconstruction of seeds 1101-1104 at 1024 spp and seeds 3101-3104 at 2048 spp. It does not change the historical `stable: false` result.

## What the historical code measured

`pipeline_converge.reference_stack` formed the reference mean by averaging the scalar Stokes images of four independent seeds. For each component it computed, on pixels where the mean S0 was greater than 1e-4,

`abs(render - mean) / np.maximum(reference, 1e-4)`

where `reference` was that same component's mean, not reference S0. It then took the pixel p95 and the maximum of those p95 values across the four seeds. That statistic is the deviation of one render from the four-render mean. It is not the standard error of the mean. DoLP and AoLP were not part of `stable`. The historical limits were S0 p95 0.0375 and S1/S2/S3 p95 0.02. The historical result remains `stable: false`.

Candidate results that used this denominator for S1, S2, or S3 are historical and superseded. They do not qualify an spp.

## Scenes

Reference qualification, before any candidate spp is accepted, covers:

- The multi-object scene `dataset/qualification/final/rerun/rerun_269837089/scene.xml`.
- One dielectric control, then the full primary matrix: dielectric IOR 1.33, 1.40, 1.50, 1.60, 1.70, measured pplastic, conductor Cu, and roughconductor Cu alpha 0.1.
- Representative non-sphere geometry already present in the multi-object scene. A candidate material is scored only against a reference of that material and scene.

## Render settings

- Variant `cuda_ad_spectral_polarized`. An LLVM render is not a CUDA result.
- Integrator `stokes` over `path`, as stored in the scene XML.
- Sampler `independent`. The Mitsuba seed is the reference or candidate seed.
- Resolution and box filter are the values stored in the XML (128 by 128, box).
- Reference and candidate seeds are disjoint. Historical candidate seeds 2101, 2102, and 2103 stay out of every reference.
- A new reference estimate uses at least eight independent seeds. Four seeds are a reconstruction, not a qualified reference.
- Different spp values are not mixed in one unweighted mean.

## Scalar Stokes

Each of S0, S1, S2, and S3 is the mean of its three RGB-converted channels `component.R`, `component.G`, and `component.B`. This is the same reduction as `dataset.py`. It is not a wavelength-resolved Stokes measurement. Signed values stay signed. Non-finite values are rejected. Channels are read by name.

## Masks

- Illuminated: reference S0 > 1e-4. Equality at 1e-4 is outside the mask.
- Object: saved object id > 0 and the valid ground-truth mask is true.
- Corrected metrics use the intersection of those masks.
- The historical statistic is recomputed on its original mask, mean S0 > 1e-4, with no object mask.
- Weakly illuminated pixels are reported separately and are not added to the acceptance mask.
- Interior: an object pixel whose 8-neighborhood equals its id and which is not on the image border. Boundary object pixels are reported separately.

## Candidate limits, unchanged

- S0 relative median <= 0.05 and p95 <= 0.15.
- S1, S2, and S3 relative median <= 0.02 and p95 <= 0.08 for each component. The denominator is reference S0.
- DoLP absolute median <= 0.02 and p95 <= 0.08.
- AoLP median <= 2 degrees and p95 <= 10 degrees where both DoLP values are at least 0.05.
- DoLP is `sqrt(S1^2 + S2^2) / S0` where S0 > 1e-4, else 0.
- AoLP is `0.5 * atan2(S2, S1)`, with period 180 degrees. The reference DoLP and AoLP are computed after the Stokes arrays are averaged. Per-render DoLP and AoLP are not averaged.

## Reference uncertainty

For each scalar Stokes component, the per-pixel sample standard deviation across independent reference seeds is divided by `sqrt(n_reference)`. That standard error is divided by reference S0. S1, S2, and S3 are not divided by their own means. The spatial median and spatial p95 of this map are the point estimates.

The upper confidence bound is the 95th percentile of a seed-level bootstrap with 200 resamples and seed 20261117. Each resample draws reference seeds with replacement, rebuilds the standard-error map, and recomputes the spatial median and p95. Pixels are not seed replicates. The upper bound, not only the point estimate, must meet the limit. If the bound is wider than the limit, add independent seeds before raising spp.

The original per-render p95 deviation from the mean is reported beside the standard error and is not called the uncertainty of the reference mean.

DoLP and AoLP uncertainty uses the same bootstrap:

1. Resample reference seeds.
2. Average the Stokes arrays of the resample.
3. Compute DoLP and AoLP from that average with the training-view formulas.
4. Compare with DoLP and AoLP of the full reference average.
5. Use the 180-degree periodic difference for AoLP.
6. Record the spatial median and p95, then the 95th percentile of those statistics across bootstrap replicates.

## Stability targets

These are 25 percent of the candidate caps.

- S0 relative: median 0.0125, p95 0.0375.
- S1 relative: median 0.005, p95 0.02.
- S2 relative: median 0.005, p95 0.02.
- S3 relative: median 0.005, p95 0.02.
- DoLP absolute: median 0.005, p95 0.02.
- AoLP: median 0.5 degrees, p95 2.5 degrees.

A reference scene is qualified only when every one of these upper bounds passes. The old experiment is not passed by this estimator.

## Stopping rule

Start with the multi-object scene and one dielectric control. If corrected S1-S3 error does not move with spp in the direction expected for Monte Carlo noise, stop and inspect channels, signs, masks, the RGB reduction, and the renderer. Do not schedule 4096 spp to outrun an implementation error.

If a metric remains above its target and the error behaves like Monte Carlo noise, add independent seeds at the current spp first. Raise spp only when the seed-level curve says that more samples per render address the limiting metric, and record that metric before the render. Do not mix spp.

Candidate spp values are 256, 512, and 1024, with at least three seeds that are not in the reference. A 2048-spp candidate tier is a separate qualification and is not run until its cost is written down. The lowest spp that passes every required scene and every unchanged limit may be frozen. Otherwise spp stays null.

Maximum planned reconstruction effort before that decision: the eight historical multi-object renders, then up to eight seeds on the multi-object scene and one dielectric control at the spp the audit selects, then the remaining primary-material scenes at that same spp. A higher spp is a separate decision.

## Aggregation

Report every seed. Also report the per-scene and per-object summaries. Do not subtract reference uncertainty from candidate error.
"""


def write_protocol(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PROTOCOL_TEXT)


def _finite(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} has a non-finite value")
    return values


def _same_shape(arrays: list[np.ndarray], label: str) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"{label} resolution or channel layout mismatch: {sorted(shapes)}")


def summarize(values: np.ndarray) -> dict:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return {"median": None, "p95": None, "max": None, "n": 0}
    return {
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
        "n": int(data.size),
    }


def scalar_stokes(named: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mean of the three RGB-converted channels. Not wavelength-resolved Stokes."""
    missing = [name for name in CHANNEL_NAMES if name not in named]
    if missing:
        raise ValueError(f"EXR is missing named channels {missing}")
    arrays = {name: _finite(named[name], name) for name in CHANNEL_NAMES}
    _same_shape(list(arrays.values()), "named Stokes channels")
    out = {"rgb": np.stack([arrays["R"], arrays["G"], arrays["B"]], axis=-1)}
    for component in ("S0", "S1", "S2", "S3"):
        bands = [arrays[f"{component}.{band}"] for band in ("R", "G", "B")]
        out[component.lower()] = np.mean(np.stack(bands, axis=-1), axis=-1)
        out[f"{component.lower()}_rgb"] = np.stack(bands, axis=-1)
    return out


def historical_relative(candidate: np.ndarray, reference: np.ndarray, illuminated: np.ndarray) -> dict:
    """Reproduce ``_relative``. The denominator is ``np.maximum(reference, 1e-4)``."""
    left = _finite(candidate, "candidate")
    right = _finite(reference, "reference")
    mask = np.asarray(illuminated, dtype=bool)
    _same_shape([left, right, mask], "historical relative")
    if not mask.any():
        row = summarize(np.array([]))
    else:
        denominator = np.maximum(right, 1e-4)
        row = summarize(np.abs(left - right)[mask] / denominator[mask])
    row["denominator"] = ORIGINAL_DENOMINATOR_EXPRESSION
    row["mask"] = "illuminated flag supplied by the caller; historical reference_stack used mean S0 > 1e-4"
    return row


def historical_reference_decision(renders: list[dict[str, np.ndarray]]) -> dict:
    """Max over seeds of the pixel p95 of ``abs(render - mean) / np.maximum(mean, 1e-4)``."""
    if len(renders) < 2:
        raise ValueError("historical reference decision needs at least two seeds")
    components = ("s0", "s1", "s2", "s3")
    for name in components:
        _same_shape([item[name] for item in renders], name)
        for item in renders:
            _finite(item[name], name)
    mean = {name: np.mean(np.stack([item[name] for item in renders], axis=0), axis=0) for name in components}
    illuminated = mean["s0"] > S0_MIN
    uncertainty = {}
    per_seed = []
    for render in renders:
        seed_row = {}
        for name in components:
            seed_row[name] = historical_relative(render[name], mean[name], illuminated)
        per_seed.append(seed_row)
    for name in components:
        p95_values = [row[name]["p95"] for row in per_seed]
        limit = HISTORICAL_REFERENCE_LIMITS["s0_p95"] if name == "s0" else HISTORICAL_REFERENCE_LIMITS["s1_p95"]
        uncertainty[name] = {
            "p95_vs_mean_max": float(max(p95_values)),
            "per_seed_p95": [float(value) for value in p95_values],
            "limit": limit,
            "denominator": ORIGINAL_DENOMINATOR_EXPRESSION,
        }
    stable = uncertainty["s0"]["p95_vs_mean_max"] <= HISTORICAL_REFERENCE_LIMITS["s0_p95"] and all(
        uncertainty[name]["p95_vs_mean_max"] <= HISTORICAL_REFERENCE_LIMITS["s1_p95"] for name in ("s1", "s2", "s3")
    )
    return {
        "stable": bool(stable),
        "uncertainty": uncertainty,
        "n_illuminated": int(illuminated.sum()),
        "mask": "mean S0 > 1e-4",
        "statistic": "max over seeds of pixel p95 of abs(render-mean)/np.maximum(component_mean, 1e-4)",
        "mean": mean,
        "illuminated": illuminated,
        "per_seed": per_seed,
    }


def corrected_object_mask(reference_s0: np.ndarray, object_ids: np.ndarray, valid: np.ndarray) -> np.ndarray:
    s0 = _finite(reference_s0, "reference S0")
    ids = np.asarray(object_ids)
    gate = np.asarray(valid, dtype=bool)
    _same_shape([s0, ids, gate], "corrected mask")
    return (s0 > S0_MIN) & (ids > 0) & gate


def interior_and_boundary(object_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(object_ids)
    interior = np.zeros(ids.shape, dtype=bool)
    if ids.shape[0] >= 3 and ids.shape[1] >= 3:
        center = ids[1:-1, 1:-1]
        same = np.ones(center.shape, dtype=bool)
        for row_shift in (-1, 0, 1):
            for col_shift in (-1, 0, 1):
                same &= ids[1 + row_shift : ids.shape[0] - 1 + row_shift, 1 + col_shift : ids.shape[1] - 1 + col_shift] == center
        interior[1:-1, 1:-1] = same & (center > 0)
    boundary = (ids > 0) & ~interior
    return interior, boundary


def corrected_relative(candidate: np.ndarray, reference_component: np.ndarray, reference_s0: np.ndarray, mask: np.ndarray) -> dict:
    """``abs(candidate - reference_component) / reference_S0`` on the supplied mask."""
    left = _finite(candidate, "candidate")
    component = _finite(reference_component, "reference component")
    s0 = _finite(reference_s0, "reference S0")
    gate = np.asarray(mask, dtype=bool)
    _same_shape([left, component, s0, gate], "corrected relative")
    if np.any(gate & ~(s0 > S0_MIN)):
        raise ValueError("corrected mask includes a pixel with reference S0 <= 1e-4")
    if not gate.any():
        row = summarize(np.array([]))
    else:
        row = summarize(np.abs(left - component)[gate] / s0[gate])
    row["denominator"] = "reference_S0"
    row["mask"] = "caller mask; qualified metrics use object id > 0, valid ground truth, and reference S0 > 1e-4"
    return row


def absolute_error(candidate: np.ndarray, reference_component: np.ndarray, mask: np.ndarray) -> dict:
    left = _finite(candidate, "candidate")
    right = _finite(reference_component, "reference")
    gate = np.asarray(mask, dtype=bool)
    _same_shape([left, right, gate], "absolute error")
    row = summarize(np.abs(left - right)[gate] if gate.any() else np.array([]))
    row["units"] = "radiance"
    return row


def polarization_from_stokes(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict[str, np.ndarray]:
    """Training-view DoLP and AoLP from one Stokes image. AoLP is 0 where it is masked."""
    s0 = _finite(s0, "S0")
    s1 = _finite(s1, "S1")
    s2 = _finite(s2, "S2")
    _same_shape([s0, s1, s2], "polarization")
    dolp = np.zeros(s0.shape, dtype=np.float64)
    np.divide(np.sqrt(s1 * s1 + s2 * s2), s0, out=dolp, where=s0 > S0_MIN)
    aolp = 0.5 * np.arctan2(s2, s1)
    aolp = np.where((s0 > S0_MIN) & (dolp >= DOLP_AOLP_MIN), aolp, 0.0)
    return {"dolp": dolp, "aolp": aolp}


def angular_difference_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)) % np.pi
    delta = np.minimum(delta, np.pi - delta)
    return np.degrees(delta)


def stokes_consistency(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray, mask: np.ndarray) -> dict:
    """Count physical-consistency violations. Do not drop the pixels."""
    s0 = _finite(s0, "S0")
    s1 = _finite(s1, "S1")
    s2 = _finite(s2, "S2")
    s3 = _finite(s3, "S3")
    gate = np.asarray(mask, dtype=bool)
    _same_shape([s0, s1, s2, s3, gate], "Stokes consistency")
    excess = s1 * s1 + s2 * s2 + s3 * s3 - np.maximum(s0, 0.0) ** 2
    violated = gate & ((s0 < 0.0) | (excess > 1e-6))
    return {
        "n_mask": int(gate.sum()),
        "n_violations": int(violated.sum()),
        "fraction": float(violated.sum() / gate.sum()) if gate.any() else None,
        "max_excess": float(excess[gate].max()) if gate.any() else None,
        "n_negative_s0": int(np.count_nonzero(gate & (s0 < 0.0))),
        "rule": "S0 >= 0 and S1^2 + S2^2 + S3^2 <= S0^2 + 1e-6",
        "pixels_discarded": 0,
    }


def stratify(error: np.ndarray, mask: np.ndarray, object_ids: np.ndarray, reference_s0: np.ndarray, channels: np.ndarray | None = None) -> dict:
    """Break one error image into object, S0 quantile, interior, and channel rows."""
    gate = np.asarray(mask, dtype=bool)
    ids = np.asarray(object_ids)
    s0 = _finite(reference_s0, "reference S0")
    values = _finite(error, "error")
    _same_shape([values, gate, ids, s0], "stratify")
    interior, boundary = interior_and_boundary(ids)
    rows = {
        "all": summarize(values[gate] if gate.any() else np.array([])),
        "interior": summarize(values[gate & interior] if np.any(gate & interior) else np.array([])),
        "boundary": summarize(values[gate & boundary] if np.any(gate & boundary) else np.array([])),
        "objects": {},
        "s0_quantiles": [],
    }
    present = sorted(int(item) for item in np.unique(ids[gate])) if gate.any() else []
    for value in present:
        chosen = gate & (ids == value)
        rows["objects"][str(value)] = summarize(values[chosen])
    if gate.any():
        edges = np.quantile(s0[gate], [0.0, 0.25, 0.5, 0.75, 1.0])
        for index in range(4):
            low, high = float(edges[index]), float(edges[index + 1])
            if index == 3:
                chosen = gate & (s0 >= low) & (s0 <= high)
            else:
                chosen = gate & (s0 >= low) & (s0 < high)
            row = summarize(values[chosen] if chosen.any() else np.array([]))
            row["s0_low"] = low
            row["s0_high"] = high
            rows["s0_quantiles"].append(row)
    if channels is not None:
        cube = _finite(channels, "channel error")
        if cube.shape[:2] != values.shape or cube.shape[2] != 3:
            raise ValueError(f"channel error has shape {cube.shape}, expected {(values.shape[0], values.shape[1], 3)}")
        rows["channels"] = {
            band: summarize(cube[:, :, index][gate] if gate.any() else np.array([]))
            for index, band in enumerate(("R", "G", "B"))
        }
    return rows


def sem_map(stack: np.ndarray) -> np.ndarray:
    """Sample standard deviation across seeds, divided by sqrt(n_reference)."""
    values = _finite(stack, "reference stack")
    if values.ndim != 3:
        raise ValueError(f"reference stack must be (seeds, height, width), got {values.shape}")
    if values.shape[0] < 2:
        raise ValueError("standard error needs at least two reference seeds")
    deviation = np.std(values, axis=0, ddof=1)
    return deviation / np.sqrt(values.shape[0])


def _bootstrap_indices(n_seeds: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n_seeds), size=(int(n_boot), int(n_seeds)))


def bootstrap_upper(samples: np.ndarray, quantile: float = BOOTSTRAP_UPPER_QUANTILE) -> float:
    return float(np.quantile(np.asarray(samples, dtype=np.float64), quantile))


def stokes_uncertainty(stack: np.ndarray, reference_s0: np.ndarray, mask: np.ndarray, *, n_boot: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> dict:
    """Spatial median and p95 of SEM/S0, plus the seed-bootstrap upper bounds."""
    values = _finite(stack, "Stokes stack")
    s0 = _finite(reference_s0, "reference S0")
    gate = np.asarray(mask, dtype=bool)
    _same_shape([values[0], s0, gate], "Stokes uncertainty")
    if np.any(gate & ~(s0 > S0_MIN)):
        raise ValueError("uncertainty mask includes reference S0 <= 1e-4")
    relative = np.zeros(s0.shape, dtype=np.float64)
    np.divide(sem_map(values), s0, out=relative, where=gate)
    point = summarize(relative[gate] if gate.any() else np.array([]))
    medians = []
    peaks = []
    for draw in _bootstrap_indices(values.shape[0], n_boot, seed):
        resampled = np.zeros(s0.shape, dtype=np.float64)
        np.divide(sem_map(values[draw]), s0, out=resampled, where=gate)
        row = summarize(resampled[gate] if gate.any() else np.array([]))
        medians.append(row["median"] if row["median"] is not None else np.nan)
        peaks.append(row["p95"] if row["p95"] is not None else np.nan)
    return {
        "point": point,
        "median_upper": bootstrap_upper(np.asarray(medians)),
        "p95_upper": bootstrap_upper(np.asarray(peaks)),
        "n_bootstrap": int(n_boot),
        "bootstrap_seed": int(seed),
        "n_reference": int(values.shape[0]),
        "denominator": "reference_S0",
        "estimator": "sample_std_ddof1 / sqrt(n_reference), seed bootstrap of the spatial median and p95",
        "pixels_are_seed_replicates": False,
    }


def derived_uncertainty(renders: list[dict[str, np.ndarray]], mask: np.ndarray, *, n_boot: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> dict:
    """Bootstrap DoLP and AoLP after averaging Stokes. Do not average the derived maps."""
    gate = np.asarray(mask, dtype=bool)
    mean = {name: np.mean(np.stack([item[name] for item in renders], axis=0), axis=0) for name in ("s0", "s1", "s2")}
    full = polarization_from_stokes(mean["s0"], mean["s1"], mean["s2"])
    dolp_median, dolp_p95 = [], []
    aolp_median, aolp_p95, aolp_counts = [], [], []
    draws = _bootstrap_indices(len(renders), n_boot, seed + 1)
    for draw in draws:
        chosen = [renders[int(index)] for index in draw]
        resampled = {name: np.mean(np.stack([item[name] for item in chosen], axis=0), axis=0) for name in ("s0", "s1", "s2")}
        derived = polarization_from_stokes(resampled["s0"], resampled["s1"], resampled["s2"])
        dolp_error = np.abs(derived["dolp"] - full["dolp"])
        row = summarize(dolp_error[gate] if gate.any() else np.array([]))
        dolp_median.append(row["median"] if row["median"] is not None else np.nan)
        dolp_p95.append(row["p95"] if row["p95"] is not None else np.nan)
        aolp_gate = gate & (full["dolp"] >= DOLP_AOLP_MIN) & (derived["dolp"] >= DOLP_AOLP_MIN)
        if aolp_gate.any():
            delta = angular_difference_deg(derived["aolp"], full["aolp"])
            angle = summarize(delta[aolp_gate])
            aolp_median.append(angle["median"])
            aolp_p95.append(angle["p95"])
            aolp_counts.append(angle["n"])
        else:
            aolp_median.append(np.nan)
            aolp_p95.append(np.nan)
            aolp_counts.append(0)
    return {
        "dolp": {
            "median_upper": bootstrap_upper(np.asarray(dolp_median)),
            "p95_upper": bootstrap_upper(np.asarray(dolp_p95)),
            "n": int(gate.sum()),
            "units": "DoLP fraction",
        },
        "aolp": {
            "median_upper": bootstrap_upper(np.asarray(aolp_median, dtype=np.float64)),
            "p95_upper": bootstrap_upper(np.asarray(aolp_p95, dtype=np.float64)),
            "n_min": int(min(aolp_counts) if aolp_counts else 0),
            "units": "degrees",
            "period_deg": 180.0,
        },
        "n_bootstrap": int(n_boot),
        "method": "resample seeds, average Stokes, then compute DoLP and AoLP",
    }


def reference_target_decision(stokes_rows: dict[str, dict], derived: dict) -> dict:
    """Upper bounds must meet every prospective limit. A point estimate is not enough."""
    checks = {}
    for name in ("s0", "s1", "s2", "s3"):
        limit = PROSPECTIVE_LIMITS[f"{name}_relative"]
        row = stokes_rows[name]
        checks[name] = {
            "median_upper": row["median_upper"],
            "p95_upper": row["p95_upper"],
            "median_limit": limit["median"],
            "p95_limit": limit["p95"],
            "pass": bool(row["median_upper"] <= limit["median"] and row["p95_upper"] <= limit["p95"]),
        }
    for name, key in (("dolp", "dolp_absolute"), ("aolp", "aolp_angular")):
        limit = PROSPECTIVE_LIMITS[key]
        row = derived[name]
        finite = np.isfinite(row["median_upper"]) and np.isfinite(row["p95_upper"])
        checks[name] = {
            "median_upper": row["median_upper"],
            "p95_upper": row["p95_upper"],
            "median_limit": limit["median"],
            "p95_limit": limit["p95"],
            "pass": bool(finite and row["median_upper"] <= limit["median"] and row["p95_upper"] <= limit["p95"]),
        }
    return {"pass": all(item["pass"] for item in checks.values()), "checks": checks, "n_seeds_note": "qualification requires at least 8 seeds; fewer seeds stay unqualified even if these bounds pass"}


def candidate_metrics(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    """Corrected candidate errors. This does not subtract a noise floor."""
    gate = np.asarray(mask, dtype=bool)
    out = {}
    for name in ("s0", "s1", "s2", "s3"):
        out[name] = corrected_relative(candidate[name], reference[name], reference["s0"], gate)
        out[name]["absolute"] = absolute_error(candidate[name], reference[name], gate)
    left = polarization_from_stokes(candidate["s0"], candidate["s1"], candidate["s2"])
    right = polarization_from_stokes(reference["s0"], reference["s1"], reference["s2"])
    dolp = summarize(np.abs(left["dolp"] - right["dolp"])[gate] if gate.any() else np.array([]))
    dolp["denominator"] = "absolute DoLP difference"
    aolp_gate = gate & (left["dolp"] >= DOLP_AOLP_MIN) & (right["dolp"] >= DOLP_AOLP_MIN)
    if aolp_gate.any():
        aolp = summarize(angular_difference_deg(left["aolp"], right["aolp"])[aolp_gate])
    else:
        aolp = summarize(np.array([]))
    aolp["units"] = "degrees"
    out["dolp"] = dolp
    out["aolp"] = aolp
    passed = []
    for name in ("s0", "s1", "s2", "s3"):
        limit = CANDIDATE_LIMITS[f"{name}_relative"]
        passed.append(out[name]["n"] > 0 and out[name]["median"] <= limit["median"] and out[name]["p95"] <= limit["p95"])
    limit = CANDIDATE_LIMITS["dolp_absolute"]
    passed.append(dolp["n"] > 0 and dolp["median"] <= limit["median"] and dolp["p95"] <= limit["p95"])
    limit = CANDIDATE_LIMITS["aolp_angular"]
    aolp_pass = aolp["n"] > 0 and aolp["median"] is not None and aolp["median"] <= limit["median"] and aolp["p95"] <= limit["p95"]
    out["aolp_status"] = "measured" if aolp["n"] else "not_assessable"
    passed.append(aolp_pass)
    out["pass"] = bool(all(passed))
    out["noise_floor_subtracted"] = False
    return out


def sampling_ratio(low_spp_p95: float, high_spp_p95: float, low_spp: int, high_spp: int) -> dict:
    """Compare a p95 at two spp values with the Monte Carlo 1/sqrt(spp) ratio."""
    if low_spp_p95 <= 0.0:
        return {"ratio": None, "expected_std_ratio": float(np.sqrt(low_spp / high_spp)), "decreases_with_spp": None, "consistent_with_monte_carlo": None}
    ratio = float(high_spp_p95 / low_spp_p95)
    expected = float(np.sqrt(low_spp / high_spp))
    return {
        "ratio": ratio,
        "expected_std_ratio": expected,
        "decreases_with_spp": bool(ratio < 0.95),
        "consistent_with_monte_carlo": bool(ratio <= expected * 1.35),
        "rule": "decreases_with_spp is ratio < 0.95; consistent_with_monte_carlo allows 35 percent above sqrt(spp_low/spp_high)",
    }


def summaries_match(reconstructed: float, historical: float, tolerance: float = RECONSTRUCTION_ABSOLUTE_TOLERANCE) -> bool:
    return abs(float(reconstructed) - float(historical)) <= float(tolerance)


def weak_illumination(reference_s0: np.ndarray, object_ids: np.ndarray, valid: np.ndarray) -> dict:
    """Pixels inside the object mask at or below the illumination threshold. Not an acceptance mask."""
    ids = np.asarray(object_ids)
    gate = np.asarray(valid, dtype=bool)
    s0 = _finite(reference_s0, "reference S0")
    chosen = gate & (ids > 0) & ~(s0 > S0_MIN)
    return {
        "n": int(chosen.sum()),
        "n_object": int((gate & (ids > 0)).sum()),
        "rule": "valid object pixels with reference S0 <= 1e-4",
        "included_in_acceptance_mask": False,
    }
