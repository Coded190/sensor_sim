"""Synthetic answers for the convergence audit. These are not render passes."""

from __future__ import annotations

import numpy as np
import pytest

from sensor_sim.scene_gen.convergence_v2 import prepare_xml_text
from sensor_sim.scene_gen.metric_audit import (
    CANDIDATE_LIMITS,
    ORIGINAL_DENOMINATOR_EXPRESSION,
    PROTOCOL_TEXT,
    PROSPECTIVE_LIMITS,
    RECONSTRUCTION_ABSOLUTE_TOLERANCE,
    angular_difference_deg,
    candidate_metrics,
    corrected_object_mask,
    corrected_relative,
    historical_reference_decision,
    historical_relative,
    polarization_from_stokes,
    scalar_stokes,
    stokes_consistency,
    summaries_match,
)


def _image(value: float, shape=(4, 4)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float64)


def _names(s0=1.0, s1=0.0, s2=0.0, s3=0.0, shape=(2, 2)) -> dict[str, np.ndarray]:
    named = {}
    for band, level in (("R", 0.2), ("G", 0.3), ("B", 0.4)):
        named[band] = _image(level, shape)
    for component, value in (("S0", s0), ("S1", s1), ("S2", s2), ("S3", s3)):
        for band in ("R", "G", "B"):
            named[f"{component}.{band}"] = _image(value, shape)
    return named


def test_identical_arrays_have_zero_error():
    image = _image(0.4)
    mask = np.ones(image.shape, dtype=bool)
    row = corrected_relative(image, image, _image(0.8), mask)
    assert row["median"] == 0.0
    assert row["p95"] == 0.0
    assert row["max"] == 0.0
    assert row["n"] == 16


def test_known_s0_scale_is_the_relative_error():
    reference = _image(0.5)
    candidate = reference * 1.25
    mask = np.ones(reference.shape, dtype=bool)
    row = corrected_relative(candidate, reference, reference, mask)
    assert row["median"] == pytest.approx(0.25)
    assert row["denominator"] == "reference_S0"


def test_s1_perturbation_is_divided_by_reference_s0_not_by_max_s1():
    reference_s0 = _image(1.0)
    reference_s1 = _image(0.0)
    candidate_s1 = _image(0.02)
    mask = np.ones(reference_s0.shape, dtype=bool)
    corrected = corrected_relative(candidate_s1, reference_s1, reference_s0, mask)
    historical = historical_relative(candidate_s1, reference_s1, mask)
    assert corrected["median"] == pytest.approx(0.02)
    assert historical["denominator"] == ORIGINAL_DENOMINATOR_EXPRESSION
    assert historical["median"] == pytest.approx(0.02 / 1e-4)
    assert corrected["median"] != pytest.approx(historical["median"])


def test_negative_stokes_stay_signed_and_use_reference_s0():
    named = _names(s0=0.2, s1=-0.05)
    scalar = scalar_stokes(named)
    assert scalar["s1"][0, 0] == pytest.approx(-0.05)
    mask = np.ones(scalar["s0"].shape, dtype=bool)
    candidate = np.full(scalar["s1"].shape, -0.049)
    corrected = corrected_relative(candidate, scalar["s1"], scalar["s0"], mask)
    historical = historical_relative(candidate, scalar["s1"], mask)
    assert corrected["median"] == pytest.approx(0.001 / 0.2)
    assert historical["median"] == pytest.approx(0.001 / 1e-4)


def test_aolp_wraps_across_zero_and_one_hundred_eighty_degrees():
    def stokes(angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        doubled = np.deg2rad(2.0 * angle_deg)
        return _image(1.0), _image(np.cos(doubled)), _image(np.sin(doubled))

    near_zero = polarization_from_stokes(*stokes(1.0))
    near_period = polarization_from_stokes(*stokes(179.0))
    delta = angular_difference_deg(near_zero["aolp"], near_period["aolp"])
    assert delta[0, 0] == pytest.approx(2.0, abs=1e-6)
    branch_high = polarization_from_stokes(*stokes(89.0))
    branch_low = polarization_from_stokes(*stokes(-89.0))
    wrapped = angular_difference_deg(branch_high["aolp"], branch_low["aolp"])
    assert wrapped[0, 0] == pytest.approx(2.0, abs=1e-6)


def test_illumination_threshold_and_background_are_excluded():
    s0 = np.array([[1e-4, 1e-4 + 1e-8], [0.2, 0.3]])
    ids = np.array([[1, 1], [0, 1]])
    valid = np.ones((2, 2), dtype=bool)
    mask = corrected_object_mask(s0, ids, valid)
    assert mask.tolist() == [[False, True], [False, True]]
    with pytest.raises(ValueError, match="reference S0"):
        corrected_relative(s0, s0, s0, np.ones_like(mask))


def test_channel_order_shape_and_non_finite_values_are_rejected():
    named = _names()
    named["S0.R"] = np.zeros((3, 3))
    with pytest.raises(ValueError, match="mismatch"):
        scalar_stokes(named)
    dropped = _names()
    del dropped["S3.B"]
    with pytest.raises(ValueError, match="missing"):
        scalar_stokes(dropped)
    bad = _names()
    bad["S1.G"] = np.array([[np.nan, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="non-finite"):
        scalar_stokes(bad)


def test_dolp_and_aolp_come_from_the_stokes_arrays():
    s0, s1, s2 = _image(0.4), _image(0.2), _image(-0.1)
    maps = polarization_from_stokes(s0, s1, s2)
    assert maps["dolp"][0, 0] == pytest.approx(np.sqrt(0.2**2 + 0.1**2) / 0.4)
    assert maps["aolp"][0, 0] == pytest.approx(0.5 * np.arctan2(-0.1, 0.2))
    swapped = polarization_from_stokes(s0, s2, s1)
    assert swapped["aolp"][0, 0] != pytest.approx(maps["aolp"][0, 0])


def test_historical_decision_uses_the_component_denominator_and_can_fail():
    bright = {"s0": _image(1.0), "s1": _image(0.0), "s2": _image(0.0), "s3": _image(0.0)}
    shifted = {"s0": _image(1.0), "s1": _image(0.002), "s2": _image(0.0), "s3": _image(0.0)}
    decision = historical_reference_decision([bright, bright, bright, shifted])
    assert decision["uncertainty"]["s1"]["denominator"] == ORIGINAL_DENOMINATOR_EXPRESSION
    assert decision["uncertainty"]["s1"]["p95_vs_mean_max"] > decision["uncertainty"]["s1"]["limit"]
    assert decision["stable"] is False
    assert decision["n_illuminated"] == 16


def test_consistency_violations_are_counted_and_kept():
    s0 = _image(0.1)
    s1 = _image(0.2)
    report = stokes_consistency(s0, s1, _image(0.0), _image(0.0), np.ones(s0.shape, dtype=bool))
    assert report["n_violations"] == 16
    assert report["pixels_discarded"] == 0
    assert report["n_mask"] == 16


def test_candidate_does_not_pass_by_borrowing_the_historical_denominator():
    reference = {"s0": _image(1.0), "s1": _image(0.0), "s2": _image(0.0), "s3": _image(0.0)}
    candidate = {"s0": _image(1.0), "s1": _image(0.03), "s2": _image(0.0), "s3": _image(0.0)}
    mask = np.ones((4, 4), dtype=bool)
    row = candidate_metrics(candidate, reference, mask)
    assert row["s1"]["median"] == pytest.approx(0.03)
    assert row["s1"]["median"] > CANDIDATE_LIMITS["s1_relative"]["median"]
    assert row["pass"] is False


def test_xml_spp_default_is_replaced_without_touching_float_names():
    source = '<default name="spp" value="16" /><film type="hdrfilm" name="film"><integer name="width" value="128"/></film>'
    edited = prepare_xml_text(source, 1024, float32_film=True)
    assert 'value="1024"' in edited
    assert 'value="16"' not in edited
    assert 'component_format" value="float32"' in edited
    assert prepare_xml_text(edited, 256, float32_film=True).count("component_format") == 1


def test_dataset_contract_records_the_approved_64_scene_target():
    from pathlib import Path

    import yaml

    from sensor_sim.scene_gen.dataset_contract import refuse_production_generation, validate_dataset_contract

    spec = yaml.safe_load(Path("configs/scenes/dataset_v1.yaml").read_text())
    report = validate_dataset_contract(spec)
    assert report["schema_valid"] is True
    assert report["contract_complete"] is True
    assert report["generation_authorized"] is True
    assert report["ready_to_generate"] is True
    assert report["creates_samples"] is False
    assert spec["successful_scenes"] == 64
    assert spec["quotas"]["status"] == "approved"
    assert len(spec["strata"]) == 64
    blocked = dict(spec)
    blocked["generation_authorized"] = False
    with pytest.raises(RuntimeError, match="before export"):
        refuse_production_generation(blocked)
    incomplete = dict(spec)
    incomplete["successful_scenes"] = None
    incomplete["generation_authorized"] = False
    refused = validate_dataset_contract(incomplete)
    assert refused["contract_complete"] is False
    assert "successful scene target is not frozen" in refused["blockers"]


def test_protocol_keeps_the_historical_failure_and_the_fixed_limits():
    assert "stable: false" in PROTOCOL_TEXT
    assert ORIGINAL_DENOMINATOR_EXPRESSION in PROTOCOL_TEXT
    assert "0.0375" in PROTOCOL_TEXT
    assert "0.15" in PROTOCOL_TEXT
    assert PROSPECTIVE_LIMITS["s0_relative"]["p95"] == pytest.approx(0.0375)
    assert PROSPECTIVE_LIMITS["aolp_angular"]["p95"] == pytest.approx(2.5)
    assert CANDIDATE_LIMITS["s0_relative"]["p95"] == pytest.approx(0.15)
    assert RECONSTRUCTION_ABSOLUTE_TOLERANCE == 1e-5
    assert summaries_match(0.060615128579098766, 0.060615128579098766)
    assert not summaries_match(0.060615128579098766, 0.061)
