"""Pad separation and the no-contact force threshold."""

import numpy as np

from sensor_sim.scene_gen.gripper_geometry import axis_aligned_bounds, facing_indices, noise_threshold, pad_measurement, pinch_measurement, point_in_bounds


def test_zero_baseline_uses_strict_positive_force():
    result = noise_threshold(np.zeros(100))
    assert result["method"] == "strict_positive_zero_baseline"
    assert result["threshold_n"] == 0.0


def test_threshold_is_p999_plus_three_mad():
    values = np.array([0.0, 0.0, 0.0, 0.01])
    result = noise_threshold(values)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    assert result["threshold_n"] == np.quantile(values, 0.999) + 3.0 * mad
    assert result["method"] == "p99.9_plus_3_mad"


def test_facing_pads_report_a_positive_gap():
    fixed = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01], [0.0, 0.01, 0.0], [0.02, 0.0, 0.0]])
    moving = fixed + np.array([0.04, 0.0, 0.0])
    measured = pad_measurement(fixed, moving)
    assert measured["separation_m"] > 0.015
    assert measured["closing_axis_world"][0] > 0.9
    assert facing_indices(fixed, moving.mean(axis=0)).size > 0


def test_pinch_gap_ignores_the_finger_axis():
    finger = np.array([0.0, 0.0, 1.0])
    fixed = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.04], [0.01, 0.0, 0.02]])
    moving = np.array([[0.04, 0.0, 0.0], [0.04, 0.0, 0.04], [0.03, 0.0, 0.05]])
    measured = pinch_measurement(fixed, moving, finger)
    assert measured["separation_m"] == np.float64(0.02) or abs(measured["separation_m"] - 0.02) < 1e-9
    assert measured["closing_axis_world"][0] > 0.99
    assert abs(measured["closing_axis_world"][2]) < 1e-9


def test_bounds_margin_includes_the_face_and_rejects_the_wrist():
    points = np.array([[0.0, 0.0, 0.1], [0.01, 0.0, 0.1]])
    bounds = axis_aligned_bounds(points, margin_m=0.002)
    assert point_in_bounds(np.array([0.0, 0.0, 0.1]), bounds)
    assert not point_in_bounds(np.array([0.0, 0.0, 0.0]), bounds)
