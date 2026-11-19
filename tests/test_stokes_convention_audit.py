"""Equation tests for the analyzer convention. They do not prove Mitsuba's basis."""

from __future__ import annotations

import numpy as np
import pytest

from sensor_sim.film.polar import analyzer_images


def _intensities(s0, s1, s2):
    images = analyzer_images(np.array(s0), np.array(s1), np.array(s2))
    return {key: float(np.asarray(value)) for key, value in images.items()}


@pytest.mark.parametrize(
    ("stokes", "expected"),
    [
        ((1.0, 1.0, 0.0), {"analyzer_000": 1.0, "analyzer_090": 0.0, "analyzer_045": 0.5, "analyzer_135": 0.5}),
        ((1.0, -1.0, 0.0), {"analyzer_000": 0.0, "analyzer_090": 1.0, "analyzer_045": 0.5, "analyzer_135": 0.5}),
        ((1.0, 0.0, 1.0), {"analyzer_045": 1.0, "analyzer_135": 0.0, "analyzer_000": 0.5, "analyzer_090": 0.5}),
        ((1.0, 0.0, -1.0), {"analyzer_045": 0.0, "analyzer_135": 1.0, "analyzer_000": 0.5, "analyzer_090": 0.5}),
        ((1.0, 0.0, 0.0), {"analyzer_000": 0.5, "analyzer_045": 0.5, "analyzer_090": 0.5, "analyzer_135": 0.5}),
    ],
)
def test_known_linear_and_unpolarized_states(stokes, expected):
    got = _intensities(*stokes)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value)
    assert got["analyzer_000"] + got["analyzer_090"] == pytest.approx(stokes[0])
    assert got["analyzer_045"] + got["analyzer_135"] == pytest.approx(stokes[0])


def test_double_angle_rotation_mixes_s1_and_s2():
    s1, s2 = 0.4, -0.2
    psi = 0.3
    rotated_s1 = s1 * np.cos(2.0 * psi) + s2 * np.sin(2.0 * psi)
    rotated_s2 = -s1 * np.sin(2.0 * psi) + s2 * np.cos(2.0 * psi)
    assert rotated_s1 == pytest.approx(0.4 * np.cos(0.6) + -0.2 * np.sin(0.6))
    assert rotated_s1**2 + rotated_s2**2 == pytest.approx(s1**2 + s2**2)
