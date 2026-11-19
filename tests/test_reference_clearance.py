"""Analytic cases for the audit clearance checker. It does not call mesh_clearance."""

from __future__ import annotations

import numpy as np
import pytest

from sensor_sim.scene_gen.reference_clearance import reference_clearance, reference_contains, reference_floor


def _sheet(z: float) -> dict:
    return {
        "vertices": np.array([[0.0, 0.0, z], [1.0, 0.0, z], [0.0, 1.0, z]], dtype=np.float64),
        "faces": np.array([[0, 1, 2]], dtype=np.int64),
    }


def test_parallel_triangles_match_a_5_mm_gap_and_reject_4_mm():
    separated = reference_clearance(_sheet(0.0), _sheet(0.005))
    close = reference_clearance(_sheet(0.0), _sheet(0.004))
    assert separated["clearance_m"] == pytest.approx(0.005, abs=1e-6)
    assert close["clearance_m"] == pytest.approx(0.004, abs=1e-6)
    assert close["clearance_m"] < 0.005


def test_a_vertex_on_the_other_triangle_has_zero_clearance():
    first = _sheet(0.0)
    second = {
        "vertices": np.array([[0.2, 0.2, 0.0], [0.2, 0.2, 1.0], [0.8, 0.2, 1.0]], dtype=np.float64),
        "faces": np.array([[0, 1, 2]], dtype=np.int64),
    }
    report = reference_clearance(first, second)
    assert report["clearance_m"] == pytest.approx(0.0, abs=1e-6)
    assert report["surface_intersection"]


def test_overlapping_boxes_can_still_be_separated():
    low = {"vertices": np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]]), "faces": np.array([[0, 1, 2]])}
    high = {"vertices": np.array([[0.05, 0.05, 0.008], [0.25, 0.05, 0.008], [0.05, 0.25, 0.008]]), "faces": np.array([[0, 1, 2]])}
    report = reference_clearance(low, high)
    assert report["clearance_m"] == pytest.approx(0.008, abs=1e-6)
    assert not report["surface_intersection"]


def test_floor_contact_and_penetration_and_containment():
    contact = reference_floor(np.array([[0.0, 0.0, 0.001], [1.0, 0.0, 0.002]]))
    penetrated = reference_floor(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.002]]))
    assert not contact["penetrating"]
    assert penetrated["penetrating"]
    cube_vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cube_faces = np.array(
        [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]],
        dtype=np.int64,
    )
    assert reference_contains(np.array([0.2, -0.3, 0.4]), cube_vertices, cube_faces)
    assert not reference_contains(np.array([2.0, 0.0, 0.0]), cube_vertices, cube_faces)
