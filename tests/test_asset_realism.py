"""Ingest, frames, and semantic facing for the qualification composer."""

from __future__ import annotations

import numpy as np
import pytest

from sensor_sim.scene_gen.composition_rules import office
from sensor_sim.scene_gen.composition_solver import hard_failures, solve
from sensor_sim.scene_gen.gap_assets import chair, monitor
from sensor_sim.scene_gen.mesh_io import read_mesh, write_mesh
from sensor_sim.scene_gen.scene_critic import mesh_report
from sensor_sim.scene_gen.semantic_frame import FrameError, make_frame, require_frame, screen_agrees, sofa_forward_from_regions


def test_face_material_round_trip():
    mesh = {
        "vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        "corners": [[(0, None, None), (1, None, None), (2, None, None)], [(0, None, None), (2, None, None), (3, None, None)]],
        "face_materials": ["wood", "metal"],
    }
    loaded = read_mesh(write_mesh(mesh, name="parts"))
    assert loaded["face_materials"] == ["wood", "metal"]
    assert loaded["corners"][1][2][0] == 3


def test_frame_rejects_a_left_handed_monitor_normal():
    frame = make_frame((0.0, -1.0, 0.0), screen_normal=[0.0, -1.0, 0.0], screen_up=[0.0, 0.0, 1.0])
    require_frame({"semantic": frame})
    assert screen_agrees(frame["screen_normal"], [0.0, -1.0, 0.0])["passed"] is True
    assert screen_agrees(frame["screen_normal"], [0.0, 1.0, 0.0])["passed"] is False
    broken = dict(frame)
    broken["right"] = [-value for value in frame["right"]]
    with pytest.raises(FrameError, match="not_right_handed"):
        require_frame({"semantic": broken})


def test_sofa_forward_points_from_the_backrest():
    direction = sofa_forward_from_regions([0.0, 0.1, 0.4], [0.0, 0.8, 0.4])
    assert direction[1] > 0.9


def test_office_monitor_faces_the_chair_and_a_reversed_sofa_fails():
    instances, rules = office(4.0, 3.4)
    room = {"door": {"x0": 0.35, "x1": 1.25, "depth_m": 0.9}}
    solved = solve(instances, rules, room, seed=3, steps=4)
    assert solved["valid"] is True
    assert solved["scale_optimized"] is False
    monitor_mesh = monitor(1)
    assert mesh_report(monitor_mesh)["passed"] is True
    assert float(chair(1)["vertices"][:, 0].max()) != float(chair(2)["vertices"][:, 0].max())
    reversed_rules_scene = [dict(item) for item in instances]
    for item in reversed_rules_scene:
        if item["id"] == "monitor":
            item["screen_normal"] = [0.0, 1.0, 0.0]
            item["screen_up"] = [0.0, 0.0, -1.0]
    failed = solve(reversed_rules_scene, rules, room, seed=3, steps=0)
    reasons = [item["failure_reason"] for item in hard_failures(failed["results"])]
    assert "screen_faces" in reasons or "screen_up" in reasons
