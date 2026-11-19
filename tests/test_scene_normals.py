"""The policy-scene normal comparison uses the evaluation poses and the filmed object."""

from __future__ import annotations

import numpy as np
import pytest

from sensor_sim.policy.scene_normals import evaluation_poses, require_filmed_policy_scene, score_mask


def test_evaluation_poses_match_the_scored_rng():
    poses = evaluation_poses()
    assert len(poses) == 30
    assert [pose["task"] for pose in poses[:3]] == ["slot", "tip", "lift"]
    first = poses[0]
    assert first["folder"] == "001_ellipsoid"
    assert first["family"] == "ellipsoid"
    assert first["xy"][0] == pytest.approx(0.0009457299760205345)
    assert first["xy"][1] == pytest.approx(0.05603709570607482)
    assert first["yaw"] == pytest.approx(0.9057815605287021)
    assert {pose["folder"][:3] for pose in poses} == {f"{index:03d}" for index in range(1, 11)}


def test_policy_scene_must_keep_the_dielectric_film():
    scene = {
        "integrator": {"type": "stokes", "integrator": {"type": "path", "max_depth": 8}},
        "film-shell": {
            "filename": "meshes/film.obj",
            "bsdf": {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0},
        },
        "solid": {"filename": "meshes/solid.obj", "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.5}}},
    }
    for index in range(7):
        scene[f"arm-{index}"] = {"type": "cube", "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.45}}}
    require_filmed_policy_scene(scene)
    bare = dict(scene)
    del bare["film-shell"]
    try:
        require_filmed_policy_scene(bare)
    except RuntimeError as exc:
        assert "film-shell" in str(exc)
    else:
        raise AssertionError("a scene without the film shell was accepted")


def test_score_mask_drops_dark_film_pixels():
    shape = np.zeros((2, 2), dtype=np.float64)
    shape[0, 0] = 3
    shape[0, 1] = 3
    normal = np.zeros((2, 2, 3), dtype=np.float64)
    normal[..., 2] = 1
    measurement = {
        "shape_index": shape,
        "film_index": 3,
        "s0": np.array([[1e-3, 1e-6], [1.0, 1.0]]),
        "normal": normal,
    }
    mask = score_mask(measurement)
    assert mask.tolist() == [[True, False], [False, False]]
