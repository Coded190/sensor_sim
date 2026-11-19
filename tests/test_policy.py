"""Arm predicates and the second-image connection, without a Mitsuba render."""

from __future__ import annotations

import numpy as np
import torch

from sensor_sim.policy.arm import _wrap, carried_tilt
from sensor_sim.policy.nets import denoise_epsilon
from sensor_sim.policy.observe import pose_matrix
from sensor_sim.policy.tasks import scripted_episode, success
from sensor_sim.policy.train import assert_second_image_is_connected


def test_pose_matrix_moves_the_catalog_point():
    matrix = pose_matrix((0.1, -0.02), 0.0, (-0.03, 0.04), 0.0, 0.0, 0.05)
    moved = matrix @ np.array([0.1, -0.02, 0.05, 1.0])
    assert np.allclose(moved[:3], [-0.03, 0.04, 0.05], atol=1e-6)


def test_expert_succeeds_on_the_three_tasks():
    for task, family, xy in (
        ("lift", "cone", (0.0, 0.02)),
        ("tip", "capsule", (0.02, 0.0)),
        ("slot", "ellipsoid", (-0.02, 0.03)),
    ):
        episode = scripted_episode(task, family, xy, 0.4, 0.04)
        assert episode["success"], task
        assert episode["actions"].shape == (40, 5)


def test_lift_requires_a_held_rise():
    rest = {"position": np.array([0.0, 0.0, 0.04]), "yaw": 0.0, "tilt": 0.0, "attached": True}
    low = [rest] * 6
    assert success("lift", "cone", 0.04, low) is False
    raised = dict(rest)
    raised["position"] = np.array([0.0, 0.0, 0.07])
    assert success("lift", "cone", 0.04, [rest, raised, raised, raised, raised, raised]) is True


def test_tilt_clears_only_after_two_centimeters_of_rise():
    assert carried_tilt(1.2, 0.0) == 1.2
    assert abs(carried_tilt(1.2, 0.01) - 0.6) < 1e-9
    assert carried_tilt(1.2, 0.02) == 0.0


def test_yaw_command_uses_the_short_arc():
    current = -np.pi + 0.1
    target = np.pi - 0.1
    assert abs(target - current) > 6.0
    assert abs(_wrap(target - current)) < 0.3


def test_denoiser_recovers_a_clean_action_from_the_true_epsilon():
    clean = torch.tensor([0.2, -0.1, 0.05, 0.4, 0.0])
    epsilon = torch.tensor([0.3, -0.2, 0.1, 0.0, -0.4])
    start = clean + epsilon

    def predict(sample, scale):
        return (sample - clean) / scale

    recovered = denoise_epsilon(start, predict)
    assert torch.allclose(recovered, clean, atol=1e-5)


def test_second_image_changes_the_action_only_after_an_update():
    torch.manual_seed(0)
    assert_second_image_is_connected("cpu")
