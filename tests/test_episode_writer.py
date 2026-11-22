"""The episode record is a new format. It does not touch Dataset v1."""

from pathlib import Path

import pytest

from sensor_sim.dataset.episode_writer import EpisodeError, read_episode, transform_sha256, write_episode


def _frame(index: int) -> dict:
    return {
        "frame_index": index,
        "simulation_step": index,
        "simulation_time_s": index * 0.01,
        "joint_position": [0.0, 0.1, -0.2, 0.3, 0.0, 0.5],
        "joint_velocity": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "action": [0.0, 0.1, -0.2, 0.3, 0.0, 0.5],
        "action_mode": "joint_position_target",
        "gripper_command": 0.5,
        "gripper_state": 0.4,
        "object_pose": [-0.04, -0.18, 0.86, 1.0, 0.0, 0.0, 0.0],
        "object_velocity": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "contact": {"semantics": "not_measured", "value": None},
    }


def _episode() -> dict:
    frames = [_frame(0), _frame(1), _frame(2)]
    return {
        "schema_version": "episode_record_v1",
        "episode_id": "episode_000",
        "scene_id": "pick_and_place_1",
        "scene_manifest_sha256": "ab" * 32,
        "task_id": "pick_and_place",
        "asset_ids": ["table_dining", "can"],
        "physics_dt_s": 0.01,
        "recording_stride": 1,
        "outcome": "synthetic",
        "termination_reason": "fixture",
        "frames": frames,
        "mitsuba_snapshots": [
            {"frame_index": 0, "simulation_time_s": 0.0, "transform_sha256": transform_sha256({"can_0": frames[0]["object_pose"]}), "rgb": "initial.png", "stokes": "initial.exr"},
            {"frame_index": 2, "simulation_time_s": 0.02, "transform_sha256": transform_sha256({"can_0": frames[2]["object_pose"]}), "rgb": "final.png", "stokes": "final.exr"},
        ],
    }


def test_round_trip_and_time_order(tmp_path: Path):
    written = write_episode(tmp_path / "episode", _episode())
    loaded = read_episode(tmp_path / "episode")
    assert loaded["episode_id"] == written["episode_id"]
    assert loaded["frames"][1]["simulation_time_s"] > loaded["frames"][0]["simulation_time_s"]


def test_a_decreasing_timestamp_is_rejected(tmp_path: Path):
    episode = _episode()
    episode["frames"][1]["simulation_time_s"] = 0.0
    with pytest.raises(EpisodeError, match="time_not_increasing"):
        write_episode(tmp_path / "episode", episode)


def test_joint_targets_are_not_stored_as_measured_positions():
    frame = _frame(0)
    assert frame["action_mode"] == "joint_position_target"
    assert "joint_position" in frame and "action" in frame
