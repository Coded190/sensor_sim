"""Hash-linked scene_state_snapshot_v1 records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sensor_sim.scene_gen.scene_state import (
    SceneStateError,
    content_hash,
    read_jsonl,
    stamp,
    translation_rotation_error,
    validate_sequence,
    validate_snapshot,
    write_jsonl,
    write_snapshot,
)

IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _snapshot(time_s: float, frame: int, x: float) -> dict:
    pose = [row[:] for row in IDENTITY]
    pose[0][3] = x
    return {
        "snapshot_id": f"t{frame}",
        "scene_id": "pick_and_place_1",
        "manifest_sha256": "aa" * 32,
        "catalog_sha256": "bb" * 32,
        "scene_resolution_sha256": "cc" * 32,
        "frame_index": frame,
        "physics_step": frame * 2,
        "time_s": time_s,
        "timestep_s": 1.0 / 120.0,
        "backend": {"name": "isaaclab", "isaaclab": "2.3.0", "isaacsim": "5.1.0", "physx": "test"},
        "coordinate_convention": {"up": "z", "units": "m", "quaternion": "wxyz", "handedness": "right"},
        "instances": [
            {
                "instance_id": "table_0",
                "asset_id": "table_dining",
                "package_sha256": "dd" * 32,
                "world_from_asset": IDENTITY,
                "linear_velocity_m_s": [0, 0, 0],
                "angular_velocity_rad_s": [0, 0, 0],
                "dynamic": False,
            },
            {
                "instance_id": "can_0",
                "asset_id": "can",
                "package_sha256": "ee" * 32,
                "world_from_asset": pose,
                "linear_velocity_m_s": [0.2, 0, 0],
                "angular_velocity_rad_s": [0, 0, 0],
                "dynamic": True,
            },
        ],
    }


def test_content_hash_excludes_itself_and_reloads(tmp_path: Path):
    stamped = stamp(_snapshot(0.0, 0, 0.0))
    assert "content_hash" not in {key: value for key, value in stamped.items() if key == "missing"}
    body = {key: value for key, value in stamped.items() if key != "content_hash"}
    assert stamped["content_hash"] == content_hash(body)
    loaded = write_snapshot(tmp_path / "t0.json", _snapshot(0.0, 0, 0.0))
    validate_snapshot(loaded)
    changed = dict(loaded)
    changed["time_s"] = 1.0
    with pytest.raises(SceneStateError, match="content_hash_mismatch"):
        validate_snapshot(changed)


def test_sequence_rejects_a_decreasing_time(tmp_path: Path):
    snapshots = [_snapshot(0.0, 0, 0.0), _snapshot(0.1, 6, 0.02), _snapshot(0.4, 24, 0.08)]
    digest = write_jsonl(tmp_path / "states.jsonl", snapshots)
    loaded, again = read_jsonl(tmp_path / "states.jsonl")
    assert digest == again
    assert len(loaded) == 3
    meters, degrees = translation_rotation_error(
        np.asarray(loaded[0]["instances"][1]["world_from_asset"]),
        np.asarray(loaded[2]["instances"][1]["world_from_asset"]),
    )
    assert meters == pytest.approx(0.08, abs=1e-9)
    assert degrees == pytest.approx(0.0, abs=1e-6)
    backwards = [stamp(_snapshot(0.4, 24, 0.0)), stamp(_snapshot(0.2, 12, 0.0)), stamp(_snapshot(0.1, 6, 0.0))]
    with pytest.raises(SceneStateError, match="time_not_monotonic"):
        validate_sequence(backwards)
