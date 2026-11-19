"""Qualification scenes repeat their manifests and pass support checks."""

from __future__ import annotations

from pathlib import Path

from sensor_sim.scene_gen.indoor_cli import assemble_demonstration


def test_three_scenes_repeat_without_rendering(tmp_path: Path):
    first = assemble_demonstration(tmp_path / "catalog", tmp_path / "first", render=False)
    second = assemble_demonstration(tmp_path / "catalog", tmp_path / "second", render=False)
    assert set(first) == {"scene_a", "scene_b", "scene_c"}
    for scene_id in first:
        assert first[scene_id]["contacts_passed"]
        assert first[scene_id]["support_acyclic"]
        assert first[scene_id]["collision_unintended"] == 0
        assert first[scene_id]["identity_count"] == {"scene_a": 6, "scene_b": 8, "scene_c": 11}[scene_id]
        left = (tmp_path / "first" / scene_id / "scene.json").read_bytes()
        right = (tmp_path / "second" / scene_id / "scene.json").read_bytes()
        assert left == right
        text = left.decode("utf-8")
        assert '"qualification": true' in text
        assert "not visually reviewed" in text
