"""The indoor dispatcher selects Infinigen and does not fall back to box assets."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sensor_sim.scene_gen.indoor_cli import assemble_demonstration
from sensor_sim.scene_gen.indoor_generate import IndoorGenerateError, generate
from sensor_sim.scene_gen.infinigen_backend import coarse_command


def test_missing_install_fails(tmp_path: Path):
    with pytest.raises(IndoorGenerateError, match="infinigen_not_installed"):
        generate({"kind": "room", "category": "LivingRoom", "seed": 1, "output": tmp_path}, indoors_root=tmp_path / "missing", indoors_python=tmp_path / "missing-python")


def test_missing_generator_does_not_import_gap_assets(tmp_path: Path):
    root = tmp_path / "indoors"
    (root / "infinigen_examples").mkdir(parents=True)
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    sys.modules.pop("sensor_sim.scene_gen.gap_assets", None)
    with pytest.raises(IndoorGenerateError, match="generator_missing"):
        generate({"kind": "room", "category": "Workshop", "seed": 1, "output": tmp_path / "out", "execute": False}, indoors_root=root, indoors_python=python)
    assert "sensor_sim.scene_gen.gap_assets" not in sys.modules


def test_project_source_requires_an_explicit_flag():
    with pytest.raises(IndoorGenerateError, match="source_override_not_explicit"):
        generate({"kind": "asset", "category": "chair", "source": "sensor_sim"})
    routed = generate({"kind": "asset", "category": "chair", "source": "sensor_sim", "explicit_source": True})
    assert routed["source"] == "sensor_sim"


def test_normal_room_request_names_infinigen_before_execution(tmp_path: Path):
    root = tmp_path / "indoors"
    (root / "infinigen_examples").mkdir(parents=True)
    (root / ".git").mkdir()
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    routed = generate({"kind": "room", "category": "LivingRoom", "seed": 2, "output": tmp_path / "out", "execute": False}, indoors_root=root, indoors_python=python)
    assert routed["source"] == "infinigen_indoors"
    assert routed["category"] == "LivingRoom"


def test_legacy_indoor_cli_still_builds_three_scenes(tmp_path: Path):
    result = assemble_demonstration(tmp_path / "catalog", tmp_path / "scenes", render=False)
    assert set(result) == {"scene_a", "scene_b", "scene_c"}


def test_full_solve_command_omits_fast_gin(tmp_path: Path):
    command = coarse_command("Bathroom", 7, tmp_path, fast=False, python=Path("python"))
    assert "fast_solve.gin" not in command
    assert "Bathroom" in " ".join(command)
