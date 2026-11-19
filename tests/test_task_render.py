"""Saved-bundle rendering rejects bad inputs and leaves the source untouched."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sensor_sim.scene_gen.task_render import (
    _mesh_file,
    prepare_bundle,
    refuse_output,
    render_saved_bundle,
    require_eligible,
)
from sensor_sim.scene_gen.task_schema import TaskError
from sensor_sim.scene_gen.metric_audit import CHANNEL_NAMES

SOURCE = Path("dataset/qualification/task_scenes/book_on_table/e54a5e18636df8e7")
CATALOG = Path("assets/candidates")


def _copy(tmp_path: Path) -> Path:
    destination = tmp_path / "bundle"
    shutil.copytree(SOURCE, destination)
    return destination


def _hashes(path: Path) -> dict[str, str]:
    import hashlib

    return {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in ("bundle.json", "scene.json")}


def test_eligibility_and_forbidden_paths():
    with pytest.raises(TaskError, match="license"):
        require_eligible({"provenance": {"license_status": "unresolved"}}, "renderer_validated")
    with pytest.raises(TaskError, match="renderer_status"):
        require_eligible({"provenance": {"license_status": "permitted"}}, "technically_validated")
    with pytest.raises(TaskError, match="output_forbidden"):
        refuse_output(Path("dataset/qualification/indoor_scenes/scene_a/extra"))
    with pytest.raises(TaskError, match="path_escape"):
        _mesh_file(CATALOG, "../room_floor", "1.0.0")


def test_rejects_bad_bundle_inputs(tmp_path: Path):
    copied = _copy(tmp_path)
    scene = json.loads((copied / "scene.json").read_text(encoding="utf-8"))
    bundle = json.loads((copied / "bundle.json").read_text(encoding="utf-8"))
    scene["instances"][0]["package_sha256"] = "0" * 64
    bundle["resolved"]["assets"][0]["package_sha256"] = "0" * 64
    (copied / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    (copied / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(TaskError, match="package_hash"):
        prepare_bundle(copied, CATALOG)

    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "bundle.json").write_text("{}", encoding="utf-8")
    with pytest.raises(TaskError, match="missing_bundle"):
        prepare_bundle(missing, CATALOG)

    copied = _copy(tmp_path / "matrix")
    scene = json.loads((copied / "scene.json").read_text(encoding="utf-8"))
    bundle = json.loads((copied / "bundle.json").read_text(encoding="utf-8"))
    scene["instances"][0]["matrix"] = [0.0] * 16
    pose = next(item for item in bundle["resolved"]["poses"] if item["instance_id"] == scene["instances"][0]["instance_id"])
    pose["matrix"] = [0.0] * 16
    (copied / "scene.json").write_text(json.dumps(scene, sort_keys=True), encoding="utf-8")
    (copied / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(Exception):
        prepare_bundle(copied, CATALOG)


def test_saved_bundle_renders_without_changing_source(tmp_path: Path):
    before = _hashes(SOURCE)
    prepared = prepare_bundle(SOURCE, CATALOG)
    assert prepared["replay_byte_match"] is True
    assert _hashes(SOURCE) == before
    first = render_saved_bundle(SOURCE, CATALOG, tmp_path / "first", resolution=(32, 24), spp=2)
    second = render_saved_bundle(SOURCE, CATALOG, tmp_path / "second", resolution=(32, 24), spp=2)
    assert first["rendered"] is True
    assert first["exr_names"] == list(CHANNEL_NAMES)
    assert first["exr_dtype"] == "float32"
    assert [item["matrix"] for item in first["assets"]] == [item["matrix"] for item in second["assets"]]
    assert _hashes(SOURCE) == before
    with pytest.raises(TaskError, match="output_exists"):
        render_saved_bundle(SOURCE, CATALOG, tmp_path / "first", resolution=(32, 24), spp=2)
