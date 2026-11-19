"""Schema, retrieval, sizing, layout, and replay for task scenes."""

from __future__ import annotations

from pathlib import Path

import pytest

from sensor_sim.scene_gen.task_cli import book_inside_crate, evaluation_tasks, reachable_task
from sensor_sim.scene_gen.task_replay import generate, replay
from sensor_sim.scene_gen.task_retrieve import rank_key, retrieve
from sensor_sim.scene_gen.task_schema import TaskError, load_task
from sensor_sim.scene_gen.task_size import fits_interior


def test_schema_rejects_duplicate_keys_and_bad_ids():
    with pytest.raises(TaskError, match="duplicate_json_key"):
        load_task('{"schema_version": 1, "schema_version": 1}\n')
    task = book_inside_crate()
    task["task_id"] = "../escape"
    with pytest.raises(TaskError, match="task_id"):
        load_task(__import__("json").dumps({"schema_version": 1, "task": task}))


def test_retrieval_filters_and_tie_break():
    catalog = Path("assets/candidates")
    book = retrieve(catalog, "book", top_k=10, require_renderer=True)
    assert book["results"][0]["asset_id"] == "prop_book"
    assert book["returned_count"] < 10
    crate = retrieve(catalog, "crate", top_k=10, require_renderer=True)
    assert any(item["asset_id"] == "prop_crate_open" for item in crate["results"])
    assert all(item["license_status"] == "permitted" for item in crate["results"])
    table = retrieve(catalog, "table", top_k=10, require_renderer=True)
    assert all(not item["asset_id"].startswith("infinigen") for item in table["results"])
    assert any(item["rejection"] == "license" and item["asset_id"] == "infinigen2_table" for item in table["rejected"])
    sphere = retrieve(catalog, "sphere", top_k=10, require_renderer=True)
    assert all(item["asset_id"] != "fixture_sphere" for item in sphere["results"])
    discovery = retrieve(catalog, "sphere", top_k=10, require_renderer=False)
    assert any(item["asset_id"] == "fixture_sphere" for item in discovery["results"])
    empty = retrieve(catalog, "", top_k=10, require_renderer=True)
    assert empty["results"] == []
    mug = retrieve(catalog, "mug", top_k=10, require_renderer=True)
    assert mug["results"][0]["asset_id"] == "prop_cup"
    ordered = sorted(
        [
            {"score": 0.2, "asset_id": "b", "version": "1.0.0", "package_sha256": "b" * 64},
            {"score": 0.2, "asset_id": "a", "version": "1.0.0", "package_sha256": "a" * 64},
        ],
        key=rank_key,
    )
    assert [item["asset_id"] for item in ordered] == ["a", "b"]


def test_interior_fit_and_negative_tasks(tmp_path: Path):
    interior = {"margin_m": 0.005, "size_m": [0.37, 0.27, 0.185]}
    assert fits_interior([0.18, 0.24, 0.03], interior)["fits"]
    assert not fits_interior([0.50, 0.50, 0.20], interior)["fits"]
    with pytest.raises(TaskError, match="robot_required"):
        generate(reachable_task(), Path("assets/candidates"), tmp_path / "tasks")
    task = book_inside_crate()
    task["workspace_m"] = {"minimum_m": [0.0, 0.0, 0.0], "maximum_m": [0.01, 0.01, 0.01]}
    result = generate(task, Path("assets/candidates"), tmp_path / "tasks")
    assert result["valid"] is False
    task = evaluation_tasks()[3]
    task["task_id"] = "forced_overlap"
    task["constraints"][-1]["constraints"]["minimum_m"] = 5.0
    separated = generate(task, Path("assets/candidates"), tmp_path / "tasks")
    assert separated["valid"] is False


def test_six_tasks_replay(tmp_path: Path):
    first = []
    for task in evaluation_tasks(17011):
        result = generate(task, Path("assets/candidates"), tmp_path / "one")
        assert result["valid"]
        again = generate(task, Path("assets/candidates"), tmp_path / "one")
        assert again["path"] == result["path"]
        replayed = replay(Path(result["path"]), Path("assets/candidates"))
        assert replayed["valid"]
        from sensor_sim.scene_gen.asset_package import canonical_json_bytes

        scene_bytes = (Path(result["path"]) / "scene.json").read_bytes()
        assert scene_bytes == canonical_json_bytes(replayed["scene"])
        first.append(scene_bytes)
    second = []
    for task in evaluation_tasks(17012):
        result = generate(task, Path("assets/candidates"), tmp_path / "two", seed=17012)
        assert result["valid"]
        second.append((Path(result["path"]) / "scene.json").read_bytes())
    assert first != second
