"""Scene builder rejects a bare plane and accepts furnished rooms."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from sensor_sim.scene_gen.scene_context import (
    GEOMETRY_KIND,
    ROOM_DISTRIBUTION,
    SceneContextError,
    compose_scene_records,
    require_scene_context,
    sample_room_kinds,
)

WALLS = ("south", "north", "east", "west")


def _shell() -> list[dict]:
    rows = [
        {"asset_id": "room_kitchen_floor", "role": "context"},
        {"asset_id": "room_kitchen_ceiling", "role": "context"},
    ]
    rows.extend({"asset_id": f"room_kitchen_wall_{direction}", "role": "context"} for direction in WALLS)
    return rows


def test_bare_plane_is_rejected():
    instances = [
        {"asset_id": "ground", "kind": "plane", "role": "ground"},
        {"asset_id": "sphere", "role": "subject"},
        {"asset_id": "box", "role": "subject"},
        {"asset_id": "apple", "role": "subject"},
    ]
    with pytest.raises(SceneContextError, match="bare_plane"):
        require_scene_context(instances, room_kind="kitchen")


def test_room_without_other_objects_is_rejected():
    instances = _shell() + [{"asset_id": "sphere", "role": "subject"}]
    with pytest.raises(SceneContextError, match="missing_context_objects"):
        require_scene_context(instances, room_kind="kitchen")


def test_missing_wall_is_rejected():
    instances = [row for row in _shell() if "wall_east" not in row["asset_id"]]
    instances.append({"asset_id": "furn_table", "role": "furniture"})
    with pytest.raises(SceneContextError, match="missing_wall:east"):
        require_scene_context(instances, room_kind="kitchen")


def test_exemption_requires_a_reason():
    with pytest.raises(SceneContextError, match="exemption_unspecified"):
        require_scene_context(
            [{"asset_id": "ground", "kind": "plane"}],
            room_kind="probe",
            allow_bare_plane=True,
            allow_without_context_objects=True,
        )


def test_specified_probe_can_be_a_bare_plane():
    report = require_scene_context(
        [{"asset_id": "ground", "kind": "plane"}, {"asset_id": "sphere", "role": "subject"}],
        room_kind="probe",
        allow_bare_plane=True,
        allow_without_context_objects=True,
        exemption="controlled_material_probe",
    )
    assert report["passed"] is True
    assert report["bare_planes"] == 1
    assert report["exemption"] == "controlled_material_probe"
    assert "context_objects excludes the shell" in report["definition"]


def test_furnished_shell_with_one_other_object_passes():
    report = require_scene_context(_shell() + [{"asset_id": "furn_table", "role": "furniture"}], room_kind="kitchen")
    assert report["passed"] is True
    assert report["context_objects"] == 1
    assert report["walls"] == list(WALLS)
    assert report["min_context_objects"] == 1


def test_room_kind_sample_is_deterministic_and_inside_the_distribution():
    allowed = [name for name, _weight in ROOM_DISTRIBUTION]
    assert {"room", "kitchen", "living", "bedroom", "lab"} <= set(allowed)
    first = sample_room_kinds(random.Random(17521), 4)
    second = sample_room_kinds(random.Random(17521), 4)
    assert first == second
    assert len(first) == len(set(first)) == 4
    assert set(first) <= set(allowed)


def test_saved_furnished_rooms_pass_with_subjects():
    for room_kind in GEOMETRY_KIND:
        loaded = compose_scene_records(room_kind, ["sphere", "box", "apple"])
        report = loaded["report"]
        assert report["passed"] is True
        assert report["context_objects"] >= 1
        assert report["subjects"] == 3
        assert report["bare_planes"] == 0
        assert loaded["support_top_m"][2] > 0.15
        if room_kind == "lab":
            assert loaded["geometry_kind"] == "workshop"
            assert loaded["lab_uses_workshop_shell"] is True


def test_saved_scene_builders_already_have_a_room_and_other_objects():
    """Interior, semantic, room-diversity, and tabletop bundles use the same check."""
    root = Path("dataset/qualification")
    interior = json.loads(next((root / "diverse_interior_review" / "kitchen").glob("*/bundle.json")).read_text(encoding="utf-8"))
    semantic_path = next((root / "semantic_scenes").glob("*/*/bundle.json"))
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    room = json.loads(next((root / "room_diversity").glob("*/*/bundle.json")).read_text(encoding="utf-8"))
    tabletop = json.loads(next((root / "tabletop_scenes").glob("*/*/bundle.json")).read_text(encoding="utf-8"))
    reports = [
        require_scene_context(interior["instances"], room_kind="kitchen"),
        require_scene_context(semantic["room_components"] + semantic["furniture_instances"] + semantic["object_instances"], room_kind=semantic["recipe"]["scene_type"]),
        require_scene_context(room["instances"], room_kind=room["profile_id"]),
        require_scene_context(tabletop["resolved"]["assets"], room_kind="room"),
    ]
    for report in reports:
        assert report["passed"] is True
        assert report["bare_planes"] == 0
        assert report["context_objects"] >= report["min_context_objects"]
        assert report["walls"] == ["south", "north", "east", "west"]
