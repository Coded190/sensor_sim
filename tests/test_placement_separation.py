"""Objects must stay outside one another, including on a furnished support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sensor_sim.scene_gen.placement_separation import (
    MINIMUM_CLEARANCE_M,
    PlacementSeparationError,
    blockers_for,
    place_clear,
    require_outside,
    separation_report,
    support_frame,
)
from sensor_sim.scene_gen.scene_context import compose_scene_records
from sensor_sim.scene_gen.asset_validate import triangulate
from sensor_sim.scene_gen.structural import box_mesh


def _box(minimum, maximum, name):
    vertices, polygons = box_mesh(minimum, maximum)
    return {"instance_id": name, "name": name, "vertices": vertices, "faces": triangulate(polygons)}


def test_overlapping_boxes_are_inside():
    left = _box([0.0, 0.0, 0.0], [0.2, 0.2, 0.2], "left")
    right = _box([0.05, 0.05, 0.05], [0.25, 0.25, 0.25], "right")
    with pytest.raises(PlacementSeparationError, match="object_inside"):
        require_outside([left, right])


def test_separated_boxes_pass():
    left = _box([0.0, 0.0, 0.0], [0.1, 0.1, 0.1], "left")
    right = _box([0.2, 0.0, 0.0], [0.3, 0.1, 0.1], "right")
    report = require_outside([left, right])
    assert report["passed"] is True
    assert report["minimum_clearance_m"] == MINIMUM_CLEARANCE_M
    assert "triangle-to-triangle" in report["definition"]


def test_a_listed_support_pair_may_touch():
    table = _box([0.0, 0.0, 0.0], [1.0, 1.0, 0.05], "table")
    cup = _box([0.2, 0.2, 0.05], [0.28, 0.28, 0.15], "cup")
    report = require_outside([table, cup], allowed_pairs={frozenset(("table", "cup"))})
    assert report["passed"] is True


def test_fixed_center_on_the_dining_table_is_rejected_and_a_clear_cell_is_not():
    loaded = compose_scene_records("dining", ["sphere", "box", "apple"])
    frame = support_frame(loaded)
    blockers = blockers_for(loaded, frame)
    subjects = [
        _box([-0.04, -0.04, 0.0], [0.04, 0.04, 0.08], "sphere"),
        _box([-0.05, -0.05, 0.0], [0.05, 0.05, 0.1], "box"),
        _box([-0.04, -0.04, 0.0], [0.04, 0.04, 0.08], "apple"),
    ]
    crowded = []
    for subject, shift in zip(subjects, ([-0.16, 0.0], [0.0, 0.0], [0.16, 0.0]), strict=True):
        world = subject["vertices"] + np.array([frame["polygon"][:, 0].mean() + shift[0], frame["polygon"][:, 1].mean() + shift[1], frame["z"]])
        crowded.append({**subject, "vertices": world})
    center = separation_report(crowded, existing=blockers)
    assert center["passed"] is False
    placed = place_clear(
        subjects,
        blockers,
        frame["polygon"],
        offsets={"sphere": (-0.12, 0.0), "box": (0.0, 0.0), "apple": (0.12, 0.0)},
    )
    assert placed["report"]["passed"] is True
def test_real_subjects_fit_without_entering_existing_objects():
    from sensor_sim.scene_gen.asset_validate import triangulate
    from sensor_sim.scene_gen.mesh_io import read_obj

    paths = {
        "sphere": Path("assets/candidates/fixture_sphere/1.0.0/geometry/render.obj"),
        "box": Path("dataset/qualification/material_wrap_review/targets/box.obj"),
        "apple": Path("assets/candidates/prop_apple/1.0.0/geometry/render.obj"),
    }
    subjects = []
    for name, path in paths.items():
        vertices, polygons = read_obj(path)
        subjects.append({"name": name, "vertices": vertices, "faces": triangulate(polygons)})
    offsets = {"sphere": (-0.12, 0.0), "box": (0.0, 0.0), "apple": (0.12, 0.0)}
    for room_kind in ("bedroom", "dining", "kitchen", "office", "workshop"):
        loaded = compose_scene_records(room_kind, list(paths))
        frame = support_frame(loaded)
        placed = place_clear(subjects, blockers_for(loaded, frame), frame["polygon"], offsets=offsets)
        assert placed["report"]["passed"] is True
        assert placed["report"]["failures"] == []
        for subject, translation in zip(subjects, placed["translations"], strict=True):
            bottom = float(np.min(subject["vertices"][:, 2]))
            assert translation[2] == pytest.approx(frame["z"] - bottom, abs=1e-6)
