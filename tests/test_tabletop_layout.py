"""Placement and room-contact checks. These tests read the candidate catalog and do not rewrite it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.indoor_transform import deserialize_matrix
from sensor_sim.scene_gen.tabletop_layout import load_mesh, place_tabletop
from sensor_sim.scene_gen.tabletop_scene import RECIPES, _requests, _room

CATALOG = Path("assets/candidates")


def _controlled(seed: int, separation: float = 0.03):
    room = _room(CATALOG, "south_open")
    assert room["valid"]
    recipe = RECIPES["tabletop_polarization_controlled"]
    return place_tabletop(
        CATALOG,
        room["table_matrix"],
        room["table_support_local"],
        _requests(recipe, room["clusters"]),
        seed=seed,
        minimum_separation_m=separation,
    )


def test_room_furniture_is_supported_and_separated():
    for name in ("south_open", "east_work", "west_shelf"):
        room = _room(CATALOG, name)
        assert room["valid"], room["failure"]
        assert room["intersections"] == []
        assert all(item["passed"] for item in room["contacts"])
        assert abs(float(room["table_support_world"][:, 2].mean()) - 0.79) < 1e-6


def test_same_seed_is_identical_and_a_different_seed_moves_objects():
    first = _controlled(17101)
    second = _controlled(17101)
    third = _controlled(17102)
    assert first["valid"] and third["valid"]
    assert first["poses"] == second["poses"]
    assert first["poses"] != third["poses"]
    assert len({row["instance_id"] for row in first["placed"]}) == 8


def test_objects_stay_on_the_inset_table():
    placed = _controlled(17101)
    support = _room(CATALOG, "south_open")["table_support_world"]
    inset = (
        float(support[:, 0].min()) + 0.015,
        float(support[:, 1].min()) + 0.015,
        float(support[:, 0].max()) - 0.015,
        float(support[:, 1].max()) - 0.015,
    )
    for row in placed["placed"]:
        points = row["world"][:, :2]
        assert np.all(points[:, 0] >= inset[0] - 1e-6)
        assert np.all(points[:, 0] <= inset[2] + 1e-6)
        assert np.all(points[:, 1] >= inset[1] - 1e-6)
        assert np.all(points[:, 1] <= inset[3] + 1e-6)
        assert abs(float(row["world"][:, 2].min()) - 0.79) < 1e-6
        matrix = deserialize_matrix(row["matrix"])
        assert np.allclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-8)


def test_clearance_respects_the_configured_separation():
    placed = _controlled(17101, separation=0.03)
    clearances = [row["nearest_clearance_m"] for row in placed["placed"] if row["nearest_clearance_m"] is not None]
    assert clearances
    assert min(clearances) + 1e-9 >= 0.03


def test_impossible_separation_is_a_recorded_failure():
    failed = _controlled(17101, separation=2.0)
    assert failed["valid"] is False
    assert failed["collision_log"]["termination"] == "placement_failed"
    assert failed["collision_log"]["failed_instance"]


def test_repeated_ids_are_unique_on_the_dense_recipe():
    room = _room(CATALOG, "west_shelf")
    placed = place_tabletop(
        CATALOG,
        room["table_matrix"],
        room["table_support_local"],
        _requests(RECIPES["tabletop_dense_clutter"], room["clusters"]),
        seed=17101,
        minimum_separation_m=0.008,
    )
    assert placed["valid"]
    ids = [row["instance_id"] for row in placed["placed"]]
    assert len(ids) == 16
    assert len(set(ids)) == 16
    assert "prop_can__1" in ids and "prop_can__2" in ids
    can = load_mesh(CATALOG, "prop_can")
    assert len(can["vertices"]) > 0
