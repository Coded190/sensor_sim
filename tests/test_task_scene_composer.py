"""Deterministic resolver tests on fixture boxes, not on Infinigen output."""

import numpy as np
import pytest

from sensor_sim.scene_gen.task_scene_composer import ComposerError, compose_scene
from sensor_sim.scene_gen.task_scene_request import TaskRequestError, validate_request


def _box_vertices() -> np.ndarray:
    return np.array([[-0.01, -0.01, 0.0], [0.01, -0.01, 0.0], [0.01, 0.01, 0.0], [-0.01, 0.01, 0.0], [-0.01, -0.01, 0.04], [0.01, -0.01, 0.04], [0.01, 0.01, 0.04], [-0.01, 0.01, 0.04]], dtype=np.float64)


def _faces() -> np.ndarray:
    return np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1]], dtype=np.int64)


def _asset(asset_id: str, category: str, frame_name: str) -> dict:
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    polygon = [[-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [0.4, 0.4, 0.0], [-0.4, 0.4, 0.0]]
    return {
        "category": category,
        "package": {"asset_id": asset_id, "version": "1.0.0", "package_sha256": "ab" * 32},
        "extension": {
            "schema_version": "asset_extension_v1",
            "asset_id": asset_id,
            "package_version": "1.0.0",
            "package_sha256": "ab" * 32,
            "frames": [
                {"name": "asset_root", "purpose": "origin", "origin_m": [0, 0, 0], "axes": {}, "transform_to_asset": identity, "units": "m", "provenance": "sourced"},
                {"name": frame_name, "purpose": frame_name, "origin_m": [0, 0, 0], "axes": {}, "transform_to_asset": identity, "units": "m", "provenance": "estimated"},
            ],
            "regions": [],
            "geometries": [
                {"role": "render", "path": "geometry/render.obj", "sha256": "cd" * 32, "frame": "asset_root", "units": "m", "method": "fixture", "validation": "fixture"},
                {"role": "collision", "path": "geometry/collision.obj", "sha256": "ef" * 32, "frame": "asset_root", "units": "m", "method": "convex_mesh", "validation": "fixture"},
            ],
        },
        "render_vertices": _box_vertices() if category == "can" else np.array(polygon, dtype=np.float64),
        "render_faces": _faces() if category == "can" else np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        "support_polygon_local": np.array(polygon, dtype=np.float64),
    }


def _request(selection: int = 4) -> dict:
    return {
        "schema_version": "task_scene_request_v1",
        "task_type": "pick_and_place",
        "support": {"asset_id": "table_dining", "count": {"minimum": 1, "maximum": 1}},
        "objects": [{"category": "can", "count": {"minimum": 1, "maximum": 1}}],
        "target_region": {"frame": "support_surface", "origin_m": [0.0, 0.0, 0.0], "size_m": [0.1, 0.1, 0.02]},
        "camera_profile": {"name": "overview"},
        "light_profile": {"name": "relative"},
        "seeds": {"selection": selection, "layout": 8, "appearance": 3, "camera": 5, "lighting": 6},
        "relations": [{"type": "supported_on"}],
        "clutter": 0.0,
    }


def _library() -> list[dict]:
    return [_asset("table_dining", "table", "support_surface"), _asset("can_a", "can", "base_contact"), _asset("can_b", "can", "base_contact")]


def test_same_request_has_the_same_manifest_hash():
    first = compose_scene(_request(), _library())
    second = compose_scene(_request(), _library())
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert {row["status"] for row in first["validation"] if row["check_id"] == "reachable_from"} == {"not_evaluated"}


def test_seed_domains_change_only_their_fields():
    base = compose_scene(_request(), _library())["manifest"]
    layout = dict(_request())
    layout["seeds"] = dict(layout["seeds"], layout=99)
    moved = compose_scene(layout, _library())["manifest"]
    assert moved["instances"][1]["world_from_asset"] != base["instances"][1]["world_from_asset"]
    assert moved["instances"][1]["asset_id"] == base["instances"][1]["asset_id"]
    assert moved["cameras"] == base["cameras"]
    assert moved["lights"] == base["lights"]
    assert moved["instances"][1]["material_profile"] == base["instances"][1]["material_profile"]

    appearance = dict(_request())
    appearance["seeds"] = dict(appearance["seeds"], appearance=100)
    styled = compose_scene(appearance, _library())["manifest"]
    assert styled["instances"][1]["material_profile"] != base["instances"][1]["material_profile"]
    assert styled["instances"][1]["world_from_asset"] == base["instances"][1]["world_from_asset"]
    assert styled["cameras"] == base["cameras"]

    camera = dict(_request())
    camera["seeds"] = dict(camera["seeds"], camera=101)
    viewed = compose_scene(camera, _library())["manifest"]
    assert viewed["cameras"] != base["cameras"]
    assert viewed["instances"][1]["world_from_asset"] == base["instances"][1]["world_from_asset"]
    assert viewed["lights"] == base["lights"]

    lighting = dict(_request())
    lighting["seeds"] = dict(lighting["seeds"], lighting=102)
    lit = compose_scene(lighting, _library())["manifest"]
    assert lit["lights"] != base["lights"]
    assert lit["instances"] == base["instances"]
    assert lit["cameras"] == base["cameras"]


def test_selection_seed_changes_only_the_asset_id():
    first = compose_scene(_request(4), _library())["manifest"]
    second_request = _request(4)
    second_request["seeds"] = dict(second_request["seeds"], selection=5)
    second = compose_scene(second_request, _library())["manifest"]
    assert first["instances"][1]["asset_id"] != second["instances"][1]["asset_id"]
    assert first["instances"][1]["world_from_asset"] == second["instances"][1]["world_from_asset"]
    assert first["cameras"] == second["cameras"]
    assert len({item["instance_id"] for item in first["instances"]}) == 2


def test_facing_relation_fails_closed():
    request = _request()
    request["relations"] = [{"type": "facing"}]
    with pytest.raises(TaskRequestError, match="unsupported_relation"):
        validate_request(request, known_categories={"table", "can"}, known_assets={"table_dining", "can_a", "can_b"})


def test_bad_support_is_a_recorded_failure(monkeypatch):
    assets = _library()
    assets[1]["render_vertices"] = assets[1]["render_vertices"] + np.array([0.0, 0.0, 0.05])
    assets[2]["render_vertices"] = assets[2]["render_vertices"] + np.array([0.0, 0.0, 0.05])
    with pytest.raises(ComposerError, match="support_gap"):
        compose_scene(_request(), assets)
