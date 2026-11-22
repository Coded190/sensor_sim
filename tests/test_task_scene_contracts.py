"""Validators for the task-scene contracts. Fixtures are not Infinigen assets."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from sensor_sim.scene_gen.asset_extension import AssetExtensionError, geometry_path, validate_extension
from sensor_sim.scene_gen.scene_manifest import (
    SceneManifestError,
    apply_transform,
    blender_world_to_canonical,
    canonical_to_mitsuba,
    canonical_to_usd,
    compose_transforms,
    exported_y_up_to_canonical,
    pinhole_camera_to_mitsuba,
    usd_to_canonical,
    validate_manifest,
    validate_rigid,
)
from sensor_sim.scene_gen.task_scene_request import TaskRequestError, domain_seed, validate_request


def _package() -> dict:
    return {"asset_id": "table_dining", "version": "1.0.0", "package_sha256": "a" * 64}


def _extension(**overrides) -> dict:
    document = {
        "schema_version": "asset_extension_v1",
        "asset_id": "table_dining",
        "package_version": "1.0.0",
        "package_sha256": "a" * 64,
        "frames": [
            {
                "name": "asset_root",
                "purpose": "mesh origin",
                "origin_m": [0, 0, 0],
                "axes": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]},
                "transform_to_asset": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "units": "m",
                "provenance": "sourced",
            }
        ],
        "regions": [],
        "geometries": [
            {"role": "render", "path": "geometry/render.obj", "sha256": "b" * 64, "frame": "asset_root", "units": "m", "method": "export", "validation": "checked"},
            {"role": "collision", "path": "geometry/collision.obj", "sha256": "c" * 64, "frame": "asset_root", "units": "m", "method": "convex_mesh", "validation": "checked"},
        ],
    }
    document.update(overrides)
    return document


def _request() -> dict:
    return {
        "schema_version": "task_scene_request_v1",
        "task_type": "pick_and_place",
        "support": {"asset_id": "table_dining", "count": {"minimum": 1, "maximum": 1}},
        "objects": [{"asset_id": "can", "count": {"minimum": 1, "maximum": 1}}],
        "target_region": {"frame": "support_surface", "origin_m": [0.2, 0.1, 0], "size_m": [0.15, 0.15, 0.05]},
        "camera_profile": {"name": "overview"},
        "light_profile": {"name": "relative_area"},
        "seeds": {"selection": 1, "layout": 2, "appearance": 3, "camera": 4, "lighting": 5},
        "relations": [{"type": "supported_on"}],
    }


def _manifest() -> dict:
    return {
        "schema_version": "scene_manifest_v1",
        "scene_id": "pick_0001",
        "task_id": "pick_and_place",
        "configuration_version": "1",
        "assets": [{"asset_id": "table_dining", "version": "1.0.0", "package_sha256": "a" * 64}],
        "instances": [
            {
                "instance_id": "table_0",
                "asset_id": "table_dining",
                "asset_version": "1.0.0",
                "package_sha256": "a" * 64,
                "package_path": "table_dining/1.0.0",
                "world_from_asset": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }
        ],
        "cameras": [
            {
                "camera_id": "camera_0",
                "width": 64,
                "height": 48,
                "projection": "pinhole",
                "fx": 50,
                "fy": 50,
                "cx": 32,
                "cy": 24,
                "world_from_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                "camera_frame": "x_right_y_down_z_forward",
            }
        ],
        "lights": [{"light_id": "light_0", "type": "constant", "radiometry": {"mode": "relative", "value": 1.0, "units": "relative"}, "calibration": "relative"}],
        "units": "m",
        "handedness": "right",
        "up_axis": "z",
        "seeds": {"selection": 1, "layout": 2, "appearance": 3, "camera": 4, "lighting": 5},
        "task_objects": {"pick": "can_0", "support": "table_0"},
    }


def test_extension_rejects_a_different_package_hash():
    document = _extension(package_sha256="d" * 64)
    with pytest.raises(AssetExtensionError, match="package_sha256_mismatch"):
        validate_extension(document, _package())


def test_extension_rejects_a_shared_render_and_collision_path():
    document = _extension()
    document["geometries"][1]["path"] = "geometry/render.obj"
    with pytest.raises(AssetExtensionError, match="collision_reuses_render_path"):
        validate_extension(document, _package())


def test_extension_accepts_a_matching_sidecar():
    validate_extension(_extension(), _package())


def test_geometry_path_rejects_escape(tmp_path: Path):
    with pytest.raises(Exception, match="path_traversal|path_escape"):
        geometry_path(tmp_path, "../outside.obj")


def test_transform_composition_uses_column_vectors():
    parent = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    child = np.array([[1, 0, 0, 3], [0, 1, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]], dtype=np.float64)
    world = compose_transforms(parent, child)
    assert np.allclose(apply_transform(world, np.array([0, 0, 0])), [5, 4, 5])


def test_basis_conversions_round_trip_known_points():
    point = np.array([1.0, 2.0, 3.0])
    canonical = np.eye(4)
    canonical[:3, 3] = point
    assert np.allclose(canonical_to_mitsuba(canonical), canonical)
    assert np.allclose(blender_world_to_canonical(canonical), canonical)
    usd = canonical_to_usd(canonical)
    assert np.allclose(apply_transform(usd, np.array([0, 0, 0])), [1, 3, -2])
    assert np.allclose(usd_to_canonical(usd)[:3, 3], point)
    exported = exported_y_up_to_canonical(np.eye(4))
    assert np.allclose(apply_transform(exported, np.array([1, 2, 3])), [1, -3, 2])
    reflected = np.diag([1.0, 1.0, -1.0, 1.0])
    with pytest.raises(SceneManifestError, match="transform_reflection"):
        validate_rigid(reflected)


def test_pinhole_forward_is_mitsuba_positive_z():
    from sensor_sim.scene_gen.scene_manifest import look_at_y_down

    origin = np.array([1.0, -1.3, 1.5])
    target = np.array([0.0, 0.0, 0.4])
    world = look_at_y_down(origin, target)
    mitsuba = pinhole_camera_to_mitsuba(world)
    desired = (target - origin) / np.linalg.norm(target - origin)
    assert np.allclose(mitsuba[:3, 2], desired)
    assert float(mitsuba[:3, 1] @ np.array([0.0, 0.0, 1.0])) > 0.0


def test_manifest_rejects_a_parent_path(tmp_path: Path):
    document = _manifest()
    document["instances"][0]["package_path"] = "../secret"
    with pytest.raises(SceneManifestError):
        validate_manifest(document, tmp_path)


def test_request_rejects_facing_and_keeps_seed_domains_independent():
    document = _request()
    document["relations"] = [{"type": "facing"}]
    with pytest.raises(TaskRequestError, match="unsupported_relation relation=facing"):
        validate_request(document, known_assets={"table_dining", "can"})
    validate_request(_request(), known_assets={"table_dining", "can"})
    assert domain_seed(9, "layout") != domain_seed(9, "lighting")
    assert domain_seed(9, "layout") == domain_seed(9, "layout")
    digest = hashlib.sha256(b"9:layout").digest()
    assert domain_seed(9, "layout") == int.from_bytes(digest[:8], "big") % (2**31 - 1)


def test_request_rejects_an_unknown_category_and_a_nonpositive_size():
    document = _request()
    document["objects"][0] = {"category": "spaceship", "count": {"minimum": 1, "maximum": 1}}
    with pytest.raises(TaskRequestError, match="unknown_category"):
        validate_request(document, known_categories={"table", "can"})
    document = _request()
    document["target_region"]["size_m"] = [0.1, 0.0, 0.1]
    with pytest.raises(TaskRequestError, match="target_size"):
        validate_request(document)
