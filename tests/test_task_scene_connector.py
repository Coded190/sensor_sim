"""Shared scene resolution, camera projection, and pose conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sensor_sim.scene_gen.asset_library import build_catalog, load_catalog, write_catalog
from sensor_sim.scene_gen.scene_manifest import load_scene_manifest, look_at_y_down, matrix_lists, validate_rigid
from sensor_sim.scene_gen.task_scene_connector import (
    ConnectorError,
    canonical_to_isaac_sim,
    isaac_camera_is_representable,
    matrix_to_quat_wxyz,
    pose_wxyz,
    project_mitsuba,
    project_pinhole,
    quat_wxyz_to_matrix,
    resolve_scene,
    resolve_scene_files,
    scene_resolution_hash,
)

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "dataset" / "task_scene" / "library"
REPORTS = ROOT / "dataset" / "task_scene" / "reports"
MANIFEST = ROOT / "dataset" / "task_scene" / "scenes" / "pick_can" / "scene_manifest.json"


def _catalog(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.json"
    write_catalog(path, build_catalog(LIBRARY, reports_root=REPORTS))
    return path


def test_pick_can_resolves_both_geometries(tmp_path: Path):
    catalog_path = _catalog(tmp_path)
    resolved = resolve_scene_files(MANIFEST, catalog_path, LIBRARY)
    ids = {item["instance_id"]: item for item in resolved["instances"]}
    assert set(ids) == {"table_0", "can_0"}
    assert ids["table_0"]["render_path"].endswith("render.obj")
    assert ids["can_0"]["collision_path"].endswith("collision.obj")
    assert ids["can_0"]["render_geometry_hash"] != ids["can_0"]["collision_geometry_hash"]
    can = np.asarray(ids["can_0"]["world_from_asset"], dtype=np.float64)
    manifest = load_scene_manifest(MANIFEST.read_text(encoding="utf-8"))
    expected = np.asarray(manifest["instances"][1]["world_from_asset"], dtype=np.float64)
    assert np.allclose(can, expected, atol=1e-12)
    assert resolved["manifest_sha256"] != resolved["scene_resolution_sha256"]
    again = scene_resolution_hash(resolved)
    assert again == resolved["scene_resolution_sha256"]


def test_projection_matches_across_the_mitsuba_roll():
    origin = np.array([1.0, -1.3, 1.5])
    pose = look_at_y_down(origin, np.array([0.0, 0.0, 0.4]))
    camera = {"fx": 120.0, "fy": 120.0, "cx": 80.0, "cy": 60.0, "width": 160, "height": 120, "projection": "pinhole"}
    forward = pose[:3, 2]
    point = origin + forward * 1.0
    u, v = project_pinhole(pose, point, camera)
    um, vm = project_mitsuba(pose, point, camera)
    assert u == pytest.approx(um, abs=1e-6)
    assert v == pytest.approx(vm, abs=1e-6)
    assert u == pytest.approx(80.0, abs=1e-6)
    assert v == pytest.approx(60.0, abs=1e-6)
    right = origin + pose[:3, 0] * 0.1 + forward
    down = origin + pose[:3, 1] * 0.1 + forward
    right_pixel = project_pinhole(pose, right, camera)
    down_pixel = project_pinhole(pose, down, camera)
    assert right_pixel[0] > u
    assert down_pixel[1] > v


def test_quaternion_round_trip_and_isaac_stage_is_not_the_y_up_helper():
    pose = look_at_y_down(np.array([0.2, -0.4, 1.1]), np.array([0.0, 0.0, 0.4]))
    position, quaternion = pose_wxyz(pose)
    rebuilt = quat_wxyz_to_matrix(quaternion, position)
    assert np.allclose(rebuilt, validate_rigid(pose), atol=1e-8)
    identity = np.eye(4)
    assert matrix_to_quat_wxyz(identity) == pytest.approx([1.0, 0.0, 0.0, 0.0])
    z_point = canonical_to_isaac_sim(identity) @ np.array([0.0, 0.0, 1.0, 1.0])
    assert np.allclose(z_point[:3], [0.0, 0.0, 1.0])


def test_noncentered_camera_fails_closed(tmp_path: Path):
    catalog = load_catalog(_catalog(tmp_path))
    manifest = load_scene_manifest(MANIFEST.read_text(encoding="utf-8"))
    manifest["cameras"][0]["cx"] = 10.0
    assert isaac_camera_is_representable(manifest["cameras"][0]) is False
    with pytest.raises(ConnectorError, match="isaac_camera_intrinsics_not_representable"):
        resolve_scene(
            manifest,
            catalog,
            LIBRARY,
            manifest_sha256="ab" * 32,
            catalog_sha256="cd" * 32,
        )


def test_pose_change_changes_the_resolution_hash(tmp_path: Path):
    catalog = load_catalog(_catalog(tmp_path))
    manifest = load_scene_manifest(MANIFEST.read_text(encoding="utf-8"))
    first = resolve_scene(manifest, catalog, LIBRARY, manifest_sha256="aa" * 32, catalog_sha256="bb" * 32)
    moved = json_copy(manifest)
    moved["instances"][1]["world_from_asset"] = matrix_lists(
        np.asarray(moved["instances"][1]["world_from_asset"], dtype=np.float64) + np.array([[0, 0, 0, 0.01], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    )
    second = resolve_scene(moved, catalog, LIBRARY, manifest_sha256="aa" * 32, catalog_sha256="bb" * 32)
    assert first["scene_resolution_sha256"] != second["scene_resolution_sha256"]


def json_copy(value):
    import json

    return json.loads(json.dumps(value))
