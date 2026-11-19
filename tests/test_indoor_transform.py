"""Measured Blender-compatible transforms against installed Mitsuba 3.9.1."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sensor_sim.scene_gen.asset_package import PackageError
from sensor_sim.scene_gen.indoor_transform import (
    apply_normals,
    apply_points,
    compose,
    inspect_linear,
    load_mitsuba_camera,
    load_mitsuba_mesh,
    rotation_matrix,
    scale_matrix,
    translation_matrix,
)
from sensor_sim.scene_gen.mesh_io import write_obj

BLENDER = Path("tools/blender-4.5.9-linux-x64/blender")
TRIANGLE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float64)
REST_NORMAL = np.cross(TRIANGLE[1] - TRIANGLE[0], TRIANGLE[2] - TRIANGLE[0])
REST_NORMAL = REST_NORMAL / np.linalg.norm(REST_NORMAL)


def _write(tmp_path: Path) -> Path:
    path = tmp_path / "triangle.obj"
    path.write_bytes(write_obj(TRIANGLE, [[0, 1, 2]], name="triangle"))
    return path


def _matches(obj_path: Path, matrix: np.ndarray) -> None:
    loaded = load_mitsuba_mesh(obj_path, matrix)
    expected = apply_points(matrix, TRIANGLE)
    expected_normal = apply_normals(matrix, REST_NORMAL.reshape(1, 3))[0]
    assert loaded["variant"] == "cuda_ad_spectral_polarized"
    assert np.allclose(loaded["positions"], expected, atol=1e-5)
    assert np.allclose(loaded["normals"][0], expected_normal, atol=1e-4)


def test_mitsuba_round_trip_cases(tmp_path: Path):
    obj_path = _write(tmp_path)
    _matches(obj_path, translation_matrix([1.0, -2.0, 0.5]))
    for axis in ("x", "y", "z"):
        _matches(obj_path, rotation_matrix(axis, 0.7))
    _matches(obj_path, scale_matrix([2.0, 2.0, 2.0]))
    _matches(obj_path, scale_matrix([2.0, 3.0, 4.0]))
    mirrored = scale_matrix([-1.0, 1.0, 1.0])
    report = inspect_linear(mirrored)
    assert report["mirrored"] is True
    _matches(obj_path, mirrored)
    parent = scale_matrix([2.0, 1.0, 1.0])
    local = rotation_matrix("z", np.pi / 2.0)
    orthogonal = compose(parent, local)
    assert inspect_linear(orthogonal)["shear"] < 1e-6
    _matches(obj_path, orthogonal)
    with pytest.raises(PackageError, match="shear"):
        inspect_linear(compose(parent, rotation_matrix("z", np.pi / 4.0)))
    with pytest.raises(PackageError, match="singular_transform"):
        inspect_linear(scale_matrix([0.0, 1.0, 1.0]))


def test_support_normal_and_camera(tmp_path: Path):
    floor = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    path = tmp_path / "floor.obj"
    path.write_bytes(write_obj(floor, [[0, 1, 2]], name="floor"))
    matrix = translation_matrix([0.2, 0.3, 0.4])
    loaded = load_mitsuba_mesh(path, matrix)
    assert np.allclose(loaded["normals"][0], [0.0, 0.0, 1.0], atol=1e-5)
    assert np.allclose(loaded["positions"][0], [0.2, 0.3, 0.4], atol=1e-5)
    camera = load_mitsuba_camera([0.0, -2.0, 1.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
    assert np.allclose(camera["origin"], [0.0, -2.0, 1.0], atol=1e-5)
    forward = np.array([0.0, 2.0, -0.5])
    forward = forward / np.linalg.norm(forward)
    assert np.allclose(camera["forward"], forward, atol=1e-5)
    assert np.allclose(camera["center_direction"], forward, atol=1e-4)
    assert abs(float(np.dot(camera["camera_x"], camera["forward"]))) < 1e-5
    assert float(np.dot(camera["camera_y"], [0.0, 0.0, 1.0])) > 0.0


def test_blender_matrix_matches_mitsuba(tmp_path: Path):
    if not BLENDER.is_file():
        pytest.skip("pinned Blender binary is not present")
    report_path = tmp_path / "blender_transform.json"
    completed = subprocess.run(
        [
            str(BLENDER),
            "--background",
            "--python",
            "assets/blender/create_indoor_scene.py",
            "--",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    obj_path = tmp_path / "blender_triangle.obj"
    obj_path.write_bytes(write_obj(np.array(report["local_vertices"]), [[0, 1, 2]], name="triangle"))
    for case in report["cases"]:
        matrix = np.asarray(case["matrix"], dtype=np.float64)
        if case["expect"] == "reject_shear":
            with pytest.raises(PackageError, match="shear"):
                inspect_linear(matrix)
            continue
        loaded = load_mitsuba_mesh(obj_path, matrix)
        assert np.allclose(loaded["positions"], np.asarray(case["world_vertices"]), atol=1e-5)
        assert np.allclose(apply_points(matrix, report["local_vertices"]), np.asarray(case["world_vertices"]), atol=1e-6)
