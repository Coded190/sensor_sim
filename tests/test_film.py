"""Geometry, analyzer identities, split groups, and the U-Net channel counts."""

from __future__ import annotations

import numpy as np
import pytest

from sensor_sim.film.dataset import assert_split_groups, iter_views
from sensor_sim.film.config import dataset_config
from sensor_sim.film.geometry import (
    CORNELL_HIGH,
    CORNELL_LOW,
    ball_center_offset,
    cap_shell,
    cornell_shell,
    platform_mesh,
    room_box,
    solid_box,
    theta_max,
    uv_sphere,
    write_obj,
)
from sensor_sim.film.lighting import DISPLAY_EXPOSURE, area_power_watts, display_srgb
from sensor_sim.film.images import read_gray_png, write_gray_png
from sensor_sim.film.metrics import scores
from sensor_sim.film.polar import analyzer_images, reconstruction_error


def _face_normal(vertices, face):
    a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    return np.cross(b - a, c - a)


def test_cap_coverage_angle_and_normals():
    coverage = 0.45
    polar = theta_max(coverage)
    assert (1.0 - np.cos(polar)) / 2.0 == pytest.approx(coverage)
    mesh = cap_shell(0.05, 2e-5, coverage, (0.0, -1.0, 0.0), n_rings=8, n_phi=12)
    vertices = mesh["vertices"]
    for group, sign in (("outer", 1.0), ("inner", -1.0)):
        for face_index in mesh["groups"][group]:
            face = mesh["faces"][face_index]
            centroid = vertices[list(face)].mean(axis=0)
            assert np.dot(_face_normal(vertices, face), centroid) * sign > 0.0
    thickness = 2e-5
    radii = np.linalg.norm(vertices, axis=1)
    assert radii.max() == pytest.approx(0.05, rel=1e-6)
    assert radii.min() == pytest.approx(0.05 - thickness, rel=1e-6)


def test_room_normals_point_inward():
    mesh = room_box(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    center = np.zeros(3)
    for face in mesh["faces"]:
        centroid = mesh["vertices"][list(face)].mean(axis=0)
        normal = _face_normal(mesh["vertices"], face)
        assert np.dot(normal, center - centroid) > 0.0


def test_cornell_shell_is_open_at_the_front_and_inward():
    walls = cornell_shell()
    assert set(walls) == {"floor", "ceiling", "wall-back", "wall-left", "wall-right"}
    center = 0.5 * (CORNELL_LOW + CORNELL_HIGH)
    for mesh in walls.values():
        for face in mesh["faces"]:
            centroid = mesh["vertices"][list(face)].mean(axis=0)
            assert np.dot(_face_normal(mesh["vertices"], face), center - centroid) > 0.0
    back = walls["wall-back"]["vertices"][walls["wall-back"]["faces"]].mean(axis=1)
    assert np.allclose(back[:, 1], CORNELL_HIGH[1])
    left = walls["wall-left"]["vertices"][walls["wall-left"]["faces"]].mean(axis=1)
    assert np.allclose(left[:, 0], CORNELL_LOW[0])


def test_platform_normals_point_outward_and_ball_sits_on_it():
    mesh = platform_mesh(0.0)
    center = mesh["vertices"].mean(axis=0)
    for face in mesh["faces"]:
        centroid = mesh["vertices"][list(face)].mean(axis=0)
        assert np.dot(_face_normal(mesh["vertices"], face), centroid - center) > 0.0
    offset = ball_center_offset(0.05, 1.0, 0.001, floor_z=0.0)
    assert offset[2] == pytest.approx(0.04 + 0.001 + 0.05)
    solid = solid_box(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    assert solid["faces"].shape == (12, 3)


def test_flat_box_normals_stay_on_the_face(tmp_path):
    mesh = platform_mesh(0.0)
    path = tmp_path / "platform.obj"
    write_obj(path, mesh["vertices"], mesh["faces"], smooth=False)
    normals = np.array(
        [[float(value) for value in line.split()[1:]] for line in path.read_text().splitlines() if line.startswith("vn ")],
        dtype=np.float64,
    )
    assert normals.shape == (36, 3)
    assert np.allclose(np.max(np.abs(normals), axis=1), 1.0)
    assert any(np.allclose(normal, [0.0, -1.0, 0.0]) for normal in normals)


def test_obj_writer_stores_a_normal_per_vertex(tmp_path):
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    path = tmp_path / "tri.obj"
    write_obj(path, vertices, faces)
    text = path.read_text()
    assert "vn 0.0000000000 0.0000000000 1.0000000000" in text
    assert "f 1//1 2//2 3//3" in text


def test_uv_sphere_is_closed_and_outward():
    mesh = uv_sphere(0.05, n_rings=8, n_phi=12)
    assert mesh["faces"].shape[0] > 0
    radii = np.linalg.norm(mesh["vertices"], axis=1)
    assert radii == pytest.approx(0.05)
    for face in mesh["faces"]:
        centroid = mesh["vertices"][list(face)].mean(axis=0)
        assert np.dot(_face_normal(mesh["vertices"], face), centroid) > 0.0


def test_area_power_is_pi_times_area_times_radiance():
    # Cycles 4.5, Normalize on, exposure 0: radiance = energy / (pi * width * height).
    width, height, radiance = 0.4, 0.4, 10.0
    power = area_power_watts(width, height, radiance)
    assert power == pytest.approx(np.pi * width * height * radiance)
    assert power / (np.pi * width * height) == pytest.approx(radiance)
    with pytest.raises(ValueError):
        area_power_watts(0.0, 0.4, 10.0)


def test_display_shoulder_lifts_the_dark_body_and_stays_below_one():
    # Filmed-body linear radiance is near 0.04. A hard clip leaves that near black.
    body = display_srgb(np.array([0.041]))
    hard_clip = float(np.clip(0.041, 0.0, 1.0) ** (1.0 / 2.2))
    assert float(body[0]) > hard_clip
    assert float(display_srgb(np.array([10.0]))[0]) < 1.0
    mapped = (DISPLAY_EXPOSURE * 0.041) / (1.0 + DISPLAY_EXPOSURE * 0.041)
    assert float(body[0]) == pytest.approx(mapped ** (1.0 / 2.2))


def test_analyzer_identities_reconstruct_stokes():
    s0 = np.array([[1.0, 0.8], [0.4, 0.2]])
    s1 = np.array([[0.2, -0.1], [0.0, 0.05]])
    s2 = np.array([[-0.1, 0.2], [0.1, -0.05]])
    images = analyzer_images(s0, s1, s2)
    assert images["analyzer_000"] == pytest.approx(0.5 * (s0 + s1))
    assert images["analyzer_045"] == pytest.approx(0.5 * (s0 + s2))
    assert images["analyzer_090"] == pytest.approx(0.5 * (s0 - s1))
    assert images["analyzer_135"] == pytest.approx(0.5 * (s0 - s2))
    error = reconstruction_error(s0, s1, s2, images)
    assert error["max"] == pytest.approx(0.0, abs=1e-12)
    assert images["analyzer_000"].min() >= -1e-12


def test_scores_empty_and_perfect():
    label = np.array([[1, 0], [1, 1]], dtype=float)
    perfect = scores(label, label, threshold=0.5)
    assert perfect["iou"] == pytest.approx(1.0)
    assert perfect["precision"] == pytest.approx(1.0)
    assert perfect["recall"] == pytest.approx(1.0)
    empty = np.zeros((2, 2))
    both_empty = scores(empty, empty, threshold=0.5)
    assert both_empty["iou"] == pytest.approx(1.0)
    assert both_empty["precision"] == pytest.approx(1.0)
    assert scores(label, empty)["recall"] == pytest.approx(0.0)
    assert scores(empty, label)["precision"] == pytest.approx(0.0)


def test_mask_png_keeps_literal_zero_and_one(tmp_path):
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    path = tmp_path / "film_mask.png"
    write_gray_png(path, mask)
    loaded = read_gray_png(path)
    assert loaded.dtype == np.uint8
    assert loaded.tolist() == mask.tolist()


def test_dataset_splits_keep_scene_ids_together():
    samples = iter_views(dataset_config(), base_intensity=1.0)
    assert_split_groups(samples)
    by_id = {}
    for sample in samples:
        by_id.setdefault(sample["scene_id"], set()).add(sample["split"])
    assert by_id["sphere_cov45"] == {"train"}
    assert len([sample for sample in samples if sample["scene_id"] == "sphere_cov45"]) == 3
    test_coverages = {sample["coverage"] for sample in samples if sample["split"] == "test" and sample["geometry"] == "sphere"}
    train_coverages = {sample["coverage"] for sample in samples if sample["split"] == "train"}
    assert test_coverages.isdisjoint(train_coverages)
    assert any(sample["geometry"] == "ellipsoid" and sample["split"] == "test" for sample in samples)


def test_unet_channel_count_and_output_shape():
    torch = pytest.importorskip("torch")
    from sensor_sim.film.unet import FilmUNet, parameter_count

    torch.manual_seed(0)
    model_a = FilmUNet(3)
    torch.manual_seed(0)
    model_b = FilmUNet(7)
    assert parameter_count(model_b) - parameter_count(model_a) == 1152
    logits = model_b(torch.zeros(1, 7, 32, 32))
    assert logits.shape == (1, 1, 32, 32)
    with pytest.raises(ValueError):
        model_a(torch.zeros(1, 3, 30, 32))
