"""IoU, DoLP features, and a synthetic film probe that does not peek at test frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sensor_sim.phase4 import (
    TEST_IDS,
    TRAIN_IDS,
    Frame,
    choose_color_threshold,
    feature_columns,
    frame_iou,
    load_frame,
    run_probe,
)
from sensor_sim.stokes import polarization_maps


def test_iou_identity_and_empty_cases():
    mask = np.array([[1, 0], [1, 1]], dtype=bool)
    assert frame_iou(mask, mask) == pytest.approx(1.0)
    empty = np.zeros((2, 2), dtype=bool)
    assert frame_iou(empty, empty) == pytest.approx(1.0)
    assert frame_iou(mask, empty) == pytest.approx(0.0)
    assert frame_iou(empty, mask) == pytest.approx(0.0)


def test_dolp_on_a_known_pixel_matches_polarization_maps():
    s0 = np.full((1, 1, 3), 1.0)
    s1 = np.full((1, 1, 3), 0.6)
    s2 = np.full((1, 1, 3), 0.8)
    s3 = np.zeros((1, 1, 3))
    expected = polarization_maps(s0, s1, s2, s3)["dolp"][0, 0]
    assert expected == pytest.approx(np.sqrt(0.6**2 + 0.8**2))
    dark = np.zeros((1, 1, 3))
    assert polarization_maps(dark, s1, s2, s3)["dolp"][0, 0] == pytest.approx(0.0)

    frame = load_frame(Path("outputs/phase3"), 0)
    import mitsuba as mi

    mi.set_variant("scalar_rgb")
    stokes = dict(mi.Bitmap("outputs/phase3/frames/000/stokes/render.exr").split())
    again = polarization_maps(
        np.array(stokes["S0"]),
        np.array(stokes["S1"]),
        np.array(stokes["S2"]),
        np.array(stokes["S3"]),
    )["dolp"]
    assert frame.dolp == pytest.approx(again)


def test_feature_stack_widths_and_order():
    columns = feature_columns()
    assert [len(columns[name]) for name in ("rgb", "rgb_dolp", "rgb_dolp_cube")] == [3, 4, 12]
    assert columns["rgb_dolp"][:4] == ("R", "G", "B", "DoLP")
    assert columns["rgb_dolp_cube"][4] == "band_01_400"
    assert columns["rgb_dolp_cube"][-1] == "band_08_662"

    frame = load_frame(Path("outputs/phase3"), 0)
    from sensor_sim.phase4 import stack_features

    stacked = stack_features(frame)
    assert stacked["rgb"].shape[-1] == 3
    assert stacked["rgb_dolp"].shape[-1] == 4
    assert stacked["rgb_dolp_cube"].shape[-1] == 12
    assert stacked["rgb_dolp"][:, :, 3] == pytest.approx(frame.dolp)
    assert stacked["rgb_dolp_cube"][:, :, 4:] == pytest.approx(frame.cube)
    assert frame.mask.dtype == bool
    assert frame.mask.any()


def test_color_threshold_does_not_read_test_frames():
    class Bomb(np.ndarray):
        def __new__(cls):
            return np.zeros((4, 4, 3)).view(cls)

        def __array_finalize__(self, obj):
            pass

        def __getitem__(self, item):
            raise AssertionError("test frame RGB was read while choosing the threshold")

        def sum(self, axis=None, dtype=None):
            raise AssertionError("test frame RGB was read while choosing the threshold")

    rgb = np.zeros((4, 4, 3))
    rgb[:2, :2, 0] = 0.8
    rgb[:2, :2, 1] = 0.2
    rgb[2:, :, 0] = 0.2
    rgb[2:, :, 1] = 0.8
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    train = Frame(index=0, rgb=rgb, dolp=np.zeros((4, 4)), cube=np.zeros((4, 4, 8)), mask=mask)
    decoy = Frame(
        index=1,
        rgb=Bomb(),
        dolp=np.zeros((4, 4)),
        cube=np.zeros((4, 4, 8)),
        mask=np.ones((4, 4), dtype=bool),
    )
    threshold, score = choose_color_threshold([train, decoy], [0])
    assert threshold == pytest.approx(0.21)
    assert score == pytest.approx(1.0)


def test_split_ids_do_not_overlap():
    assert TRAIN_IDS == list(range(20)) + list(range(24, 28))
    assert TEST_IDS == list(range(20, 24)) + list(range(28, 32))
    assert set(TRAIN_IDS).isdisjoint(TEST_IDS)


def test_synthetic_probe_writes_finite_iou(tmp_path: Path):
    def frame(index: int, positive: bool) -> Frame:
        rgb = np.zeros((4, 4, 3))
        dolp = np.zeros((4, 4))
        cube = np.zeros((4, 4, 8))
        mask = np.zeros((4, 4), dtype=bool)
        if positive:
            rgb[:2, :2, 0] = 1.0
            dolp[:2, :2] = 0.8
            cube[:2, :2, -1] = 2.0
            mask[:2, :2] = True
        else:
            rgb[:, :, :] = 0.3
        return Frame(index=index, rgb=rgb, dolp=dolp, cube=cube, mask=mask)

    frames = [frame(0, True), frame(1, False), frame(2, True)]
    payload = run_probe(
        frames,
        tmp_path,
        train_ids=[0, 1],
        test_ids=[2],
        film_ids={0, 2},
        overlay_ids=(),
    )
    test_iou = payload["methods"]["rgb"]["test"]["iou"]
    assert np.isfinite(test_iou)
    assert 0.0 <= test_iou <= 1.0
    written = (tmp_path / "metrics.json").read_text()
    assert "definition_iou" in written
    assert (tmp_path / "comparison.png").is_file()
