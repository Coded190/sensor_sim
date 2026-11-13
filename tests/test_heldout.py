"""Catalog, film-only score, and device choice. No CUDA render and no training."""

from pathlib import Path

import numpy as np
import pytest

from sensor_sim.film.geometry import DRAGON_CENTER_XY, DRAGON_HEIGHT, read_ascii_ply, stand_dragon
from sensor_sim.film.heldout import (
    RENDER_PNGS,
    assert_no_dragon_key,
    main,
    pack_measurement,
    score_definitions,
)
from sensor_sim.film.sfp import _channels
from sensor_sim.film.sfp_math import (
    MIN_FREE_BYTES,
    angle_fractions,
    angular_error_summary,
    choose_device,
    mean_angular_error_deg,
)
from sensor_sim.film.sfp_scene import filmed_object_dict
from sensor_sim.film.shape_mesh import (
    FILM_THICKNESS,
    MIN_FILM_FRACTION,
    TEST_COUNTS,
    TRAIN_COUNTS,
    build_shape,
    is_watertight,
    project_ndc,
    shape_catalog,
)
from sensor_sim.stokes import S0_MIN


def _angled(degrees: float) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(degrees)
    prediction = np.array([np.sin(radians), 0.0, -np.cos(radians)])
    truth = np.array([0.0, 0.0, -1.0])
    return prediction, truth


def test_identical_normals_have_median_zero_and_cutoffs_are_strict():
    truth = np.array([[0.0, 0.0, -1.0], [0.2, 0.0, -0.9797959]])
    truth = truth / np.linalg.norm(truth, axis=-1, keepdims=True)
    same = angular_error_summary(truth, truth, np.array([True, True]))
    assert same["mae_deg"] == pytest.approx(0.0, abs=1e-6)
    assert same["median_deg"] == pytest.approx(0.0, abs=1e-6)
    assert same["fraction_below_11_25"] == pytest.approx(1.0)
    assert np.isnan(mean_angular_error_deg(truth, truth, np.array([False, False])))
    # The cutoffs are strict. 11.25 and 30.0 are not below themselves.
    exact = angle_fractions(np.array([11.25, 11.24, 30.0]))
    assert exact["fraction_below_11_25"] == pytest.approx(1.0 / 3.0)
    assert exact["fraction_below_22_5"] == pytest.approx(2.0 / 3.0)
    assert exact["fraction_below_30"] == pytest.approx(2.0 / 3.0)
    prediction = np.stack([_angled(angle)[0] for angle in (0.0, 10.0, 20.0)])
    target = np.stack([_angled(angle)[1] for angle in (0.0, 10.0, 20.0)])
    summary = angular_error_summary(prediction, target, np.ones(3, dtype=bool))
    assert summary["mae_deg"] == pytest.approx(10.0, abs=1e-3)
    assert summary["median_deg"] == pytest.approx(10.0, abs=1e-3)
    assert summary["fraction_below_11_25"] == pytest.approx(2.0 / 3.0)
    assert summary["pixels"] == 3
    empty = angular_error_summary(prediction, target, np.zeros(3, dtype=bool))
    assert empty["pixels"] == 0
    assert np.isnan(empty["median_deg"])


def test_film_mask_ignores_every_other_shape():
    from sensor_sim.film.sfp_math import object_pixel_mask

    index = np.array([[3, 4, 3], [1, 3, 0]])
    s0 = np.array([[1.0, 1.0, S0_MIN], [1.0, 1.0, 1.0]])
    mask = object_pixel_mask(index, {3}, s0)
    assert mask.tolist() == [[True, False, False], [False, True, False]]


def test_summary_keys_do_not_name_the_copper_mesh():
    payload = {
        "experiment": "01_cornell_film",
        "views": {"view_a": {"stokes": {"mae_deg": 1.0}, "film_pixels": 10}},
        "definitions": score_definitions(),
    }
    assert_no_dragon_key(payload)
    with pytest.raises(RuntimeError, match="copper mesh"):
        assert_no_dragon_key({"views": {"dragon_mae": 1.0}})


def test_s3_is_packed_and_is_not_a_network_input():
    shape = (4, 4)
    bundle = {
        "rgb": np.ones((*shape, 3)),
        "s0": np.full(shape, 0.4),
        "s1": np.full(shape, 0.1),
        "s2": np.full(shape, -0.2),
        "s3": np.full(shape, 0.05),
        "dolp": np.full(shape, 0.3),
        "aolp": np.full(shape, 0.2),
        "analyzer_000": np.full(shape, 0.2),
        "analyzer_045": np.full(shape, 0.2),
        "analyzer_090": np.full(shape, 0.2),
        "analyzer_135": np.full(shape, 0.2),
    }
    normals = np.zeros((*shape, 3))
    normals[..., 2] = -1.0
    view = np.zeros((*shape, 3))
    view[..., 2] = 1.0
    packed = pack_measurement(bundle, normals, np.ones(shape), view)
    assert set(RENDER_PNGS) >= {"rgb.png", "normal_gt.png", "dolp.png", "aolp.png", "s3.png"}
    assert packed["s3"].shape == shape
    before = _channels(packed, "stokes")
    packed["s3"] = packed["s3"] + 3.0
    after = _channels(packed, "stokes")
    assert before.shape[-1] == 9
    assert np.allclose(before, after)
    assert _channels(packed, "analyzer").shape[-1] == 7
    assert _channels(packed, "rgb").shape[-1] == 6


def test_device_uses_the_gpu_only_when_eight_gib_are_free():
    device, reason = choose_device(False, None)
    assert device == "cpu"
    assert "is false" in reason
    device, reason = choose_device(True, None)
    assert device == "cpu"
    assert "could not be read" in reason
    device, reason = choose_device(True, MIN_FREE_BYTES - 1)
    assert device == "cpu"
    assert "below" in reason
    device, reason = choose_device(True, MIN_FREE_BYTES)
    assert device == "cuda"
    assert "8 GiB" in reason


def test_catalog_counts_seeds_and_folders():
    catalog = shape_catalog(0)
    again = shape_catalog(0)
    assert [item["instance_seed"] for item in catalog] == [item["instance_seed"] for item in again]
    train = [item for item in catalog if item["split"] == "train"]
    test = [item for item in catalog if item["split"] == "test"]
    assert len(train) == 50 and len(test) == 10
    assert TRAIN_COUNTS["sphere"] == 4
    assert "sphere" not in TEST_COUNTS
    assert [item["folder"] for item in train] == [f"{index:03d}_{item['family']}" for index, item in enumerate(train, start=1)]
    assert [item["folder"] for item in test][0] == "001_ellipsoid"
    assert [item["folder"] for item in test][-1] == "010_cone"
    overlap = {item["instance_seed"] for item in train} & {item["instance_seed"] for item in test}
    assert overlap == set()


def test_every_catalog_mesh_is_closed_and_covers_the_frame():
    """One build of all 60 shapes. The hull fraction is the pre-render coverage check."""
    seen = set()
    for spec in shape_catalog(0):
        built = build_shape(spec)
        assert built["sagitta_m"] < FILM_THICKNESS
        assert built["frame_fraction"] >= MIN_FILM_FRACTION
        assert spec["family"] != "sphere" or spec["split"] == "train"
        if spec["family"] in seen:
            continue
        seen.add(spec["family"])
        assert is_watertight(built["solid"]["faces"])
        assert is_watertight(built["film"]["faces"])
    assert seen == set(TRAIN_COUNTS)


def test_dragon_stands_on_the_floor_at_the_recorded_height():
    root = Path(__file__).resolve().parents[1]
    mesh = read_ascii_ply(root / "assets" / "dragon" / "dragon_vrip_res4.ply")
    stood = stand_dragon(mesh["vertices"])
    assert stood[:, 2].min() == pytest.approx(0.0, abs=1e-8)
    assert stood[:, 2].max() == pytest.approx(DRAGON_HEIGHT)
    center_x = 0.5 * (stood[:, 0].min() + stood[:, 0].max())
    center_y = 0.5 * (stood[:, 1].min() + stood[:, 1].max())
    assert center_x == pytest.approx(DRAGON_CENTER_XY[0])
    assert center_y == pytest.approx(DRAGON_CENTER_XY[1])
    assert mesh["faces"].shape[1] == 3


def test_ndc_matches_mitsuba_and_the_lamp_does_not_follow_the_object():
    mitsuba = pytest.importorskip("mitsuba")
    mitsuba.set_variant("scalar_rgb")
    origin = [0.0, -0.2, 0.08]
    target = [0.02, 0.0, 0.05]
    scene = mitsuba.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "path"},
            "sensor": {
                "type": "perspective",
                "fov": 55,
                "to_world": mitsuba.ScalarTransform4f.look_at(origin=origin, target=target, up=[0, 0, 1]),
                "film": {"type": "hdrfilm", "width": 64, "height": 64, "rfilter": {"type": "box"}},
            },
        }
    )
    from sensor_sim.film.capture import project_point

    sensor = scene.sensors()[0]
    for point in (target, [0.06, 0.0, 0.05], [0.02, 0.0, 0.09]):
        _row, _column, ndc = project_point(sensor, point, 64, 64)
        estimated = project_ndc(np.array([point], dtype=np.float64), origin, target, 55.0)
        assert estimated[0] == pytest.approx(ndc[:2], abs=1e-5)

    def camera(x_position: float) -> dict:
        return {
            "position": [x_position, -0.3, 0.08],
            "target": [x_position, 0.0, 0.05],
            "up": [0.0, 0.0, 1.0],
            "fov_deg": 55.0,
        }

    left = filmed_object_dict(Path("room"), Path("mesh"), camera(0.0), resolution=(8, 8), spp=4, max_depth=8, polarized=True)
    right = filmed_object_dict(Path("room"), Path("mesh"), camera(0.1), resolution=(8, 8), spp=4, max_depth=8, polarized=True)
    left_light = np.array(left["light"]["to_world"].matrix, dtype=np.float64)
    right_light = np.array(right["light"]["to_world"].matrix, dtype=np.float64)
    assert left_light == pytest.approx(right_light)
    left_camera = np.array(left["sensor"]["to_world"].matrix, dtype=np.float64)
    right_camera = np.array(right["sensor"]["to_world"].matrix, dtype=np.float64)
    assert left_camera != pytest.approx(right_camera)
    assert left["film-shell"]["bsdf"] == {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0}
    assert left["solid"]["bsdf"]["type"] == "diffuse"
    assert left["integrator"]["type"] == "stokes"
    assert left["integrator"]["integrator"]["max_depth"] == 8
    assert "polarizer" not in left["light"]["emitter"]["type"]
    assert left["light"]["emitter"]["radiance"]["value"] == pytest.approx(18.0)


def test_cornell_meshes_are_written_without_blender(tmp_path):
    from sensor_sim.film.sfp_scene import write_cornell_film

    poses = write_cornell_film(tmp_path)
    expected = [
        "ball.obj",
        "film_cap.obj",
        "platform.obj",
        "dragon.obj",
        "floor.obj",
        "ceiling.obj",
        "wall-left.obj",
        "wall-right.obj",
        "wall-back.obj",
    ]
    missing = [name for name in expected if not (tmp_path / name).is_file()]
    assert missing == []
    assert len(poses["points"]["film"]) == 3
    assert poses["film_roughness"] == pytest.approx(0.0)
    assert poses["film_ior"] == pytest.approx(1.5)
    assert "dragon" not in poses


def test_cuda_stays_visible_for_the_normal_run_and_hidden_for_the_mask_run():
    root = Path(__file__).resolve().parents[1] / "src" / "sensor_sim" / "film"
    unet = (root / "unet.py").read_text()
    trainer = (root / "train.py").read_text()
    heldout = (root / "heldout.py").read_text()
    assignment = 'os.environ["CUDA_VISIBLE_DEVICES"]'
    assert assignment not in unet
    assert assignment not in heldout
    assert "film.train" not in heldout
    assert assignment in trainer
    assert trainer.index(assignment) < trainer.index("import torch")


def test_usage_rejects_an_unknown_experiment():
    with pytest.raises(SystemExit, match="usage"):
        main(["nope"])
