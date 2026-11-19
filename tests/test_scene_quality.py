"""Scene-quality pilots: grouped placement, framing, masks, daylight, and locked inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sensor_sim.scene_gen.asset_package import read_index
from sensor_sim.scene_gen.indoor_mitsuba import _bitmap_names, render_stokes
from sensor_sim.scene_gen.indoor_transform import deserialize_matrix, translation_matrix
from sensor_sim.scene_gen.mesh_io import write_obj
from sensor_sim.scene_gen.metric_audit import CHANNEL_NAMES
from sensor_sim.scene_gen.scene_quality import (
    RECIPES,
    bundle_hashes,
    close_table_camera,
    daylight_emitter,
    measure_window,
    place_recipe,
    quality_room,
    write_gallery,
)
from sensor_sim.scene_gen.structural import box_mesh
from sensor_sim.scene_gen.tabletop_layout import INSET_M, _inside_rect, _inset_rect, load_mesh
from sensor_sim.scene_gen.tabletop_render import _project_point, _scene, instance_visibility
from sensor_sim.scene_gen.indoor_contact import world_vertices

CATALOG = Path("assets/candidates")
OLD_HASHES = Path("dataset/qualification/scene_quality_v2/old_bundle_hashes.json")
QUALITY = Path("dataset/qualification/scene_quality_v2")
INDEX_SHA256 = "081bf07291a4cc3f8f2af1fe27b8f8beb0164153100244aefa25df1ba209c157"
PROTECTED = {
    "src/sensor_sim/film/sfp_scene.py": "3287fc531e50069d3fb912292f3a3bc907fa1e80a39d1c1a30398a8940ea3ea0",
    "src/sensor_sim/film/mitsuba_scene.py": "6bdd40ac8ed12a243b0693da78c87219f3c5d51d66466b8772cd42c89d9b0653",
    "src/sensor_sim/scenes.py": "752e7c5c7aa067eaaa77559b307066a8c57d72a481bd8853c1a0194d80aa0022",
    "src/sensor_sim/scene_gen/xml_patch.py": "6b8ced6842897aa462eb1d855d4f7fe8cf8702e9f796af5bebe497d4b3942e32",
    "src/sensor_sim/scene_gen/generate.py": "78ee05ab9afe551436b0f35836880e8c6b1680e8c559f642083269ff3931ddf8",
    "src/sensor_sim/scene_gen/cli.py": "29a5ddc431bc4fa13887afed2fab774f485331d7aa5e97d0732b7d42a806763f",
    "src/sensor_sim/scene_gen/meshcheck.py": "051676b437f4ae4394d201b8470cf179ddeebbbb3d25f3b96479f3fbeb684198",
    "src/sensor_sim/scene_gen/placement.py": "650e9b36e9fb4422877b4cb392b9d7253c258cc1385329be502661a9d6ca466d",
    "src/sensor_sim/scene_gen/indoor_mitsuba.py": "dfbd5e9019be0d512f787ce2ec1ebffd78cf60cf52fff427d00a8a1e6c26d2b3",
    "src/sensor_sim/scene_gen/structural.py": "6e839ad9debb3bcbc4b16c1016f1eac77b6af58e8f1dc93b49c5302e007db1a7",
}
SCENE_JSON = {
    "dataset/qualification/task_scenes/book_on_table/e54a5e18636df8e7/scene.json": "fd783a414450af9130225eb6a7d6dc345544cbe0798f8d4b6bfc59b5829879c0",
    "dataset/qualification/task_scenes/cup_on_shelf/8a0a6b98b048ef83/scene.json": "464bb6215a98dc0f9421646588e882332f8d3a7008c161633a11573c551fc6fb",
    "dataset/qualification/task_scenes/book_inside_crate/e547b5c803391dbe/scene.json": "46cf481bc6113876a7d6f599d6e2d6ba9f3e83164f2e1b6885f60a7f74c6437e",
    "dataset/qualification/task_scenes/crate_and_book_on_table/44ac50f76b66afcb/scene.json": "8cd3f9a43d86e9a12c6d8d186957aaf28ad45cc7f4984dbc9e417063dff64591",
    "dataset/qualification/task_scenes/cup_near_book/95d8b3f496b78836/scene.json": "51a2bdaca8d2bd5d86dc384b6ea1a313275d548b9d9b4804d72f07294fa5dca9",
    "dataset/qualification/task_scenes/objects_in_workspace/3cf07be40ab68252/scene.json": "07137d96cd7a86a5e96a788cf5ac974b668642fdaf3f279408fec9fb78a9fe48",
    "dataset/qualification/indoor_scenes/scene_a/scene.json": "3aabf6862bf8c24de6d4e96975aa11ff3b93c07e4c9acc3d5115311a0228005a",
    "dataset/qualification/indoor_scenes/scene_b/scene.json": "248b6f7db28d3eb93ed192d788ec46db1f49d33214ba0ea024724a2673b5358b",
    "dataset/qualification/indoor_scenes/scene_c/scene.json": "d4d5f17d1c0d4641cb96060e55fa1c5abda861493eb6060a064ee082d3fb57a7",
}


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ndc_peak(camera: dict, polygon: np.ndarray, width: int, height: int) -> float:
    peak = 0.0
    for point in polygon:
        pixel = _project_point(camera, width, height, point)
        assert pixel is not None
        ndc_x = (pixel[0] + 0.5) / width * 2.0 - 1.0
        ndc_y = (pixel[1] + 0.5) / height * 2.0 - 1.0
        peak = max(peak, abs(ndc_x), abs(ndc_y))
        assert 0.0 <= pixel[0] < width
        assert 0.0 <= pixel[1] < height
    return peak


def test_grouped_placement_is_deterministic_and_stays_inside_the_inset():
    room = quality_room(CATALOG)
    assert room["valid"] is True
    assert room["intersections"] == []
    first = place_recipe(CATALOG, "quality_controlled", room, 17201)
    second = place_recipe(CATALOG, "quality_controlled", room, 17201)
    assert first["valid"] and second["valid"]
    assert first["poses"] == second["poses"]
    inset = _inset_rect(room["table_support_world"], INSET_M)
    for instance_id, matrix in first["poses"].items():
        mesh = load_mesh(CATALOG, instance_id.rsplit("__", 1)[0])
        world = world_vertices(mesh["vertices"], deserialize_matrix(matrix))
        assert _inside_rect(world[:, :2], inset)
    occupancy = first["collision_log"]["occupancy"]
    assert occupancy["nearest_clearance_m"] >= RECIPES["quality_controlled"]["separation_m"]
    assert 0.10 <= occupancy["occupied_fraction"] <= 0.15
    zones = {row["instance_id"]: row["zone"] for row in first["collision_log"]["attempts"]}
    assert set(zones.values()) == {"dining"}
    mixed = place_recipe(CATALOG, "quality_mixed", room, 17202)
    assert mixed["valid"]
    mixed_zones = {row["instance_id"]: row["zone"] for row in mixed["collision_log"]["attempts"]}
    assert mixed_zones["prop_plate__1"] == "dining"
    assert mixed_zones["prop_phone__1"] == "work"
    assert mixed_zones["prop_marker__1"] == "work"
    assert 0.15 <= mixed["collision_log"]["occupancy"]["occupied_fraction"] <= 0.25


def test_dense_pack_is_valid_and_covers_more_than_the_old_baseline():
    room = quality_room(CATALOG)
    placed = place_recipe(CATALOG, "quality_dense", room, 17203)
    assert placed["valid"]
    occupancy = placed["collision_log"]["occupancy"]
    # Old dense scenes cover about 0.096 of the 0.840 m^2 support. This pack must clear that.
    assert occupancy["occupied_fraction"] > 0.096
    assert occupancy["occupied_fraction"] >= 0.25
    assert occupancy["nearest_clearance_m"] >= RECIPES["quality_dense"]["separation_m"]
    assert occupancy["instance_count"] == 24
    saved = json.loads(next((QUALITY / "quality_dense").glob("*/bundle.json")).read_text(encoding="utf-8"))
    saved_poses = {item["instance_id"]: item["matrix"] for item in saved["resolved"]["poses"]}
    for instance_id, matrix in placed["poses"].items():
        assert np.allclose(matrix, saved_poses[instance_id])


def test_chair_yaw_points_the_seat_toward_the_table():
    room = quality_room(CATALOG)
    mesh = load_mesh(CATALOG, "prop_chair")
    world = world_vertices(mesh["vertices"], room["matrices"]["prop_chair"])
    local_y = mesh["vertices"][:, 1]
    back = world[np.argsort(local_y)[-8:], 1].mean()
    front = world[np.argsort(local_y)[:8], 1].mean()
    table_near = float(room["table_support_world"][:, 1].min())
    assert front > back
    assert front < table_near
    assert abs(float(world[:, 2].min()) - 0.05) < 1e-6
    assert all(row["passed"] and row["gap_m"] == 0.0 for row in room["contacts"])


def test_close_camera_frames_a_known_support_quad_with_an_eight_percent_margin():
    quad = np.array(
        [[1.55, 1.70, 0.79], [2.75, 1.70, 0.79], [2.75, 2.40, 0.79], [1.55, 2.40, 0.79]],
        dtype=np.float64,
    )
    camera = close_table_camera(quad, margin=0.08, aspect=1600 / 900)
    peak = _ndc_peak(camera, quad, 1600, 900)
    # peak == 1 / (1 + margin). The accepted band is a 5–10 percent extra margin.
    assert 1.0 / 1.10 <= peak <= 1.0 / 1.05
    assert camera["origin"][1] > 0.12
    assert camera["origin"][2] > quad[0, 2]


def test_roi_counts_and_a_fully_hidden_box_have_stable_integer_ids(tmp_path: Path):
    front_vertices, front_faces = box_mesh([-0.5, -0.02, -0.4], [0.5, 0.02, 0.4])
    hidden_vertices, hidden_faces = box_mesh([-0.05, -0.02, -0.05], [0.05, 0.02, 0.05])
    front = tmp_path / "front.obj"
    hidden = tmp_path / "hidden.obj"
    front.write_bytes(write_obj(front_vertices, front_faces, name="front"))
    hidden.write_bytes(write_obj(hidden_vertices, hidden_faces, name="hidden"))
    instances = [
        {"id": "front", "asset_id": "front", "role": "required", "mesh": front, "matrix": translation_matrix([0.0, -0.30, 0.0]), "bsdf": {"type": "diffuse", "reflectance": [0.5, 0.5, 0.5]}},
        {"id": "hidden", "asset_id": "hidden", "role": "required", "mesh": hidden, "matrix": translation_matrix([0.0, 0.40, 0.0]), "bsdf": {"type": "diffuse", "reflectance": [0.2, 0.2, 0.2]}},
    ]
    camera = {"name": "test", "origin": [0.0, -1.8, 0.0], "target": [0.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0], "fov_deg": 40.0}
    support = np.array([[-0.8, -0.50, -0.5], [0.8, -0.50, -0.5], [0.8, -0.50, 0.5], [-0.8, -0.50, 0.5]], dtype=np.float64)
    first = instance_visibility(instances, camera, (64, 48), support_polygon=support, return_image=True)
    second = instance_visibility(instances, camera, (64, 48), support_polygon=support, return_image=True)
    assert np.array_equal(first.pop("id_image"), second.pop("id_image"))
    assert np.array_equal(first.pop("roi_mask"), second.pop("roi_mask"))
    assert first == second
    rows = {row["instance_id"]: row for row in first["instances"]}
    assert rows["front"]["visible_pixels"] > 0
    assert rows["front"]["roi_visible_pixels"] > 0
    assert first["roi_pixels"] > 0
    assert rows["hidden"]["unoccluded_projected_pixels"] > 0
    assert rows["hidden"]["visible_pixels"] == 0
    assert rows["hidden"]["occlusion"] == "full"
    assert [row["label"] for row in first["instances"]] == [1, 2]
    scene = _scene(instances, camera, (64, 48), {"type": "constant", "radiance": {"type": "rgb", "value": 1.0}})
    stokes = render_stokes(scene, tmp_path / "tiny.exr", 1, 3)
    mask = instance_visibility(instances, camera, (64, 48), return_image=True)["id_image"]
    assert list(mask.shape) == stokes["shape"][:2]
    assert mask.dtype == np.int32


def test_daylight_rectangle_sits_outside_the_measured_opening():
    window = measure_window(CATALOG)
    assert window["opening_is_clear"] is True
    assert window["solid_ray"]["hit"] is True
    emitter = daylight_emitter(8.0)
    matrix = np.array(emitter["to_world"].matrix, dtype=np.float64)
    for x_value, y_value in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        point = matrix @ np.array([x_value, y_value, 0.0, 1.0])
        assert point[1] > 3.44
        assert 0.62 <= point[0] <= 1.82
        assert 0.95 <= point[2] <= 1.95


def test_saved_renders_reopen_as_fifteen_channel_float_exrs():
    import mitsuba as mi

    from sensor_sim.scene_gen.indoor_mitsuba import require_variant

    require_variant()
    paths = sorted(QUALITY.glob("*/*/render_*/*.exr"))
    assert len(paths) == 12
    for path in paths:
        bitmap = mi.Bitmap(str(path))
        image = np.array(bitmap)
        assert image.dtype == np.float32
        assert image.shape[2] == 15
        assert np.isfinite(image).all()
        assert _bitmap_names(bitmap) == list(CHANNEL_NAMES)
        mask = np.load(path.with_name(path.stem + "_instance_ids.npy"))
        assert mask.shape == image.shape[:2]
        assert mask.dtype == np.int32


def test_gallery_links_resolve_and_state_the_pending_review(tmp_path: Path):
    review = tmp_path / "review" / "index.html"
    write_gallery(review, QUALITY, Path("dataset/qualification/tabletop_scenes"))
    text = review.read_text(encoding="utf-8")
    assert "Human visual review, admission, and registration are pending." in text
    assert "comparison_bowl_diffuse" in text
    assert "ceiling_plus_daylight" in text or "Ceiling plus daylight" in text
    for token in text.split('href="')[1:]:
        href = token.split('"', 1)[0]
        assert not href.startswith("http")
        assert (review.parent / href).resolve().exists(), href


def test_previous_bundles_scenes_and_catalog_statuses_are_unchanged():
    assert bundle_hashes() == json.loads(OLD_HASHES.read_text(encoding="utf-8"))
    assert len(bundle_hashes()) == 15
    for path, digest in {**PROTECTED, **SCENE_JSON}.items():
        assert _sha(path) == digest
    index_path = CATALOG / "index.json"
    assert _sha(str(index_path)) == INDEX_SHA256
    index = read_index(index_path)
    for asset_id in (
        "prop_plate",
        "prop_bowl",
        "prop_cup",
        "prop_chair",
        "table_rect",
        "room_wall_north",
        "prop_book",
    ):
        assert index["assets"][asset_id]["1.0.0"]["status"] == "renderer_validated"


@pytest.mark.parametrize("recipe_id", ["quality_controlled", "quality_mixed", "quality_dense"])
def test_recipe_ids_and_seeds_are_new(recipe_id: str):
    assert recipe_id in RECIPES
    assert RECIPES[recipe_id]["separation_m"] >= 0.008
