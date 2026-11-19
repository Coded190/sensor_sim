"""Renderer, visibility, comparison, and regression checks for the tabletop milestone."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sensor_sim.scene_gen.asset_package import verify_package
from sensor_sim.scene_gen.indoor_mitsuba import render_stokes
from sensor_sim.scene_gen.indoor_transform import identity_matrix, translation_matrix
from sensor_sim.scene_gen.mesh_io import read_obj, write_obj
from sensor_sim.scene_gen.metric_audit import CHANNEL_NAMES
from sensor_sim.scene_gen.structural import box_mesh
from sensor_sim.scene_gen.tabletop_assets import material_for
from sensor_sim.scene_gen.tabletop_render import (
    _bounds,
    _frame_camera,
    _inspection_light,
    _scene,
    instance_visibility,
    mitsuba_bsdf,
    write_gallery,
)
from sensor_sim.scene_gen.task_schema import TaskError
import pytest

PROTECTED = {
    "src/sensor_sim/film/sfp_scene.py": "3287fc531e50069d3fb912292f3a3bc907fa1e80a39d1c1a30398a8940ea3ea0",
    "src/sensor_sim/film/mitsuba_scene.py": "6bdd40ac8ed12a243b0693da78c87219f3c5d51d66466b8772cd42c89d9b0653",
    "src/sensor_sim/scenes.py": "752e7c5c7aa067eaaa77559b307066a8c57d72a481bd8853c1a0194d80aa0022",
    "src/sensor_sim/scene_gen/xml_patch.py": "6b8ced6842897aa462eb1d855d4f7fe8cf8702e9f796af5bebe497d4b3942e32",
    "src/sensor_sim/scene_gen/generate.py": "78ee05ab9afe551436b0f35836880e8c6b1680e8c559f642083269ff3931ddf8",
    "src/sensor_sim/scene_gen/cli.py": "29a5ddc431bc4fa13887afed2fab774f485331d7aa5e97d0732b7d42a806763f",
    "src/sensor_sim/scene_gen/meshcheck.py": "051676b437f4ae4394d201b8470cf179ddeebbbb3d25f3b96479f3fbeb684198",
    "src/sensor_sim/scene_gen/placement.py": "650e9b36e9fb4422877b4cb392b9d7253c258cc1385329be502661a9d6ca466d",
}
def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_each_bsdf_loads_in_the_polarized_variant():
    import mitsuba as mi

    mi.set_variant("cuda_ad_spectral_polarized")
    for asset_id in ("prop_bowl", "prop_bowl_glossy", "prop_can", "prop_can_rough", "prop_cup"):
        material = material_for(asset_id) if asset_id != "prop_cup" else {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0}
        mapped = mitsuba_bsdf(material)
        scene = mi.load_dict({"type": "scene", "shape": {"type": "sphere", "bsdf": mapped}})
        assert scene.shapes()[0].bsdf() is not None
        if asset_id != "prop_cup":
            assert mapped["type"] == material["type"]


def test_unknown_bsdf_is_rejected():
    with pytest.raises(TaskError):
        mitsuba_bsdf({"type": "plastic", "diffuse_reflectance": [0.2, 0.2, 0.2]})


def test_small_stokes_exr_matches_the_channel_contract(tmp_path: Path):
    mesh = Path("assets/candidates/prop_small_box/1.0.0/geometry/render.obj")
    vertices, _polygons = read_obj(mesh)
    bounds = _bounds(vertices)
    camera = _frame_camera(bounds, "front", 45.0, 1.25)
    material = material_for("prop_small_box")
    scene = _scene(
        [{"id": "prop_small_box", "mesh": mesh, "matrix": identity_matrix(), "bsdf": material}],
        camera,
        (32, 24),
        _inspection_light(bounds, camera),
    )
    report = render_stokes(scene, tmp_path / "small.exr", 2, 3)
    assert report["channel_names"] == list(CHANNEL_NAMES)
    assert report["exr_names"] == list(CHANNEL_NAMES)
    assert report["dtype"] == "float32"
    assert report["finite"] is True
    assert report["shape"] == [24, 32, 15]


def test_visibility_is_deterministic_and_sees_occlusion(tmp_path: Path):
    front_vertices, front_faces = box_mesh([-0.4, -0.02, -0.4], [0.4, 0.02, 0.4])
    back_vertices, back_faces = box_mesh([-0.08, -0.02, -0.08], [0.08, 0.02, 0.08])
    front = tmp_path / "front.obj"
    back = tmp_path / "back.obj"
    front.write_bytes(write_obj(front_vertices, front_faces, name="front"))
    back.write_bytes(write_obj(back_vertices, back_faces, name="back"))
    instances = [
        {"id": "front", "asset_id": "front", "role": "required", "mesh": front, "matrix": translation_matrix([0.0, -0.25, 0.0])},
        {"id": "back", "asset_id": "back", "role": "required", "mesh": back, "matrix": translation_matrix([0.0, 0.45, 0.0])},
    ]
    camera = {"name": "test", "origin": [0.0, -1.6, 0.0], "target": [0.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0], "fov_deg": 35.0}
    first = instance_visibility(instances, camera, (48, 36))
    second = instance_visibility(instances, camera, (48, 36))
    assert first == second
    rows = {row["instance_id"]: row for row in first["instances"]}
    assert rows["front"]["visible_pixels"] > 0
    assert rows["back"]["unoccluded_projected_pixels"] > 0
    assert rows["back"]["visible_pixels"] < rows["back"]["unoccluded_projected_pixels"]
    assert rows["back"]["occlusion"] in {"partial", "full"}


def test_comparison_pairs_hold_the_room_fixed():
    root = Path("dataset/qualification/tabletop_scenes")
    pairs = (
        ("comparison_bowl_diffuse", "comparison_bowl_glossy", "prop_bowl", "prop_bowl_glossy"),
        ("comparison_can_polished", "comparison_can_rough", "prop_can", "prop_can_rough"),
        ("comparison_geometry_plate", "comparison_geometry_apple", "prop_plate", "prop_apple"),
    )
    for left_id, right_id, left_asset, right_asset in pairs:
        left = json.loads(next((root / left_id).glob("*/bundle.json")).read_text(encoding="utf-8"))
        right = json.loads(next((root / right_id).glob("*/bundle.json")).read_text(encoding="utf-8"))
        assert left["resolved"]["collision_log"]["cameras"] == right["resolved"]["collision_log"]["cameras"]
        assert left["task"]["seed"] == right["task"]["seed"] == 17121
        left_pose = {item["instance_id"]: item["matrix"] for item in left["resolved"]["poses"]}
        right_pose = {item["instance_id"]: item["matrix"] for item in right["resolved"]["poses"]}
        assert left_pose[f"{left_asset}__1"] == right_pose[f"{right_asset}__1"]
        for instance_id in ("room_floor", "room_ceiling", "table_1", "shelf_1", "chair_1"):
            assert left_pose[instance_id] == right_pose[instance_id]
        if left_id.startswith("comparison_geometry"):
            left_render = json.loads(next((root / left_id).glob("*/render_17111/validation.json")).read_text(encoding="utf-8"))
            right_render = json.loads(next((root / right_id).glob("*/render_17111/validation.json")).read_text(encoding="utf-8"))
            left_bsdf = next(item["bsdf"] for item in left_render["assets"] if item["asset_id"] == left_asset)
            right_bsdf = next(item["bsdf"] for item in right_render["assets"] if item["asset_id"] == right_asset)
            assert left_bsdf["reflectance"] == right_bsdf["reflectance"]
        if left_asset.startswith("prop_bowl") or left_asset.startswith("prop_can"):
            left_hash = verify_package(Path("assets/candidates") / left_asset / "1.0.0")["geometry"]["geometry_sha256"]
            right_hash = verify_package(Path("assets/candidates") / right_asset / "1.0.0")["geometry"]["geometry_sha256"]
            assert left_hash == right_hash


def test_gallery_links_resolve(tmp_path: Path):
    review = tmp_path / "review" / "index.html"
    write_gallery(review, Path("dataset/qualification/tabletop_assets"), Path("dataset/qualification/tabletop_scenes"))
    text = review.read_text(encoding="utf-8")
    assert "Human visual review, admission, and registration are pending." in text
    assert "comparison_bowl_diffuse" in text
    for token in text.split('href="')[1:]:
        href = token.split('"', 1)[0]
        if href.startswith("http"):
            raise AssertionError(href)
        target = (review.parent / href).resolve()
        assert target.exists(), href


def test_protected_sources_and_previous_scenes_are_unchanged():
    for path, digest in PROTECTED.items():
        assert _sha(path) == digest
    expected = {
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
    for path, digest in expected.items():
        assert _sha(path) == digest
