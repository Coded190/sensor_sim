"""Axis recovery, material binding, and planted checks for Infinigen packages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sensor_sim.scene_gen.asset_realism import AssetRealismError, object_bsdf
from sensor_sim.scene_gen.infinigen_checks import (
    InfinigenCheckError,
    require_appliance_access,
    require_camera_sees_room,
    require_chair_near_desk,
    require_color_space,
    require_displacement_realized,
    require_door_clear,
    require_doorway_clear,
    require_frame_measured,
    require_instance_transform,
    require_material_assignment,
    require_monitor_toward_user,
    require_monitor_upright,
    require_normal_convention,
    require_regions,
    require_scale,
    require_separated,
    require_sofa_near_television,
    require_supported,
    require_texture,
    require_uvs,
)
from sensor_sim.scene_gen.infinigen_package import MESH_FORMAT_REASON, _category, displacement_audit, normal_convention, vertical_fov_degrees, world_from_exported_vertex


def test_factory_suffix_is_not_the_category():
    assert _category("SofaFactory(1).spawn_asset(2)") == "Sofa"
    assert _category("living-room_0/0.floor") == "living"


def test_floor_vertex_returns_to_the_blend_world_point():
    exported = [-6.046027, 0.000618, -1.458727]
    translation = [7.174103, 7.169349, 0.127457]
    world = world_from_exported_vertex(exported, translation)
    assert world == pytest.approx([1.128076, 8.628076, 0.128075], abs=1e-5)


def test_vertical_fov_matches_the_cycles_intrinsics():
    fov = vertical_fov_degrees(15.0, 32.0, 1280, 720)
    assert fov == pytest.approx(61.927513, abs=1e-4)


def test_normal_convention_comes_from_the_bake_setting():
    record = normal_convention("TANGENT", "POS_Y")
    assert record["flip_green"] is False
    assert record["orientation"] == "opengl_tangent"
    with pytest.raises(Exception, match="normal_green_untested"):
        normal_convention("TANGENT", "UNKNOWN")


def test_bump_is_not_recorded_as_geometric_displacement():
    audit = displacement_audit([{"displacement_method": "BUMP"}])
    assert audit["claim"] == "no_exported_geometric_displacement"
    assert audit["mitsuba_displacement_mode"] == "NONE"


def test_ply_is_not_the_textured_interchange():
    assert "vertex_colors" in MESH_FORMAT_REASON


def test_flat_diffuse_assignment_is_rejected():
    with pytest.raises(AssetRealismError, match="flat_diffuse_removed"):
        object_bsdf({"id": "sofa"})


def _box(origin, size) -> np.ndarray:
    origin = np.asarray(origin, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    return np.vstack([origin, origin + size])


def test_planted_arrangement_failures_name_their_reason():
    sofa = _box([0, 0, 0], [2, 1, 0.8])
    television = _box([0.5, 2.0, 0.2], [1.2, 0.4, 0.7])
    require_sofa_near_television(sofa, television)
    with pytest.raises(InfinigenCheckError, match="sofa_away_from_television"):
        require_sofa_near_television(sofa, _box([0, 8, 0], [1, 0.4, 0.6]))
    require_monitor_toward_user([0, -1, 0], [0, -2, 1], [0, 0, 1])
    with pytest.raises(InfinigenCheckError, match="monitor_away_from_user"):
        require_monitor_toward_user([0, 1, 0], [0, -2, 1], [0, 0, 1])
    with pytest.raises(InfinigenCheckError, match="upside_down_monitor"):
        require_monitor_upright([0, 0, -1])
    chair = _box([0, 0, 0], [0.5, 0.5, 0.9])
    desk = _box([0.6, 0, 0], [1.2, 0.6, 0.75])
    require_chair_near_desk(chair, desk)
    with pytest.raises(InfinigenCheckError, match="chair_away_from_desk"):
        require_chair_near_desk(chair, _box([3, 0, 0], [1, 0.6, 0.7]))
    with pytest.raises(InfinigenCheckError, match="blocked_refrigerator_door"):
        require_door_clear(_box([0, 0, 0], [0.8, 0.8, 1.8]), _box([0.2, 0.2, 0], [0.4, 0.4, 1]))
    with pytest.raises(InfinigenCheckError, match="inaccessible_microwave"):
        require_appliance_access(_box([0, 0, 1], [0.5, 0.4, 0.3]), _box([0, -0.1, 0.9], [0.6, 0.2, 0.5]), _box([0, -0.2, 1], [0.2, 0.2, 0.2]))
    with pytest.raises(InfinigenCheckError, match="blocked_doorway"):
        require_doorway_clear(_box([0, 0, 0], [0.9, 0.2, 2.1]), _box([0.2, 0, 0], [0.4, 0.4, 1]))
    with pytest.raises(InfinigenCheckError, match="unsupported_object"):
        require_supported(_box([0, 0, 0.4], [0.2, 0.2, 0.2]), 0.0)
    with pytest.raises(InfinigenCheckError, match="intersection"):
        require_separated(_box([0, 0, 0], [1, 1, 1]), _box([0.2, 0.2, 0.2], [1, 1, 1]))


def test_planted_material_and_frame_failures_name_their_reason():
    with pytest.raises(InfinigenCheckError, match="missing_required_texture"):
        require_texture(set(), "base_color")
    with pytest.raises(InfinigenCheckError, match="wrong_color_space"):
        require_color_space({"role": "base_color", "color_space": "non_color"})
    with pytest.raises(InfinigenCheckError, match="missing_uvs"):
        require_uvs(0)
    with pytest.raises(InfinigenCheckError, match="missing_material_assignment"):
        require_material_assignment(["default"])
    with pytest.raises(InfinigenCheckError, match="bad_frame"):
        require_frame_measured("category_default")
    with pytest.raises(InfinigenCheckError, match="dropped_material_region"):
        require_regions(2, 1)
    with pytest.raises(InfinigenCheckError, match="wrong_normal_convention"):
        require_normal_convention(True, "POS_Y")
    with pytest.raises(InfinigenCheckError, match="displacement_not_in_mesh"):
        require_displacement_realized("DISPLACEMENT", False)
    with pytest.raises(InfinigenCheckError, match="wrong_scale"):
        require_scale(2.0, 1.0)
    with pytest.raises(InfinigenCheckError, match="wrong_instance_transform"):
        require_instance_transform(np.eye(4), np.eye(4) + 0.01)
    with pytest.raises(InfinigenCheckError, match="camera_sees_opening"):
        require_camera_sees_room(np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]))


def test_replay_rebuilds_a_frozen_scene(tmp_path: Path):
    from sensor_sim.scene_gen.infinigen_package import build_package
    from sensor_sim.scene_gen.mitsuba_scene_package import replay_infinigen_scene

    obj = tmp_path / "export" / "Box_0" / "Box_0.obj"
    obj.parent.mkdir(parents=True)
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 0 1\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl painted\nf 1/1 2/2 3/3\n", encoding="utf-8")
    obj.with_suffix(".mtl").write_text("newmtl painted\nKd 0.7 0.2 0.1\nPr 0.4\nPm 0.0\nNi 1.5\nd 1.0\n", encoding="utf-8")
    graph = {
        "normal_space": "TANGENT",
        "normal_g": "POS_Y",
        "render_resolution": [64, 48],
        "cameras": [{"name": "camera_0_0", "lens_mm": 15.0, "sensor_width_mm": 32.0, "sensor_fit": "AUTO", "clip_start_m": 0.01, "clip_end_m": 100.0}],
        "lights": [{"name": "lamp", "light_type": "POINT", "energy": 40.0, "color": [1, 1, 1], "size": 0.0}],
        "objects": [
            {"name": "Box.0", "type": "MESH", "hide_render": False, "vertices": 3, "matrix_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "materials": [{"displacement_method": "BUMP"}], "parent": None},
            {"name": "camera_0_0", "type": "CAMERA", "hide_render": False, "vertices": 0, "matrix_world": [[1, 0, 0, 0.2], [0, 0, -1, -1.2], [0, 1, 0, 0.8], [0, 0, 0, 1]], "materials": [], "parent": None},
            {"name": "lamp", "type": "LIGHT", "hide_render": False, "vertices": 0, "matrix_world": [[1, 0, 0, 0.2], [0, 1, 0, -0.4], [0, 0, 1, 1.4], [0, 0, 0, 1]], "materials": [], "parent": None},
        ],
    }
    path = tmp_path / "graph.json"
    path.write_text(__import__("json").dumps(graph), encoding="utf-8")
    build_package(tmp_path / "export", path, tmp_path / "pkg")
    png = tmp_path / "pkg" / "preview.png"
    result = replay_infinigen_scene(tmp_path / "pkg" / "scene_manifest.json", png, spp=2, seed=3, width=32, height=24)
    assert result["replay_matches"] is True
    assert png.is_file() and png.stat().st_size > 0
    assert result["variant"] == "cuda_ad_spectral_polarized"


def test_package_round_trip_uses_region_materials(tmp_path: Path):
    from sensor_sim.scene_gen.infinigen_package import build_package

    obj = tmp_path / "export" / "Box_0" / "Box_0.obj"
    obj.parent.mkdir(parents=True)
    obj.write_text("v 0 0 0\nv 1 0 0\nv 1 0 1\nvt 0 0\nvt 1 0\nvt 1 1\nusemtl painted\nf 1/1 2/2 3/3\n", encoding="utf-8")
    (obj.with_suffix(".mtl")).write_text("newmtl painted\nKd 0.2 0.3 0.4\nPr 0.5\nPm 0.0\nNi 1.5\nd 1.0\n", encoding="utf-8")
    graph = {
        "normal_space": "TANGENT",
        "normal_g": "POS_Y",
        "render_resolution": [1280, 720],
        "cameras": [{"name": "camera_0_0", "lens_mm": 15.0, "sensor_width_mm": 32.0, "sensor_fit": "AUTO", "clip_start_m": 0.1, "clip_end_m": 100.0}],
        "lights": [{"name": "lamp", "light_type": "POINT", "energy": 10.0, "color": [1, 1, 1], "size": 0.0}],
        "objects": [
            {"name": "Box.0", "type": "MESH", "hide_render": False, "vertices": 3, "matrix_world": np.eye(4).tolist(), "materials": [{"displacement_method": "BUMP"}], "parent": None, "rotation_euler": [0, 0, 0], "scale": [1, 1, 1]},
            {"name": "camera_0_0", "type": "CAMERA", "hide_render": False, "vertices": 0, "matrix_world": np.eye(4).tolist(), "materials": [], "parent": "rig"},
            {"name": "lamp", "type": "LIGHT", "hide_render": False, "vertices": 0, "matrix_world": [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], "materials": [], "parent": None},
        ],
    }
    # The identity camera looks along -Z of an identity matrix, which is fine for the package.
    graph["objects"][0]["matrix_world"][0][3] = 2.0
    path = tmp_path / "graph.json"
    path.write_text(__import__("json").dumps(graph), encoding="utf-8")
    manifest = build_package(tmp_path / "export", path, tmp_path / "pkg")
    assert manifest["mesh_format"] == "obj"
    assert manifest["instances"][0]["matrix"][0][3] == pytest.approx(2.0)
    assert (tmp_path / "pkg" / "scene_manifest.json").is_file()
    region = manifest["instances"][0]["regions"][0]
    assert region["slot"] == "painted"
    assert region["spec"]["parameters"]["alpha"] == pytest.approx(0.25)
