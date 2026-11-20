"""Component material seeds stay independent of geometry and wrap."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from sensor_sim.scene_gen.material_assign import assign_components, replay_assignment
from sensor_sim.scene_gen.material_spec import bitmap_settings


def _texture(path: Path) -> dict:
    Image.new("RGB", (4, 4), (20, 40, 200)).save(path)
    return {
        "role": "base_color",
        "path": path.name,
        "color_space": "srgb",
        "resolution": [4, 4],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parameter_class": "authored_appearance",
        "channel": "rgb",
        "normal_orientation": "opengl_tangent",
        "normal_strength": 1.0,
        "uv_transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }


def test_material_seed_selects_texture_and_replay_is_exact(tmp_path: Path):
    texture = _texture(tmp_path / "color.png")
    transforms = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2.0, 0.0, 0.0, 0.0, 2.0, 0.0]]
    body = {
        "instance_id": "widget",
        "component_id": "body",
        "families": ["pplastic"],
        "texture_sets": ["panel"],
        "uv_transforms": transforms,
        "geometry_seed": 11,
        "placement_seed": 12,
        "wrap_seed": 13,
    }
    lid = {
        "instance_id": "widget",
        "component_id": "lid",
        "families": ["conductor", "roughconductor"],
        "geometry_seed": 11,
        "placement_seed": 12,
        "wrap_seed": 13,
    }
    seen = set()
    for seed in range(1, 40):
        records = assign_components([body, lid], seed, {"panel": [texture]})
        assert records[0]["spec"]["family"] == "pplastic"
        assert records[1]["spec"]["family"] in {"conductor", "roughconductor"}
        assert records[0]["component_id"] == "body" and records[1]["component_id"] == "lid"
        chosen = tuple(records[0]["spec"]["sampled"]["uv_transform"])
        assert chosen in {tuple(item) for item in transforms}
        seen.add(chosen)
        restored = replay_assignment(records[0])
        assert restored["textures"] == records[0]["spec"]["textures"]
        assert restored["parameters"] == records[0]["spec"]["parameters"]
    assert len(seen) == 2
    moved = dict(body)
    moved["geometry_seed"] = 99
    moved["wrap_seed"] = 77
    original = assign_components([body], 5, {"panel": [texture]})[0]
    changed = assign_components([moved], 5, {"panel": [texture]})[0]
    assert changed["spec"]["parameters"] == original["spec"]["parameters"]
    assert changed["spec"]["textures"] == original["spec"]["textures"]
    assert changed["seeds"]["geometry"] == 99
    assert changed["seeds"]["wrap"] == 77
    assert original["seeds"]["geometry"] == 11


def test_pinned_scene_reads_uvs_and_rejects_trilinear(tmp_path: Path):
    import mitsuba as mi
    import numpy as np

    from sensor_sim.scene_gen.indoor_mitsuba import require_variant
    from sensor_sim.scene_gen.mesh_io import validate_uv
    from sensor_sim.scene_gen.texture_qualification import _panel_mesh, _panel_view, _render, _spec, _write_package

    require_variant()
    package = tmp_path / "panel" / "1.0.0"
    mesh = _panel_mesh()
    uv = validate_uv(mesh, texture_resolution=(256, 256), island_padding=0.0)
    uv["uv_method"] = "explicit_quad_parameterization"
    _write_package(package, mesh, uv, {"source": "unit"})
    color = package / "textures" / "base_color.png"
    try:
        mi.load_dict({"type": "bitmap", "filename": str(color), "raw": False, "filter_type": "trilinear", "wrap_mode": "repeat"})
    except Exception as exc:
        assert "trilinear" in str(exc)
    else:
        raise AssertionError("reason=trilinear_was_accepted")
    image = _render(
        [{"id": "panel", "mesh": package / "geometry" / "render.obj", "matrix": np.eye(4), "bsdf": _spec("unit_panel", package, {"base_color", "alpha_mitsuba", "normal_opengl"})}],
        *_panel_view(),
        tmp_path / "panel.png",
    )
    assert image["finite"] is True
    assert image["observed"]["shapes"][0]["vertex_texcoords"] is True
    assert image["observed"]["shapes"][0]["vertex_normals"] is True
    assert any("alpha.data" in key for key in image["observed"]["parameter_keys"])
    assert any("normalmap.data" in key for key in image["observed"]["parameter_keys"])
    assert image["halves"]["half_mean_abs"] > 20.0


def test_bitmap_plugin_omits_filter_notes(tmp_path: Path):
    texture = _texture(tmp_path / "color.png")
    plugin = bitmap_settings(texture, tmp_path / "color.png")
    assert plugin["raw"] is False
    assert plugin["filter_type"] == "bilinear"
    assert "filter_note" not in plugin
    assert "filter_requested" not in plugin
    assert "to_uv" not in plugin
    texture["uv_transform"] = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0]
    moved = bitmap_settings(texture, tmp_path / "color.png")
    assert "to_uv" in moved
