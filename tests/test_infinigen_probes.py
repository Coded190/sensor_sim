"""Pinned-variant probes for two material regions, normals, and light units."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sensor_sim.scene_gen.indoor_mitsuba import render_stokes, require_variant
from sensor_sim.scene_gen.light_units import point_intensity_from_watts
from sensor_sim.scene_gen.material_spec import make_spec, to_mitsuba


def _plane(path: Path, x0: float, x1: float) -> None:
    path.write_text(
        "\n".join([
            "v {0} -0.5 0".format(x0),
            "v {0} 0.5 0".format(x0),
            "v {0} 0.5 0".format(x1),
            "v {0} -0.5 0".format(x1),
            "vt 0 0",
            "vt 0 1",
            "vt 1 1",
            "vt 1 0",
            "vn 0 0 1",
            "f 1/1/1 2/2/1 3/3/1",
            "f 1/1/1 3/3/1 4/4/1",
            "",
        ]),
        encoding="utf-8",
    )


def _render(tmp: Path, name: str, shapes: dict, intensity: float) -> np.ndarray:
    require_variant()
    scene = {
        "type": "scene",
        "integrator": {"type": "stokes", "integrator": {"type": "path", "max_depth": 3}},
        "sensor": {
            "type": "perspective",
            "fov": 40,
            "to_world": __import__("mitsuba").ScalarTransform4f.look_at(origin=[0.0, -1.4, 1.2], target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0]),
            "sampler": {"type": "independent"},
            "film": {"type": "hdrfilm", "width": 32, "height": 16, "pixel_format": "rgb", "rfilter": {"type": "box"}},
        },
        "light": {"type": "point", "position": [-0.6, -0.8, 1.0], "intensity": {"type": "rgb", "value": intensity}},
    }
    scene.update(shapes)
    path = tmp / f"{name}.exr"
    render_stokes(scene, path, 8, 3)
    return np.array(__import__("mitsuba").Bitmap(str(path)))[:, :, :3]


def test_two_material_regions_are_both_visible(tmp_path: Path):
    left = tmp_path / "left.obj"
    right = tmp_path / "right.obj"
    _plane(left, -0.8, -0.05)
    _plane(right, 0.05, 0.8)
    red = to_mitsuba(make_spec("red", "diffuse", {"reflectance": [0.8, 0.05, 0.05]}), package_root=tmp_path)
    blue = to_mitsuba(make_spec("blue", "diffuse", {"reflectance": [0.05, 0.1, 0.8]}), package_root=tmp_path)
    image = _render(tmp_path, "split", {
        "left": {"type": "obj", "filename": str(left), "bsdf": red},
        "right": {"type": "obj", "filename": str(right), "bsdf": blue},
    }, 8.0)
    left_mean = image[:, :16, 0].mean()
    right_mean = image[:, 16:, 2].mean()
    assert left_mean > right_mean
    assert image[:, 16:, 2].mean() > image[:, :16, 2].mean()


def test_normal_map_changes_the_shaded_image(tmp_path: Path):
    from PIL import Image

    mesh = tmp_path / "plane.obj"
    _plane(mesh, -0.5, 0.5)
    normal = np.zeros((8, 8, 3), dtype=np.uint8)
    normal[..., 0] = 220
    normal[..., 1] = 128
    normal[..., 2] = 220
    Image.fromarray(normal, mode="RGB").save(tmp_path / "normal.png")
    flat = to_mitsuba(make_spec("flat", "diffuse", {"reflectance": [0.6, 0.6, 0.6]}), package_root=tmp_path)
    tilted_spec = make_spec("tilted", "diffuse", {"reflectance": [0.6, 0.6, 0.6]}, textures=[{
        "role": "normal_opengl",
        "path": "normal.png",
        "color_space": "non_color",
        "channel": "rgb",
        "resolution": [8, 8],
        "sha256": __import__("hashlib").sha256((tmp_path / "normal.png").read_bytes()).hexdigest(),
        "parameter_class": "authored_appearance",
        "uv_transform": [1, 0, 0, 0, 1, 0],
        "normal_strength": 1.0,
        "normal_orientation": "opengl_tangent",
        "filter_type": "bilinear",
        "wrap_mode": "repeat",
        "raw": True,
    }])
    tilted = to_mitsuba(tilted_spec, package_root=tmp_path)
    shape = {"type": "obj", "filename": str(mesh)}
    off = _render(tmp_path, "off", {"plane": {**shape, "bsdf": flat}}, 6.0)
    on = _render(tmp_path, "on", {"plane": {**shape, "bsdf": tilted}}, 6.0)
    assert float(np.mean(np.abs(on - off))) > 1e-4


def _normal_png(path: Path, red: int, green: int) -> None:
    from PIL import Image

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = red
    image[..., 1] = green
    image[..., 2] = 220
    Image.fromarray(image, mode="RGB").save(path)


def _normal_spec(root: Path, name: str) -> dict:
    import hashlib

    return to_mitsuba(make_spec(name, "diffuse", {"reflectance": [0.7, 0.7, 0.7]}, textures=[{
        "role": "normal_opengl",
        "path": f"{name}.png",
        "color_space": "non_color",
        "channel": "rgb",
        "resolution": [8, 8],
        "sha256": hashlib.sha256((root / f"{name}.png").read_bytes()).hexdigest(),
        "parameter_class": "authored_appearance",
        "uv_transform": [1, 0, 0, 0, 1, 0],
        "normal_strength": 1.0,
        "normal_orientation": "opengl_tangent",
        "filter_type": "bilinear",
        "wrap_mode": "repeat",
        "raw": True,
    }]), package_root=root)


def test_normal_map_tangent_matches_pos_x_and_pos_y(tmp_path: Path):
    """OpenGL tangent +X is red and bitangent +Y is green. Green is not flipped."""
    mesh = tmp_path / "plane.obj"
    _plane(mesh, -0.5, 0.5)
    _normal_png(tmp_path / "plus_x.png", 220, 128)
    _normal_png(tmp_path / "minus_x.png", 35, 128)
    _normal_png(tmp_path / "plus_y.png", 128, 220)
    _normal_png(tmp_path / "minus_y.png", 128, 35)
    shape = {"type": "obj", "filename": str(mesh)}
    plus_x = _render(tmp_path, "plus_x", {"plane": {**shape, "bsdf": _normal_spec(tmp_path, "plus_x")}}, 8.0)
    # The shared light sits at x=-0.6. A normal tilted toward -X faces that light.
    minus_x = _render(tmp_path, "minus_x", {"plane": {**shape, "bsdf": _normal_spec(tmp_path, "minus_x")}}, 8.0)
    assert float(minus_x.mean()) > float(plus_x.mean())
    plus_y = _normal_spec(tmp_path, "plus_y")
    minus_y = _normal_spec(tmp_path, "minus_y")
    # Move the comparison onto a +Y light by rendering both maps with the same +Y point light.
    require_variant()
    import mitsuba as mi

    def render_y(name: str, bsdf: dict) -> np.ndarray:
        scene = {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 3},
            "sensor": {
                "type": "perspective",
                "fov": 40,
                "to_world": mi.ScalarTransform4f.look_at(origin=[0.0, -1.4, 1.2], target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0]),
                "sampler": {"type": "independent"},
                "film": {"type": "hdrfilm", "width": 32, "height": 16, "pixel_format": "rgb", "rfilter": {"type": "box"}},
            },
            "light": {"type": "point", "position": [0.0, 0.9, 1.0], "intensity": {"type": "rgb", "value": 8.0}},
            "plane": {**shape, "bsdf": bsdf},
        }
        path = tmp_path / f"{name}.exr"
        mi.Bitmap(np.array(mi.render(mi.load_dict(scene), spp=16, seed=4))).write(str(path))
        return np.array(mi.Bitmap(str(path)))[:, :, :3]

    bright = render_y("y_plus", plus_y)
    dark = render_y("y_minus", minus_y)
    assert float(bright.mean()) > float(dark.mean())


def test_converted_point_intensity_scales_the_render(tmp_path: Path):
    mesh = tmp_path / "plane.obj"
    _plane(mesh, -0.4, 0.4)
    bsdf = to_mitsuba(make_spec("gray", "diffuse", {"reflectance": [0.7, 0.7, 0.7]}), package_root=tmp_path)
    shape = {"plane": {"type": "obj", "filename": str(mesh), "bsdf": bsdf}}
    power = 20.0
    low = _render(tmp_path, "low", shape, point_intensity_from_watts(power))
    high = _render(tmp_path, "high", shape, point_intensity_from_watts(2.0 * power))
    assert high[:, :, 0].mean() == pytest.approx(2.0 * low[:, :, 0].mean(), rel=0.25)
