"""Texture records, roughness conversion, and Mitsuba bitmap mapping."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sensor_sim.scene_gen.material_spec import (
    MaterialSpecError,
    alpha_to_blender_roughness,
    blender_roughness_to_alpha,
    make_spec,
    preset,
    roughness_image_to_alpha,
    to_mitsuba,
)


def _png(path: Path, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (4, 4), color).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _texture(path: Path, role: str, color_space: str, digest: str) -> dict:
    return {
        "role": role,
        "path": path.name,
        "color_space": color_space,
        "resolution": [4, 4],
        "sha256": digest,
        "parameter_class": "authored_appearance",
        "channel": "r" if "rough" in role or role.startswith("alpha") else "rgb",
        "normal_orientation": "opengl_tangent",
    }


def test_roughness_conversion_is_once_and_rejects_bad_values():
    assert blender_roughness_to_alpha(0.0) == 0.0
    assert blender_roughness_to_alpha(0.5) == pytest.approx(0.25)
    assert blender_roughness_to_alpha(1.0) == 1.0
    assert alpha_to_blender_roughness(0.25) == pytest.approx(0.5)
    image = np.array([[0.0, 0.5], [1.0, 0.2]], dtype=np.float64)
    converted = roughness_image_to_alpha(image)
    assert converted[0, 1] == pytest.approx(0.25)
    assert converted[1, 0] == pytest.approx(1.0)
    with pytest.raises(MaterialSpecError, match="roughness_out_of_range"):
        blender_roughness_to_alpha(1.1)
    with pytest.raises(MaterialSpecError, match="roughness_image_out_of_range"):
        roughness_image_to_alpha(np.array([np.nan]))


def test_old_preset_still_maps_without_textures():
    mapped = to_mitsuba(preset("plastic_smooth"))
    assert mapped["type"] == "pplastic"
    assert mapped["alpha"] == pytest.approx(0.04)
    assert "bitmap" not in str(mapped["diffuse_reflectance"])


def test_missing_texture_fails_and_base_color_is_not_raw(tmp_path: Path):
    digest = _png(tmp_path / "color.png", (180, 40, 30))
    spec = make_spec(
        "apple",
        "pplastic",
        {"diffuse_reflectance": [0.5, 0.1, 0.1], "int_ior": 1.5, "ext_ior": 1.0, "alpha": 0.04, "distribution": "beckmann"},
        textures=[_texture(tmp_path / "color.png", "base_color", "srgb", digest)],
    )
    mapped = to_mitsuba(spec, package_root=tmp_path)
    assert mapped["diffuse_reflectance"]["type"] == "bitmap"
    assert mapped["diffuse_reflectance"]["raw"] is False
    assert mapped["diffuse_reflectance"]["filter_type"] == "bilinear"
    spec["textures"][0]["sha256"] = "0" * 64
    with pytest.raises(MaterialSpecError, match="texture_hash"):
        to_mitsuba(make_spec("apple2", "pplastic", spec["parameters"], textures=spec["textures"]), package_root=tmp_path)


def test_conductor_rejects_a_base_color_texture(tmp_path: Path):
    digest = _png(tmp_path / "paint.png", (10, 10, 10))
    with pytest.raises(MaterialSpecError, match="base_color_unsupported"):
        to_mitsuba(
            make_spec("can", "roughconductor", {"material": "Al", "alpha": 0.1, "distribution": "beckmann"}, textures=[_texture(tmp_path / "paint.png", "base_color", "srgb", digest)]),
            package_root=tmp_path,
        )


def test_absolute_texture_path_is_rejected(tmp_path: Path):
    digest = _png(tmp_path / "color.png", (1, 2, 3))
    record = _texture(tmp_path / "color.png", "base_color", "srgb", digest)
    record["path"] = str(tmp_path / "color.png")
    with pytest.raises(MaterialSpecError, match="texture_path"):
        make_spec("apple", "diffuse", {"reflectance": [0.2, 0.2, 0.2]}, textures=[record])
