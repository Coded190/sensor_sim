"""Stokes definitions and one scalar polarized smoke render."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sensor_sim.render import render_job
from sensor_sim.stokes import AOLP_DOLP_MIN, S0_MIN, polarization_maps, summarize_polarization


def _pixel(s0, s1, s2, s3):
    shape = (1, 1, 3)
    return (
        np.full(shape, s0),
        np.full(shape, s1),
        np.full(shape, s2),
        np.full(shape, s3),
    )


def test_unpolarized_light_has_zero_polarization():
    maps = polarization_maps(*_pixel(1.0, 0.0, 0.0, 0.0))
    assert maps["dop"][0, 0] == pytest.approx(0.0)
    assert maps["dolp"][0, 0] == pytest.approx(0.0)
    assert bool(maps["aolp_masked"][0, 0])


def test_horizontal_linear_polarization():
    maps = polarization_maps(*_pixel(1.0, 1.0, 0.0, 0.0))
    assert maps["dolp"][0, 0] == pytest.approx(1.0)
    assert maps["dop"][0, 0] == pytest.approx(1.0)
    assert maps["aolp"][0, 0] == pytest.approx(0.0)
    assert not bool(maps["aolp_masked"][0, 0])


def test_diagonal_linear_polarization():
    maps = polarization_maps(*_pixel(1.0, 0.0, 1.0, 0.0))
    assert maps["dolp"][0, 0] == pytest.approx(1.0)
    assert maps["aolp"][0, 0] == pytest.approx(np.pi / 4.0)


def test_vertical_linear_polarization_is_plus_or_minus_half_pi():
    maps = polarization_maps(*_pixel(1.0, -1.0, 0.0, 0.0))
    # arctan2(0, -1) is pi, so half of that is pi/2, the closed end of (-pi/2, pi/2].
    assert maps["aolp"][0, 0] == pytest.approx(np.pi / 2.0)


def test_circular_polarization_has_no_linear_part():
    maps = polarization_maps(*_pixel(1.0, 0.0, 0.0, 1.0))
    assert maps["dolp"][0, 0] == pytest.approx(0.0)
    assert maps["dop"][0, 0] == pytest.approx(1.0)
    assert bool(maps["aolp_masked"][0, 0])


def test_dark_pixel_is_excluded_from_means():
    maps = polarization_maps(*_pixel(0.0, 0.0, 0.0, 0.0))
    summary = summarize_polarization(maps)
    assert summary["n_valid"] == 0
    assert summary["mean_dolp"] == 0.0
    assert not bool(maps["valid"][0, 0])
    assert S0_MIN == pytest.approx(1e-4)
    assert AOLP_DOLP_MIN == pytest.approx(0.05)


def test_summary_counts_only_pixels_above_s0_min():
    s0 = np.array([[1.0, S0_MIN / 10.0]])
    s1 = np.array([[0.5, 10.0]])
    s2 = np.zeros((1, 2))
    s3 = np.zeros((1, 2))
    summary = summarize_polarization(polarization_maps(s0, s1, s2, s3))
    assert summary["n_valid"] == 1
    assert summary["mean_dolp"] == pytest.approx(0.5)
    assert summary["mean_abs_s3"] == pytest.approx(0.0)


def test_cornell_box_smoke_render(tmp_path: Path):
    metrics = render_job(
        "cornell_box",
        "scalar_rgb",
        resolution=16,
        spp=1,
        seed=0,
        output_root=tmp_path,
    )
    assert metrics["mean_rgb"] > 0.0
    assert "S0.R" not in metrics["channel_names"]
    assert (tmp_path / "cornell_box_scalar_rgb" / "render.exr").is_file()


def test_cbox_pol_smoke_render(tmp_path: Path):
    metrics = render_job(
        "cbox_pol",
        "scalar_spectral_polarized",
        resolution=16,
        spp=1,
        seed=0,
        output_root=tmp_path,
    )
    assert "S0.R" in metrics["channel_names"]
    assert metrics["mean_abs_s0_minus_rgb"] < 1e-3
    assert metrics["max_dolp"] > 0.05


def test_dielectric_smoke_render(tmp_path: Path):
    metrics = render_job(
        "dielectric_sphere",
        "scalar_spectral_polarized",
        resolution=32,
        spp=4,
        seed=0,
        output_root=tmp_path,
    )
    assert metrics["channel_names"][:3] == ["R", "G", "B"]
    assert "S0.R" in metrics["channel_names"]
    assert "S3.B" in metrics["channel_names"]
    assert metrics["n_valid"] > 0
    assert np.isfinite(metrics["max_dop"])
    assert metrics["mean_abs_s0_minus_rgb"] < 1e-3
    # Mean DoLP is small because the walls are unpolarized. Max DoLP checks
    # that Fresnel reflection on the sphere polarized some pixels.
    assert metrics["max_dolp"] > 0.05

    written = json.loads((tmp_path / "dielectric_sphere_scalar_spectral_polarized" / "metrics.json").read_text())
    assert written["spp"] == 4
    assert (tmp_path / "dielectric_sphere_scalar_spectral_polarized" / "render.exr").is_file()
    assert (tmp_path / "dielectric_sphere_scalar_spectral_polarized" / "contact_sheet.png").is_file()


def test_phase2_materials_share_ior_and_alpha():
    from sensor_sim.scenes import (
        PHASE2_ALPHA,
        PHASE2_EXT_IOR,
        PHASE2_INT_IOR,
        PHASE2_MATERIALS,
        PHASE2_REFLECTANCE,
        material_bsdf,
        material_sphere_scene,
    )

    assert PHASE2_MATERIALS == (
        "dielectric",
        "thindielectric",
        "roughdielectric",
        "pplastic",
        "plastic",
        "diffuse",
    )
    for name in ("dielectric", "thindielectric", "roughdielectric", "pplastic"):
        bsdf = material_bsdf(name)
        assert bsdf["int_ior"] == PHASE2_INT_IOR == 1.5
        assert bsdf["ext_ior"] == PHASE2_EXT_IOR == 1.0
    assert material_bsdf("roughdielectric")["alpha"] == PHASE2_ALPHA == 0.1
    assert material_bsdf("roughdielectric")["distribution"] == "beckmann"
    assert material_bsdf("pplastic")["alpha"] == 0.1
    assert material_bsdf("pplastic")["diffuse_reflectance"] == PHASE2_REFLECTANCE == 0.5
    assert material_bsdf("plastic")["int_ior"] == 1.5
    assert material_bsdf("plastic")["diffuse_reflectance"] == 0.5
    assert "ext_ior" not in material_bsdf("plastic")
    assert material_bsdf("diffuse") == {"type": "diffuse", "reflectance": 0.5}
    scene = material_sphere_scene("dielectric")
    assert scene["dielectric-sphere"]["radius"] == 0.45
    assert list(scene["dielectric-sphere"]["center"])[1] > -1.0


def test_dielectric_max_dolp_beats_diffuse_control(tmp_path: Path):
    from sensor_sim.phase2 import render_material

    dielectric = render_material(
        "dielectric",
        output_root=tmp_path,
        variant="scalar_spectral_polarized",
        resolution=32,
        spp=8,
        seed=0,
    )
    diffuse = render_material(
        "diffuse",
        output_root=tmp_path,
        variant="scalar_spectral_polarized",
        resolution=32,
        spp=8,
        seed=0,
    )
    assert dielectric["status"] == "ok"
    assert diffuse["status"] == "ok"
    assert dielectric["max_dolp"] > 0.2
    assert dielectric["max_dolp"] > 5.0 * diffuse["max_dolp"]


def test_pplastic_renders_finite_stokes(tmp_path: Path):
    from sensor_sim.phase2 import render_material

    result = render_material(
        "pplastic",
        output_root=tmp_path,
        variant="scalar_spectral_polarized",
        resolution=32,
        spp=8,
        seed=0,
    )
    assert result["status"] == "ok"
    assert "S0.R" in result["channel_names"]
    assert "S3.B" in result["channel_names"]
    assert np.isfinite(result["max_dolp"])
    assert np.isfinite(result["mean_abs_s3_over_s0"])
    assert np.isfinite(result["mean_dolp"])


def test_unsupported_polarized_bsdf_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sensor_sim import phase2

    def boom(*_args, **_kwargs):
        raise RuntimeError("eval_pol not implemented for this BSDF")

    monkeypatch.setattr(phase2, "render_job", boom)
    result = phase2.render_material(
        "thindielectric",
        output_root=tmp_path,
        variant="scalar_spectral_polarized",
        resolution=8,
        spp=1,
        seed=0,
    )
    assert result["status"] == "unsupported"
    assert result["max_dolp"] is None
    written = json.loads(
        (tmp_path / "thindielectric_scalar_spectral_polarized" / "metrics.json").read_text()
    )
    assert written["status"] == "unsupported"
    assert "not implemented" in written["error"]
