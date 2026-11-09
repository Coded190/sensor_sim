"""Phase 1 geometry, band order, and a small CUDA Cornell render."""

from __future__ import annotations

import mitsuba as mi
import numpy as np
import pytest

from sensor_sim.phase1 import render_cube, render_stokes
from sensor_sim.scenes import (
    BALL_FILM_THICKNESS,
    BALL_INNER_RADIUS,
    BALL_OUTER_RADIUS,
    ball_film_scene,
    ball_reflectance_spectrum,
    cornell_box_scene,
)
from sensor_sim.spectral import spectral_bands, specfilm_dict


def test_ball_shell_is_five_hundredths_thick():
    assert BALL_OUTER_RADIUS - BALL_INNER_RADIUS == pytest.approx(BALL_FILM_THICKNESS)
    assert BALL_FILM_THICKNESS == pytest.approx(0.05)
    mi.set_variant("scalar_rgb")
    scene = ball_film_scene(polarized=False)
    shell = scene["film-shell"]
    ball = scene["inner-ball"]
    assert shell["radius"] - ball["radius"] == pytest.approx(0.05)
    assert shell["center"] == ball["center"]
    assert shell["bsdf"]["type"] == "dielectric"
    assert shell["bsdf"]["int_ior"] == pytest.approx(1.5)
    # Inner bottom stays above the floor at y = -1.
    assert ball["center"][1] - ball["radius"] > -1.0


def test_eight_bands_cover_400_to_700_in_order():
    bands = spectral_bands()
    assert len(bands) == 8
    assert bands[0].wavelength_min == pytest.approx(400.0)
    assert bands[-1].wavelength_max == pytest.approx(700.0)
    assert [band.wavelength_max - band.wavelength_min for band in bands] == pytest.approx([37.5] * 8)
    names = [band.name for band in bands]
    assert names == sorted(names)
    film = specfilm_dict(16)
    assert film["type"] == "specfilm"
    assert set(names).issubset(film)


def test_red_rising_ball_makes_the_longest_band_brighter():
    """A diffuse square with the ball spectrum, through the same eight bands."""
    mi.set_variant("scalar_spectral")
    film = specfilm_dict(16)
    film["rfilter"] = {"type": "box"}
    scene = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 2},
        "sensor": {
            "type": "perspective",
            "fov": 40,
            "to_world": mi.ScalarTransform4f.look_at(origin=[0, 0, 2], target=[0, 0, 0], up=[0, 1, 0]),
            "sampler": {"type": "independent", "sample_count": 32},
            "film": film,
        },
        "light": {"type": "constant", "radiance": {"type": "uniform", "value": 1.0}},
        "square": {
            "type": "rectangle",
            "bsdf": {"type": "diffuse", "reflectance": ball_reflectance_spectrum()},
        },
    }
    image = np.array(mi.render(mi.load_dict(scene), spp=32, seed=0))
    assert image.shape[2] == 8
    assert np.isfinite(image).all()
    means = image.mean(axis=(0, 1))
    assert means[-1] > means[0]


def test_cuda_cornell_stokes_and_cube(tmp_path):
    stokes = render_stokes(
        "cornell_stokes_smoke",
        cornell_box_scene_stokes,
        tmp_path / "stokes",
        resolution=16,
        spp=2,
        seed=0,
    )
    assert "S0.R" in stokes["channel_names"]
    assert stokes["variant"] == "cuda_ad_spectral_polarized"
    assert np.isfinite(stokes["max_dolp"])

    cube = render_cube(
        "cornell_cube_smoke",
        cornell_box_scene,
        tmp_path / "cube",
        tmp_path / "spectral_profile.png",
        {"frame": None},
        resolution=16,
        spp=4,
        seed=0,
    )
    assert len(cube["channel_names"]) == 8
    assert len(cube["mean_radiance"]["frame"]) == 8
    assert all(np.isfinite(value) for value in cube["mean_radiance"]["frame"])
    assert (tmp_path / "spectral_profile.png").is_file()
    assert (tmp_path / "cube" / "render.exr").is_file()


def cornell_box_scene_stokes():
    from sensor_sim.scenes import cornell_box_stokes_scene

    return cornell_box_stokes_scene()
