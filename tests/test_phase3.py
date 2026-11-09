"""Phase 3 parameter records and one tiny dielectric / pplastic pair."""

from __future__ import annotations

import mitsuba as mi
import numpy as np
import pytest

from sensor_sim.phase3 import (
    FILM_COUNT,
    FRAME_COUNT,
    PPLASTIC_COUNT,
    sample_frames,
    render_frame,
)
from sensor_sim.scenes import BALL_FILM_THICKNESS, FLOOR_Y, ball_film_scene


def test_sample_frames_repeat_and_split_24_8():
    first = sample_frames(seed=0)
    second = sample_frames(seed=0)
    assert first == second
    assert len(first) == FRAME_COUNT == 32
    assert sum(row["material"] == "dielectric" for row in first) == FILM_COUNT == 24
    assert sum(row["material"] == "pplastic" for row in first) == PPLASTIC_COUNT == 8
    assert [row["material"] for row in first[:24]] == ["dielectric"] * 24
    assert [row["material"] for row in first[24:]] == ["pplastic"] * 8
    for row in first:
        assert abs(row["center"][0]) <= 0.35
        assert abs(row["center"][2]) <= 0.35
        assert row["center"][1] == pytest.approx(FLOOR_Y + 0.45 + 1e-3)
        assert abs(row["light_shift"][0]) <= 0.25
        assert abs(row["light_shift"][1]) <= 0.25
        assert 0.5 <= row["radiance_scale"] <= 1.5
        if row["material"] == "dielectric":
            assert row["film_thickness"] == pytest.approx(BALL_FILM_THICKNESS) == pytest.approx(0.05)
            assert row["inner_radius"] == pytest.approx(0.40)
        else:
            assert row["film_thickness"] == pytest.approx(0.0)
            assert row["inner_radius"] is None


def test_dielectric_scene_has_two_spheres_and_pplastic_has_one():
    mi.set_variant("scalar_rgb")
    record = sample_frames(seed=0)[0]
    scene = ball_film_scene(
        polarized=True,
        center=record["center"],
        light_shift=tuple(record["light_shift"]),
        radiance_scale=record["radiance_scale"],
        material="dielectric",
    )
    assert scene["film-shell"]["radius"] - scene["inner-ball"]["radius"] == pytest.approx(0.05)
    assert scene["film-shell"]["center"] == scene["inner-ball"]["center"]
    assert scene["inner-ball"]["center"][1] - scene["inner-ball"]["radius"] > -1.0
    assert scene["film-shell"]["bsdf"]["int_ior"] == pytest.approx(1.5)

    opaque = sample_frames(seed=0)[24]
    pplastic = ball_film_scene(
        polarized=True,
        center=opaque["center"],
        light_shift=tuple(opaque["light_shift"]),
        radiance_scale=opaque["radiance_scale"],
        material="pplastic",
    )
    assert "film-shell" not in pplastic
    assert "inner-ball" not in pplastic
    assert pplastic["pplastic-sphere"]["radius"] == pytest.approx(0.45)
    assert pplastic["pplastic-sphere"]["bsdf"]["type"] == "pplastic"

    shifted = ball_film_scene(polarized=False, light_shift=(0.2, -0.1), radiance_scale=1.5)
    column = np.array(shifted["light"]["to_world"].matrix)
    assert column[0, 3] == pytest.approx(0.2)
    assert column[2, 3] == pytest.approx(0.01 - 0.1)
    assert shifted["light"]["emitter"]["radiance"]["value"] == pytest.approx(27.0)


def test_tiny_frames_have_stokes_cube_and_mask(tmp_path):
    film = sample_frames(seed=0)[0]
    opaque = sample_frames(seed=0)[24]
    film_row, _preview = render_frame(
        film,
        tmp_path / "film",
        resolution=16,
        stokes_spp=2,
        cube_spp=4,
        showcase=False,
    )
    opaque_row, _preview = render_frame(
        opaque,
        tmp_path / "opaque",
        resolution=16,
        stokes_spp=2,
        cube_spp=4,
        showcase=False,
    )
    assert "S0.R" in film_row["channel_names"]
    assert len(film_row["cube_channels"]) == 8
    assert np.isfinite(film_row["max_dolp"])
    assert film_row["n_film_pixels"] > 0
    assert opaque_row["n_film_pixels"] == 0
    cube = np.array(mi.Bitmap(str(tmp_path / "film" / "cube" / "render.exr")))
    assert cube.shape[2] == 8
    assert np.isfinite(cube).all()
    mask = plt_read(tmp_path / "film" / "mask.png")
    assert mask.max() > 0.5
    empty = plt_read(tmp_path / "opaque" / "mask.png")
    assert empty.max() == pytest.approx(0.0)
    assert (tmp_path / "film" / "stokes" / "render.exr").is_file()
    assert (tmp_path / "film" / "rgb.png").is_file()
    assert (tmp_path / "film" / "dolp.png").is_file()


def plt_read(path):
    import matplotlib.pyplot as plt

    image = plt.imread(path)
    if image.ndim == 3:
        image = image[:, :, 0]
    return image
