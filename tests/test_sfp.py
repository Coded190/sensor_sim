"""Camera-frame normals, Fresnel branches, and the two scene materials."""

import numpy as np
import pytest

from sensor_sim.film.sfp_math import (
    aolp_from_pair,
    aolp_pair,
    angular_error_deg,
    diffuse_dolp,
    fresnel_normals,
    mean_angular_error_deg,
    normal_from_zenith_azimuth,
    object_pixel_mask,
    specular_dolp,
    view_directions,
    world_normals_to_camera,
    zenith_from_dolp,
)
from sensor_sim.film.sfp_scene import copper_bsdf, pplastic_bsdf
from sensor_sim.stokes import S0_MIN


def test_camera_normal_of_a_surface_facing_the_camera():
    # Columns are the camera axes in world, matching a look_at from -Y toward +Y with +Z up.
    camera_to_world = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.2],
            [0.0, 1.0, 0.0, 0.05],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    world = np.array([[[0.0, -1.0, 0.0]]])
    camera = world_normals_to_camera(world, camera_to_world)
    assert camera[0, 0] == pytest.approx([0.0, 0.0, -1.0])


def test_center_view_ray_points_along_camera_z():
    mitsuba = pytest.importorskip("mitsuba")
    mitsuba.set_variant("scalar_rgb")
    scene = mitsuba.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "path"},
            "sensor": {
                "type": "perspective",
                "fov": 55,
                "to_world": mitsuba.ScalarTransform4f.look_at(
                    origin=[0, -0.2, 0.05], target=[0, 0, 0.05], up=[0, 0, 1]
                ),
                "film": {"type": "hdrfilm", "width": 32, "height": 16, "rfilter": {"type": "box"}},
            },
        }
    )
    directions = view_directions(scene.sensors()[0], 32, 16)
    # Pixel centers sit half a pixel off NDC 0.5, so the four surrounding rays average to the axis.
    principal = directions[7:9, 15:17].mean(axis=(0, 1))
    principal = principal / np.linalg.norm(principal)
    assert principal == pytest.approx([0.0, 0.0, 1.0], abs=1e-5)


def test_aolp_pair_round_trips():
    angles = np.array([-1.2, -0.4, 0.0, 0.3, 1.5])
    cosine, sine = aolp_pair(angles, np.ones_like(angles))
    assert aolp_from_pair(cosine, sine) == pytest.approx(angles)
    masked_cosine, masked_sine = aolp_pair(angles, np.zeros_like(angles))
    assert aolp_from_pair(masked_cosine, masked_sine) == pytest.approx(0.0)


def test_identical_normals_have_zero_angular_error():
    normal = np.array([[0.0, 0.2, -0.9797959], [0.0, 0.0, -1.0]])
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    assert angular_error_deg(normal, normal) == pytest.approx(0.0, abs=1e-6)
    assert mean_angular_error_deg(normal, normal, np.array([True, False])) == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(mean_angular_error_deg(normal, normal, np.array([False, False])))


def test_fresnel_zero_zenith_faces_the_camera_and_branches_differ():
    assert diffuse_dolp(np.array(0.0)) == pytest.approx(0.0)
    assert specular_dolp(np.array(0.0)) == pytest.approx(0.0)
    assert diffuse_dolp(np.array(1.0)) > diffuse_dolp(np.array(0.4)) > 0.0
    view = np.zeros((1, 1, 3))
    view[..., 2] = 1.0
    head_on = normal_from_zenith_azimuth(view, np.zeros((1, 1)), np.zeros((1, 1)))
    assert head_on[0, 0] == pytest.approx([0.0, 0.0, -1.0])
    diffuse = fresnel_normals(np.ones((1, 1)), np.zeros((1, 1)), view, "diffuse")
    specular = fresnel_normals(np.ones((1, 1)), np.zeros((1, 1)), view, "specular")
    assert angular_error_deg(diffuse, specular)[0, 0] == pytest.approx(90.0, abs=1.0)
    recovered = zenith_from_dolp(diffuse_dolp(np.array([0.0, 0.6, 1.1])), "diffuse")
    assert recovered[0] == pytest.approx(0.0, abs=1e-3)
    assert recovered[1] == pytest.approx(0.6, abs=2e-2)


def test_object_mask_excludes_walls_and_dim_pixels():
    index = np.array([[2, 2, 5], [0, 7, 2]])
    s0 = np.array([[1.0, S0_MIN, 1.0], [1.0, 1.0, 1.0]])
    mask = object_pixel_mask(index, {2}, s0)
    assert mask.tolist() == [[True, False, False], [False, False, True]]


def test_scene_materials_are_pplastic_and_copper():
    plastic = pplastic_bsdf()
    assert plastic["type"] == "pplastic"
    assert plastic["int_ior"] == pytest.approx(1.5)
    assert plastic["ext_ior"] == pytest.approx(1.0)
    assert plastic["alpha"] == pytest.approx(0.1)
    assert plastic["diffuse_reflectance"] == pytest.approx(0.5)
    copper = copper_bsdf()
    assert copper == {"type": "conductor", "material": "Cu"}


def test_normal_unet_emits_three_channels():
    torch = pytest.importorskip("torch")
    from sensor_sim.film.unet import FilmUNet

    torch.manual_seed(0)
    model = FilmUNet(9, out_channels=3)
    output = model(torch.zeros(1, 9, 32, 32))
    assert output.shape == (1, 3, 32, 32)
