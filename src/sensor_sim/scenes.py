"""Phase 0 scenes.

``cornell_box`` is Mitsuba's hello-world scene. ``cbox_pol`` is the
vendored polarized tutorial. The sphere scenes reuse the Cornell box
room, replace the two boxes with spheres, and wrap the path tracer in
the Stokes integrator. Wall and emitter colors are constant spectra so
the three spectral copies of each Stokes component match.
"""

from __future__ import annotations

from pathlib import Path

import mitsuba as mi

ROOT = Path(__file__).resolve().parents[2]
CBOX_POL_XML = ROOT / "scenes" / "cbox_pol.xml"

# Floor of mi.cornell_box() is the plane y = -1. A sphere rests on it
# when its center y equals -1 + radius, plus a small gap so the surface
# does not coincide with the floor.
FLOOR_Y = -1.0
FLOOR_GAP = 1e-3

SCENE_NAMES = (
    "cornell_box",
    "cbox_pol",
    "dielectric_sphere",
    "conductor_spheres",
)


def _stokes_path(max_depth: int = 8) -> dict:
    return {"type": "stokes", "integrator": {"type": "path", "max_depth": max_depth}}


# Gray-room lamp before Phase 3 scales it. The Cornell box default is a colored RGB.
LIGHT_RADIANCE = 18.0


def _gray_room() -> dict:
    """Cornell box room with neutral materials and no boxes."""
    scene = mi.cornell_box()
    scene["white"] = {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.8}}
    scene["green"] = {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.6}}
    scene["red"] = {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.4}}
    scene["light"]["emitter"]["radiance"] = {"type": "rgb", "value": LIGHT_RADIANCE}
    scene["integrator"] = _stokes_path()
    del scene["small-box"]
    del scene["large-box"]
    return scene


def _apply_light(scene: dict, shift_x: float, shift_z: float, radiance_scale: float) -> None:
    """Shift the ceiling rectangle in x and z and scale its RGB radiance."""
    scene["light"]["to_world"] = mi.ScalarTransform4f.translate([float(shift_x), 0.0, float(shift_z)]) @ scene["light"]["to_world"]
    scene["light"]["emitter"]["radiance"] = {"type": "rgb", "value": float(LIGHT_RADIANCE * radiance_scale)}


def _sphere(center: list[float], radius: float, bsdf: dict) -> dict:
    return {
        "type": "sphere",
        "center": center,
        "radius": radius,
        "bsdf": bsdf,
    }


def cornell_box_scene() -> dict:
    """Unpolarized RGB hello world. The caller sets the variant first."""
    return mi.cornell_box()


def cornell_box_stokes_scene() -> dict:
    """Cornell box with the path tracer wrapped in the Stokes integrator."""
    scene = mi.cornell_box()
    scene["integrator"] = {
        "type": "stokes",
        "integrator": scene["integrator"],
    }
    return scene


# Visible coating, not a micron film. Outer radius matches the Phase 0 sphere.
BALL_OUTER_RADIUS = 0.45
BALL_INNER_RADIUS = 0.40
BALL_FILM_THICKNESS = BALL_OUTER_RADIUS - BALL_INNER_RADIUS
# Reflectance of the inner ball at 400, 500, 600, and 700 nm.
BALL_REFLECTANCE_NM = (400.0, 500.0, 600.0, 700.0)
BALL_REFLECTANCE = (0.15, 0.25, 0.55, 0.85)


def ball_reflectance_spectrum() -> dict:
    """Diffuse reflectance that rises toward red, so cube bands are not copies.

    Knots are every 100 nm from 400 to 700. ``regular`` is used because the
    CUDA variants do not implement ``irregular`` spectrum evaluation.
    """
    return {
        "type": "regular",
        "wavelength_min": BALL_REFLECTANCE_NM[0],
        "wavelength_max": BALL_REFLECTANCE_NM[-1],
        "values": ", ".join(str(v) for v in BALL_REFLECTANCE),
    }


def ball_rest_center(x: float = 0.0, z: float = 0.0) -> list[float]:
    """Center of a radius-0.45 sphere resting on the floor."""
    return [float(x), FLOOR_Y + BALL_OUTER_RADIUS + FLOOR_GAP, float(z)]


def ball_film_scene(
    *,
    polarized: bool,
    center: list[float] | None = None,
    light_shift: tuple[float, float] = (0.0, 0.0),
    radiance_scale: float = 1.0,
    material: str = "dielectric",
) -> dict:
    """Ball in the gray room.

    ``material="dielectric"`` is the Phase 1 twin: a diffuse inner sphere
    inside a smooth dielectric shell of thickness ``BALL_FILM_THICKNESS``.
    ``material="pplastic"`` is one coated sphere and no shell. The caller
    sets the Mitsuba variant before this builds transforms.
    """
    if material not in ("dielectric", "pplastic"):
        raise ValueError(f"unknown ball material {material!r}")
    scene = _gray_room()
    if not polarized:
        scene["integrator"] = {"type": "path", "max_depth": 8}
    if center is None:
        center = ball_rest_center()
    else:
        center = [float(center[0]), float(center[1]), float(center[2])]
    _apply_light(scene, light_shift[0], light_shift[1], radiance_scale)
    if material == "dielectric":
        scene["film-shell"] = _sphere(
            center=center,
            radius=BALL_OUTER_RADIUS,
            bsdf=material_bsdf("dielectric"),
        )
        scene["inner-ball"] = _sphere(
            center=center,
            radius=BALL_INNER_RADIUS,
            bsdf={"type": "diffuse", "reflectance": ball_reflectance_spectrum()},
        )
    else:
        scene["pplastic-sphere"] = _sphere(
            center=center,
            radius=BALL_OUTER_RADIUS,
            bsdf=material_bsdf("pplastic"),
        )
    return scene


# Phase 2 compares these BSDFs on one sphere. They are not Phase 0 figure jobs.
PHASE2_MATERIALS = (
    "dielectric",
    "thindielectric",
    "roughdielectric",
    "pplastic",
    "plastic",
    "diffuse",
)
PHASE2_INT_IOR = 1.5
PHASE2_EXT_IOR = 1.0
PHASE2_ALPHA = 0.1
PHASE2_REFLECTANCE = 0.5
PHASE2_SPHERE_RADIUS = 0.45


def material_bsdf(name: str) -> dict:
    """BSDF dict for one Phase 2 sphere. Shared IOR and roughness where the plugin has them."""
    if name not in PHASE2_MATERIALS:
        raise ValueError(f"unknown material {name!r}; expected one of {PHASE2_MATERIALS}")
    if name == "dielectric":
        return {"type": "dielectric", "int_ior": PHASE2_INT_IOR, "ext_ior": PHASE2_EXT_IOR}
    if name == "thindielectric":
        return {"type": "thindielectric", "int_ior": PHASE2_INT_IOR, "ext_ior": PHASE2_EXT_IOR}
    if name == "roughdielectric":
        return {
            "type": "roughdielectric",
            "int_ior": PHASE2_INT_IOR,
            "ext_ior": PHASE2_EXT_IOR,
            "distribution": "beckmann",
            "alpha": PHASE2_ALPHA,
        }
    if name == "pplastic":
        return {
            "type": "pplastic",
            "int_ior": PHASE2_INT_IOR,
            "ext_ior": PHASE2_EXT_IOR,
            "diffuse_reflectance": PHASE2_REFLECTANCE,
            "alpha": PHASE2_ALPHA,
        }
    if name == "plastic":
        return {
            "type": "plastic",
            "int_ior": PHASE2_INT_IOR,
            "diffuse_reflectance": PHASE2_REFLECTANCE,
        }
    return {"type": "diffuse", "reflectance": PHASE2_REFLECTANCE}


def material_sphere_scene(name: str) -> dict:
    """Gray Cornell room and one sphere. Only the sphere BSDF changes."""
    scene = _gray_room()
    radius = PHASE2_SPHERE_RADIUS
    scene[f"{name}-sphere"] = _sphere(
        center=[0.0, FLOOR_Y + radius + FLOOR_GAP, 0.0],
        radius=radius,
        bsdf=material_bsdf(name),
    )
    return scene


def dielectric_sphere_scene() -> dict:
    """One smooth dielectric sphere, interior IOR 1.5, in a Cornell box.

    This matches the dielectric-reflection case in the polarization docs.
    The published ART comparison image is not available as a scene file.
    """
    return material_sphere_scene("dielectric")


def conductor_spheres_scene() -> dict:
    """Two smooth conductor spheres, eta = 0.052 and k = 3.905.

    The polarization docs use ``eta = 0.052 - 3.905 i`` and attribute
    circular polarization to two-bounce reflections between the spheres.
    A single specular bounce of unpolarized light does not produce ``S3``.
    """
    scene = _gray_room()
    radius = 0.38
    center_y = FLOOR_Y + radius + FLOOR_GAP
    bsdf = {
        "type": "conductor",
        "eta": {"type": "spectrum", "value": 0.052},
        "k": {"type": "spectrum", "value": 3.905},
    }
    scene["conductor-sphere-a"] = _sphere(
        center=[-0.38, center_y, 0.12],
        radius=radius,
        bsdf=bsdf,
    )
    scene["conductor-sphere-b"] = _sphere(
        center=[0.38, center_y, -0.12],
        radius=radius,
        bsdf=bsdf,
    )
    return scene


def set_film(scene: dict, resolution: int, spp: int) -> dict:
    """Override the Cornell box film size and sample count in a scene dict."""
    film = scene["sensor"]["film"]
    film["width"] = int(resolution)
    film["height"] = int(resolution)
    scene["sensor"]["sampler"]["sample_count"] = int(spp)
    return scene


def load_scene(name: str, resolution: int, spp: int):
    """Load a Phase 0 scene. The Mitsuba variant must already be set."""
    if name == "cornell_box":
        return mi.load_dict(set_film(cornell_box_scene(), resolution, spp))
    if name == "dielectric_sphere":
        return mi.load_dict(set_film(dielectric_sphere_scene(), resolution, spp))
    if name in PHASE2_MATERIALS:
        return mi.load_dict(set_film(material_sphere_scene(name), resolution, spp))
    if name == "conductor_spheres":
        return mi.load_dict(set_film(conductor_spheres_scene(), resolution, spp))
    if name == "cbox_pol":
        if not CBOX_POL_XML.is_file():
            raise FileNotFoundError(CBOX_POL_XML)
        return mi.load_file(str(CBOX_POL_XML), spp=int(spp), res=int(resolution))
    raise ValueError(f"unknown scene {name!r}; expected one of {SCENE_NAMES}")
