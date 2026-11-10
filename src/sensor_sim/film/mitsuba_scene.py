"""Mitsuba dict for one exported Blender scene.

Blender does not assign the polarized BSDFs. The ball is diffuse with a
constant reflectance. The film is a smooth dielectric. Roughness other
than 0 is rejected rather than switched to roughdielectric.
"""

from __future__ import annotations

from pathlib import Path

import mitsuba as mi


def _look(origin, target, up):
    return mi.ScalarTransform4f.look_at(
        origin=list(origin),
        target=list(target),
        up=list(up),
    )


def scene_dict(
    poses: dict,
    mesh_dir: Path,
    *,
    polarized: bool,
    resolution: tuple[int, int],
    spp: int,
    max_depth: int,
    reflectance: float,
    room_reflectance: float,
    ior: float,
    roughness: float,
    rfilter: str = "gaussian",
) -> dict:
    """Build the dict. The caller sets the Mitsuba variant first."""
    if float(roughness) != 0.0:
        raise ValueError(f"film roughness must stay 0 for smooth dielectric, got {roughness}")
    mesh_dir = Path(mesh_dir)
    width, height = int(resolution[0]), int(resolution[1])
    camera = poses["camera"]
    light = poses["light"]
    size_x, size_y = float(light["size"][0]), float(light["size"][1])
    integrator: dict = {"type": "path", "max_depth": int(max_depth)}
    if polarized:
        integrator = {"type": "stokes", "integrator": integrator}
    return {
        "type": "scene",
        "integrator": integrator,
        "sensor": {
            "type": "perspective",
            "fov": float(camera["fov_deg"]),
            "to_world": _look(camera["position"], camera["target"], camera["up"]),
            "sampler": {"type": "independent", "sample_count": int(spp)},
            "film": {
                "type": "hdrfilm",
                "width": width,
                "height": height,
                "pixel_format": "rgb",
                "rfilter": {"type": rfilter},
            },
        },
        "light": {
            "type": "rectangle",
            "to_world": _look(light["position"], camera["target"], camera["up"])
            @ mi.ScalarTransform4f.scale([size_x / 2.0, size_y / 2.0, 1.0]),
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": float(light["intensity"])},
            },
        },
        "ball": {
            "type": "obj",
            "filename": str(mesh_dir / "ball.obj"),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": float(reflectance)}},
        },
        "film-shell": {
            "type": "obj",
            "filename": str(mesh_dir / "film_cap.obj"),
            "bsdf": {"type": "dielectric", "int_ior": float(ior), "ext_ior": 1.0},
        },
        "room": {
            "type": "obj",
            "filename": str(mesh_dir / "room.obj"),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": float(room_reflectance)}},
        },
    }


def without_film(scene: dict) -> dict:
    """Same scene with the film mesh removed. Used as the unfilmed control."""
    copy = dict(scene)
    copy.pop("film-shell", None)
    return copy
