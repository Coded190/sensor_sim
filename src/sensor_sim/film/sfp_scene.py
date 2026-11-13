"""Two scenes for the polarization-to-normal check.

The standard scene is one opaque ``pplastic`` ball in a gray room. That is
the BSDF this repo already measured as polarized, and the walls do not add
a colored reflection. The Cornell scene keeps the filmed ball and turns
only the dragon into a smooth copper conductor.
"""

from __future__ import annotations

from pathlib import Path

import mitsuba as mi
import numpy as np

from sensor_sim.film import ROOT
from sensor_sim.film.config import scene_config
from sensor_sim.film.geometry import (
    CORNELL_HIGH,
    CORNELL_LOW,
    ball_center_offset,
    cap_shell,
    cornell_shell,
    platform_mesh,
    read_ascii_ply,
    reference_points,
    stand_dragon,
    uv_sphere,
    write_obj,
)
from sensor_sim.film.mitsuba_scene import scene_dict
from sensor_sim.scenes import material_bsdf

STANDARD_RADIUS = 0.05
STANDARD_FLOOR_GAP = 0.001
STANDARD_FOV = 55.0
STANDARD_LIGHT_RADIANCE = 18.0
PPLASTIC_ALPHA = 0.1
PPLASTIC_REFLECTANCE = 0.5
GRAY = 0.6
# Camera sits close enough that the 5 cm ball spans most of a 55° frame.
STANDARD_CAMERA_A = {
    "position": [0.0, -0.14, STANDARD_RADIUS + STANDARD_FLOOR_GAP],
    "target": [0.0, 0.0, STANDARD_RADIUS + STANDARD_FLOOR_GAP],
    "up": [0.0, 0.0, 1.0],
    "fov_deg": STANDARD_FOV,
}
STANDARD_CAMERA_B = {
    "position": [0.04, -0.15, STANDARD_RADIUS + STANDARD_FLOOR_GAP + 0.02],
    "target": [0.0, 0.0, STANDARD_RADIUS + STANDARD_FLOOR_GAP],
    "up": [0.0, 0.0, 1.0],
    "fov_deg": STANDARD_FOV,
}
CORNELL_CAMERA_B = [0.2, -0.55, 0.32]
DRAGON_DOLP_GATE = 0.2


def pplastic_bsdf() -> dict:
    """Opaque coated plastic. Interior IOR 1.5, exterior IOR 1.0, alpha 0.1."""
    bsdf = material_bsdf("pplastic")
    if bsdf["type"] != "pplastic":
        raise RuntimeError(f"expected pplastic, got {bsdf['type']}")
    if float(bsdf["alpha"]) != PPLASTIC_ALPHA or float(bsdf["int_ior"]) != 1.5:
        raise RuntimeError(f"pplastic parameters changed: {bsdf}")
    return bsdf


def copper_bsdf() -> dict:
    """Smooth copper. Mitsuba's spectral tables supply eta and k."""
    return {"type": "conductor", "material": "Cu"}


def ball_center() -> np.ndarray:
    return np.array([0.0, 0.0, STANDARD_RADIUS + STANDARD_FLOOR_GAP], dtype=np.float64)


def write_gray_room(mesh_dir: Path) -> None:
    """Open-front box. Every face is a separate OBJ so shape indices stay distinct."""
    mesh_dir = Path(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for name, mesh in cornell_shell().items():
        write_obj(mesh_dir / f"{name}.obj", mesh["vertices"], mesh["faces"], smooth=False)


def standard_scene_dict(
    mesh_dir: Path,
    camera: dict,
    *,
    resolution: tuple[int, int],
    spp: int,
    max_depth: int,
    polarized: bool,
    rfilter: str = "gaussian",
) -> dict:
    """Gray room, ceiling rectangle, and one ``pplastic`` sphere."""
    mesh_dir = Path(mesh_dir)
    width, height = int(resolution[0]), int(resolution[1])
    center = ball_center()
    integrator: dict = {"type": "path", "max_depth": int(max_depth)}
    if polarized:
        integrator = {"type": "stokes", "integrator": integrator}
    light_target = [0.0, 0.0, float(center[2])]
    shapes: dict = {
        "type": "scene",
        "integrator": integrator,
        "sensor": {
            "type": "perspective",
            "fov": float(camera["fov_deg"]),
            "to_world": mi.ScalarTransform4f.look_at(
                origin=list(camera["position"]),
                target=list(camera["target"]),
                up=list(camera["up"]),
            ),
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
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0.0, -0.02, float(CORNELL_HIGH[2]) - 0.002],
                target=light_target,
                up=[0.0, 1.0, 0.0],
            )
            @ mi.ScalarTransform4f.scale([0.075, 0.075, 1.0]),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": STANDARD_LIGHT_RADIANCE}},
        },
        "ball": {
            "type": "sphere",
            "center": [float(center[0]), float(center[1]), float(center[2])],
            "radius": STANDARD_RADIUS,
            "bsdf": pplastic_bsdf(),
        },
    }
    gray = {"type": "diffuse", "reflectance": {"type": "rgb", "value": GRAY}}
    for name in cornell_shell():
        shapes[name] = {
            "type": "obj",
            "filename": str(mesh_dir / f"{name}.obj"),
            "bsdf": gray,
        }
    return shapes


def cornell_stress_dict(poses: dict, mesh_dir: Path, **kwargs) -> dict:
    """The filmed Cornell scene with the dragon replaced by smooth copper."""
    scene = scene_dict(poses, mesh_dir, **kwargs)
    scene["dragon"]["bsdf"] = copper_bsdf()
    return scene


def standard_probe_points() -> dict[str, list[float]]:
    """The ball center. The close camera does not see the back wall."""
    center = ball_center()
    return {"ball": [float(center[0]), float(center[1]), float(center[2])]}


def filmed_object_dict(
    room_dir: Path,
    mesh_dir: Path,
    camera: dict,
    *,
    resolution: tuple[int, int],
    spp: int,
    max_depth: int,
    polarized: bool,
    rfilter: str = "gaussian",
    reflectance: float = 0.5,
    ior: float = 1.5,
) -> dict:
    """Gray room, the same unpolarized ceiling rectangle, one diffuse solid, and a dielectric shell."""
    if float(ior) != 1.5:
        raise RuntimeError(f"film IOR stays 1.5, got {ior}")
    room_dir = Path(room_dir)
    mesh_dir = Path(mesh_dir)
    width, height = int(resolution[0]), int(resolution[1])
    integrator: dict = {"type": "path", "max_depth": int(max_depth)}
    if polarized:
        integrator = {"type": "stokes", "integrator": integrator}
    # The lamp stays on the ceiling and aims at a fixed point. It does not turn toward each object.
    lamp_target = [0.0, 0.0, float(ball_center()[2])]
    shapes: dict = {
        "type": "scene",
        "integrator": integrator,
        "sensor": {
            "type": "perspective",
            "fov": float(camera["fov_deg"]),
            "to_world": mi.ScalarTransform4f.look_at(
                origin=list(camera["position"]),
                target=list(camera["target"]),
                up=list(camera["up"]),
            ),
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
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0.0, -0.02, float(CORNELL_HIGH[2]) - 0.002],
                target=lamp_target,
                up=[0.0, 1.0, 0.0],
            )
            @ mi.ScalarTransform4f.scale([0.075, 0.075, 1.0]),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": STANDARD_LIGHT_RADIANCE}},
        },
        "solid": {
            "type": "obj",
            "filename": str(mesh_dir / "solid.obj"),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": float(reflectance)}},
        },
        "film-shell": {
            "type": "obj",
            "filename": str(mesh_dir / "film.obj"),
            "bsdf": {"type": "dielectric", "int_ior": float(ior), "ext_ior": 1.0},
        },
    }
    gray = {"type": "diffuse", "reflectance": {"type": "rgb", "value": GRAY}}
    for name in cornell_shell():
        shapes[name] = {
            "type": "obj",
            "filename": str(room_dir / f"{name}.obj"),
            "bsdf": gray,
        }
    return shapes


def write_cornell_film(mesh_dir: Path) -> dict:
    """Cornell ball, film cap, pedestal, walls, and copper-ready dragon, without Blender.

    The cap uses 160 rings and 512 segments so its chord stays under the 20 µm
    shell. The dragon OBJ is the Stanford scan stood on the floor. The scene
    dict later replaces its BSDF with copper. These files are written under
    the experiment output, not into ``assets/``.
    """
    mesh_dir = Path(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    scene = scene_config()
    radius = float(scene["object"]["radius"])
    thickness = float(scene["film"]["thickness"])
    coverage = float(scene["film"]["coverage"])
    axis = tuple(float(value) for value in scene["film"]["axis"])
    gap = float(scene["room"]["floor_gap"])
    floor = float(CORNELL_LOW[2])
    offset = ball_center_offset(radius, 1.0, gap, floor_z=floor)
    ball = uv_sphere(radius - thickness, n_rings=128, n_phi=256)
    write_obj(mesh_dir / "ball.obj", ball["vertices"] + offset, ball["faces"], smooth=True)
    shell = cap_shell(radius, thickness, coverage, axis, n_rings=160, n_phi=512)
    write_obj(mesh_dir / "film_cap.obj", shell["vertices"] + offset, shell["faces"], smooth=True)
    pedestal = platform_mesh(floor)
    write_obj(mesh_dir / "platform.obj", pedestal["vertices"], pedestal["faces"], smooth=False)
    for name, mesh in cornell_shell().items():
        write_obj(mesh_dir / f"{name}.obj", mesh["vertices"], mesh["faces"], smooth=False)
    dragon_path = ROOT / "assets" / "dragon" / "dragon_vrip_res4.ply"
    dragon = read_ascii_ply(dragon_path)
    write_obj(mesh_dir / "dragon.obj", stand_dragon(dragon["vertices"]), dragon["faces"], smooth=True)
    refs = reference_points(radius, coverage, axis, floor_z=floor)
    return {
        "camera": {
            "position": [float(value) for value in scene["camera"]["position"]],
            "target": [float(value) for value in scene["camera"]["target"]],
            "up": [float(value) for value in scene["camera"]["up"]],
            "fov_deg": float(scene["camera"]["fov_deg"]),
        },
        "light": {
            "position": [float(value) for value in scene["lighting"]["position"]],
            "intensity": float(scene["lighting"]["intensity"]),
            "size": [float(value) for value in scene["lighting"]["size"]],
            "up": [float(value) for value in scene["lighting"]["up"]],
        },
        "ball_center": [float(value) for value in offset],
        "points": {"film": (refs["film"] + offset).tolist()},
        "film_ior": float(scene["film"]["ior"]),
        "film_roughness": float(scene["film"]["roughness"]),
        "reflectance": float(scene["object"]["reflectance"]),
        "room_reflectance": float(scene["room"]["reflectance"]),
    }
