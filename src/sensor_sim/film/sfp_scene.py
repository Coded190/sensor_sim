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

from sensor_sim.film.geometry import CORNELL_HIGH, cornell_shell, write_obj
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
