"""Mitsuba film renders with the arm as a diffuse occluder.

Training calls this once per demonstration frame and stores the images.
Evaluation calls it again at every 5 Hz step, because the action has moved
the arm and the object.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from sensor_sim.film.capture import _render
from sensor_sim.film.heldout import _film_aov, pack_measurement
from sensor_sim.film.sfp import SPP
from sensor_sim.film.sfp_math import release_render_caches
from sensor_sim.film.sfp_scene import filmed_object_dict
from sensor_sim.film.shape_mesh import MIN_FILM_FRACTION
from sensor_sim.policy.arm import BASE_POS, ArmEnv
from sensor_sim.policy.stokes_checkpoint import predict_stokes
from sensor_sim.policy.tasks import scripted_episode

logger = logging.getLogger("sensor_sim.policy")

ROOM = Path("outputs/film/sfp/02_heldout_shapes/room")
POLICY_IMAGE = 128
WORKSPACE_CENTER = np.array([0.0, 0.02, 0.05])


def pose_matrix(src_xy, src_yaw: float, dst_xy, dst_yaw: float, tilt: float, center_z: float) -> np.ndarray:
    """Map a catalog-placed mesh onto a demonstration pose.

    The OBJ vertices already include the catalog yaw and floor position.
    """
    def translation(x: float, y: float, z: float) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, 3] = (x, y, z)
        return matrix

    def rotation_z(angle: float) -> np.ndarray:
        cosine, sine = np.cos(angle), np.sin(angle)
        matrix = np.eye(4)
        matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1] = cosine, -sine, sine, cosine
        return matrix

    def rotation_x(angle: float) -> np.ndarray:
        cosine, sine = np.cos(angle), np.sin(angle)
        matrix = np.eye(4)
        matrix[1, 1], matrix[1, 2], matrix[2, 1], matrix[2, 2] = cosine, -sine, sine, cosine
        return matrix

    return (
        translation(float(dst_xy[0]), float(dst_xy[1]), 0.0)
        @ translation(0.0, 0.0, float(center_z))
        @ rotation_x(float(tilt))
        @ translation(0.0, 0.0, -float(center_z))
        @ rotation_z(float(dst_yaw) - float(src_yaw))
        @ translation(-float(src_xy[0]), -float(src_xy[1]), 0.0)
    )


def _apply(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.array([point[0], point[1], point[2], 1.0])
    return (matrix @ homogeneous)[:3]


def cube_transform(start: np.ndarray, end: np.ndarray, radius: float = 0.016) -> np.ndarray:
    """A Mitsuba cube is [-1, 1]^3. This matrix turns it into a box along a link."""
    direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    axis = direction / max(length, 1e-8)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    side = np.cross(helper, axis)
    side = side / np.linalg.norm(side)
    up = np.cross(axis, side)
    rotation = np.eye(4)
    rotation[:3, 0] = side
    rotation[:3, 1] = up
    rotation[:3, 2] = axis
    scale = np.diag([radius, radius, max(length, 1e-4) / 2.0, 1.0])
    transform = np.eye(4)
    transform[:3, 3] = 0.5 * (np.asarray(start) + np.asarray(end))
    return transform @ rotation @ scale


def scene_builder(mesh_dir: Path, matrix: np.ndarray, segments, camera: dict, probe: np.ndarray):
    """Closure consumed by the held-out AOV and Stokes renderers."""
    import mitsuba as mi

    from sensor_sim.film.heldout import _set_cuda_variant

    _set_cuda_variant()
    transform = mi.ScalarTransform4f(np.asarray(matrix, dtype=np.float64))

    def build(*, polarized: bool, spp: int, rfilter: str) -> dict:
        scene = filmed_object_dict(
            ROOM,
            mesh_dir,
            camera,
            resolution=(512, 512),
            spp=spp,
            max_depth=8,
            polarized=polarized,
            rfilter=rfilter,
        )
        scene["solid"]["to_world"] = transform
        scene["film-shell"]["to_world"] = transform
        for index, (start, end) in enumerate(segments):
            scene[f"arm-{index}"] = {
                "type": "cube",
                "to_world": mi.ScalarTransform4f(cube_transform(start, end)),
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.45}},
            }
        return scene

    return build, [float(value) for value in probe]


def render_measurement(
    mesh_dir: Path,
    matrix: np.ndarray,
    segments,
    camera: dict,
    probe: np.ndarray,
    fallback: np.ndarray | None = None,
) -> dict:
    """One 64-sample Stokes frame plus the 1-sample normal and shape index."""
    build, probe_point = scene_builder(mesh_dir, matrix, segments, camera, probe)
    try:
        aov = _film_aov(build, probe_point)
    except RuntimeError:
        if fallback is None:
            raise
        aov = _film_aov(build, [float(value) for value in fallback])
    from sensor_sim.film.heldout import _set_cuda_variant

    _set_cuda_variant()
    loaded, image, _array, elapsed = _render(build(polarized=True, spp=SPP, rfilter="gaussian"), SPP, 0)
    from sensor_sim.film.capture import _stokes_from_image
    from sensor_sim.film.sfp_math import view_directions

    bundle = _stokes_from_image(loaded, image)
    height, width = bundle["s0"].shape
    view = view_directions(loaded.sensors()[0], width, height)
    del loaded, image
    release_render_caches()
    measurement = pack_measurement(bundle, aov["normals"], aov["shape_index"], view)
    measurement["film_index"] = int(aov["film_index"])
    measurement["film_fraction"] = float(aov["fraction"])
    measurement["elapsed_s"] = float(elapsed)
    return measurement


def downsample(image: np.ndarray, side: int = POLICY_IMAGE) -> np.ndarray:
    """Area average from 512 to ``side``. ``side`` must divide 512."""
    image = np.asarray(image)
    factor = 512 // int(side)
    if 512 % int(side) != 0:
        raise ValueError(f"policy image side must divide 512, got {side}")
    if image.ndim == 2:
        cropped = image[: factor * side, : factor * side]
        return cropped.reshape(side, factor, side, factor).mean(axis=(1, 3))
    cropped = image[: factor * side, : factor * side]
    return cropped.reshape(side, factor, side, factor, image.shape[-1]).mean(axis=(1, 3))


def modality_images(measurement: dict, stokes_model, mean: np.ndarray, std: np.ndarray, device: str) -> dict:
    """RGB, Stokes normals, ground-truth normals, and the raw polarization image, at the policy resolution."""
    film = np.rint(measurement["shape_index"]).astype(np.int64) == int(measurement["film_index"])
    predicted = predict_stokes(stokes_model, measurement, mean, std, device)
    predicted = np.where(film[..., None], predicted, 0.0)
    truth = np.where(film[..., None], measurement["normal"], 0.0)
    cosine, sine = measurement["aolp_cos"], measurement["aolp_sin"]
    dolp = np.stack([(cosine + 1.0) * 0.5, (sine + 1.0) * 0.5, measurement["dolp"]], axis=-1)
    dolp = np.where(film[..., None], dolp, 0.0)
    return {
        "rgb": downsample(measurement["rgb"]).astype(np.float16),
        "normals": downsample(predicted).astype(np.float16),
        "gt_normals": downsample(truth).astype(np.float16),
        "dolp": downsample(dolp).astype(np.float16),
        "film_fraction": float(measurement["film_fraction"]),
    }


def film_fraction_at(
    mesh_dir: Path,
    matrix: np.ndarray,
    segments,
    camera: dict,
    probe: np.ndarray,
    fallback: np.ndarray | None = None,
) -> float:
    """1-sample film-pixel fraction. A probe outside the image retries at ``fallback``."""
    build, probe_point = scene_builder(mesh_dir, matrix, segments, camera, probe)
    try:
        aov = _film_aov(build, probe_point)
    except RuntimeError as exc:
        if fallback is None:
            logger.info("policy stage=camera_probe_missed reason=%s", exc)
            return 0.0
        build, probe_point = scene_builder(mesh_dir, matrix, segments, camera, fallback)
        try:
            aov = _film_aov(build, probe_point)
        except RuntimeError as second:
            logger.info("policy stage=camera_probe_missed reason=%s", second)
            return 0.0
    return float(aov["fraction"])


def _camera(origin: np.ndarray, target: np.ndarray, fov_deg: float = 70.0) -> dict:
    return {
        "position": [float(value) for value in origin],
        "target": [float(value) for value in target],
        "up": [0.0, 0.0, 1.0],
        "fov_deg": float(fov_deg),
    }


def place_fixed_camera(mesh_dir: Path, src_xy, src_yaw: float, radius: float, probe: np.ndarray) -> dict:
    """Pick a fixed camera that keeps the film visible at the start, mid-reach, and grasp.

    The arm base is on the -Y side. Candidates sit above and to the side of that reach.
    """
    env = ArmEnv()
    env.reset((0.0, 0.02), 0.3, 0.0, radius)
    poses = {
        "start": (env.link_segments(), env.object_pose()),
    }
    episode = scripted_episode("lift", "sphere", (0.0, 0.02), 0.3, radius)
    poses["mid"] = (episode["frames"][12]["segments"], episode["frames"][12]["object"])
    grasp_frame = next((frame for frame in episode["frames"] if frame["object"]["attached"]), episode["frames"][20])
    poses["grasp"] = (grasp_frame["segments"], grasp_frame["object"])
    target = WORKSPACE_CENTER
    best = None
    for distance in (0.22, 0.18, 0.28):
        for azimuth in (0.6, 1.1, 2.2, -0.4, 1.8):
            for elevation in (0.5, 0.9, 1.15):
                origin = target + distance * np.array(
                    [
                        np.cos(azimuth) * np.cos(elevation),
                        np.sin(azimuth) * np.cos(elevation),
                        np.sin(elevation),
                    ]
                )
                if origin[2] < 0.12 or origin[2] > 0.55:
                    continue
                if float(np.linalg.norm(origin[:2] - np.array(BASE_POS[:2]))) < 0.08:
                    continue
                camera = _camera(origin, target)
                fractions = {}
                for name, (segments, pose) in poses.items():
                    matrix = pose_matrix(src_xy, src_yaw, pose["position"][:2], pose["yaw"], pose["tilt"], radius)
                    moved_probe = _apply(matrix, np.asarray(probe, dtype=np.float64))
                    fractions[name] = film_fraction_at(mesh_dir, matrix, segments, camera, moved_probe)
                worst = min(fractions.values())
                logger.info(
                    "policy stage=camera_candidate distance=%.2f azimuth=%.2f elevation=%.2f start=%.3f mid=%.3f grasp=%.3f",
                    distance,
                    azimuth,
                    elevation,
                    fractions["start"],
                    fractions["mid"],
                    fractions["grasp"],
                )
                if worst >= MIN_FILM_FRACTION and (best is None or worst > best["worst"]):
                    best = {"camera": camera, "fractions": fractions, "worst": worst}
        if best is not None:
            break
    if best is None:
        raise RuntimeError("no fixed camera kept the film above 0.15 at the start, mid-reach, and grasp")
    logger.info(
        "policy stage=camera_selected position=%s target=%s fractions=%s",
        best["camera"]["position"],
        best["camera"]["target"],
        best["fractions"],
    )
    return best
