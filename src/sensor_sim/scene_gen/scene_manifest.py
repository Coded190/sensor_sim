"""Canonical scene_manifest_v1.

World and asset frames are right-handed, Z-up, and in meters. A matrix is four
rows of four numbers and acts on column vectors. ``T_A_B`` maps coordinates
expressed in frame B into frame A. Composition is ``T_W_I = T_W_A @ T_A_I``.
The last row of a rigid transform is ``[0, 0, 0, 1]``.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import PackageError, canonical_json_bytes, normalize_payload_path, parse_strict_json

SCHEMA_VERSION = "scene_manifest_v1"
_ROTATION_TOLERANCE = 1e-6
_IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
# Canonical +Z up, +Y forward maps to USD +Y up, -Z forward.
_CANONICAL_TO_USD = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
# Infinigen OBJ export is Y-up. (x, y, z) -> (x, -z, y), matching world_from_exported_vertex.
_EXPORTED_Y_UP_TO_CANONICAL = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
# Pinhole frame is +X right, +Y down, +Z forward. This Mitsuba 3.9.1 build looks along +Z
# with +Y up. The map is a 180 degree roll about the forward axis: negate camera X and Y.
_PINHOLE_TO_MITSUBA_CAMERA = np.array(
    [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)


class SceneManifestError(ValueError):
    """A canonical scene failed validation."""


def load_scene_manifest(text: str) -> dict:
    try:
        parsed = parse_strict_json(text)
    except PackageError as exc:
        raise SceneManifestError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise SceneManifestError("reason=manifest_not_object")
    return parsed


def manifest_hash(document: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def validate_manifest(document: dict, root: Path | None = None) -> None:
    keys = {
        "schema_version",
        "scene_id",
        "task_id",
        "configuration_version",
        "assets",
        "instances",
        "cameras",
        "lights",
        "units",
        "handedness",
        "up_axis",
        "seeds",
        "task_objects",
    }
    if not isinstance(document, dict) or not keys <= set(document):
        missing = sorted(keys - set(document if isinstance(document, dict) else []))
        raise SceneManifestError(f"reason=missing_key keys={missing}")
    if document["schema_version"] != SCHEMA_VERSION:
        raise SceneManifestError("reason=schema_version")
    if document["units"] != "m" or document["handedness"] != "right" or document["up_axis"] != "z":
        raise SceneManifestError("reason=world_convention")
    ids = []
    for instance in document["instances"]:
        for key in ("instance_id", "asset_id", "asset_version", "package_sha256", "package_path", "world_from_asset"):
            if key not in instance:
                raise SceneManifestError(f"reason=instance_field field={key}")
        validate_rigid(instance["world_from_asset"])
        if root is not None:
            resolve_under(root, instance["package_path"])
        ids.append(instance["instance_id"])
    if len(ids) != len(set(ids)):
        raise SceneManifestError("reason=duplicate_instance_id")
    for camera in document["cameras"]:
        for key in ("camera_id", "width", "height", "projection", "fx", "fy", "cx", "cy", "world_from_camera", "camera_frame"):
            if key not in camera:
                raise SceneManifestError(f"reason=camera_field field={key}")
        if camera["projection"] != "pinhole" or camera["camera_frame"] != "x_right_y_down_z_forward":
            raise SceneManifestError("reason=camera_model")
        validate_rigid(camera["world_from_camera"])
        for key in ("fx", "fy", "cx", "cy"):
            _positive(camera[key], key)
    for light in document["lights"]:
        for key in ("light_id", "type", "radiometry", "calibration"):
            if key not in light:
                raise SceneManifestError(f"reason=light_field field={key}")
        mode = light["radiometry"].get("mode")
        if mode not in {"relative", "calibrated"}:
            raise SceneManifestError("reason=light_radiometry")


def validate_rigid(matrix: object) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4) or not np.all(np.isfinite(array)):
        raise SceneManifestError("reason=transform_shape")
    if not np.allclose(array[3], [0.0, 0.0, 0.0, 1.0], atol=_ROTATION_TOLERANCE):
        raise SceneManifestError("reason=transform_row")
    rotation = array[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise SceneManifestError("reason=transform_not_orthonormal")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 1e-5:
        raise SceneManifestError(f"reason=transform_reflection det={determinant}")
    scales = np.linalg.norm(rotation, axis=0)
    if not np.allclose(scales, 1.0, atol=1e-5):
        raise SceneManifestError("reason=transform_scale")
    return array


def apply_transform(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    return (np.asarray(matrix, dtype=np.float64) @ homogeneous)[:3]


def compose_transforms(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    """Return ``T_W_I = T_W_A @ T_A_I``."""
    return np.asarray(parent, dtype=np.float64) @ np.asarray(child, dtype=np.float64)


def canonical_to_mitsuba(matrix: np.ndarray) -> np.ndarray:
    """Mitsuba world is the same right-handed Z-up meter frame."""
    return np.array(validate_rigid(matrix), copy=True)


def canonical_to_usd(matrix: np.ndarray) -> np.ndarray:
    """USD default is right-handed Y-up. +Z canonical becomes +Y USD."""
    return _CANONICAL_TO_USD @ validate_rigid(matrix)


def usd_to_canonical(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_CANONICAL_TO_USD) @ np.asarray(matrix, dtype=np.float64)


def blender_world_to_canonical(matrix: np.ndarray) -> np.ndarray:
    """Blender's own world is already Z-up and right-handed."""
    return np.array(validate_rigid(matrix), copy=True)


def exported_y_up_to_canonical(matrix: np.ndarray) -> np.ndarray:
    """Convert an Infinigen OBJ transform. Exported Y is canonical Z."""
    return _EXPORTED_Y_UP_TO_CANONICAL @ np.asarray(matrix, dtype=np.float64)


def pinhole_camera_to_mitsuba(world_from_camera: np.ndarray) -> np.ndarray:
    """Map the +Y-down, +Z-forward pinhole onto Mitsuba 3.9.1's +Y-up, +Z-forward camera."""
    return validate_rigid(world_from_camera) @ _PINHOLE_TO_MITSUBA_CAMERA


def translation_matrix(x: float, y: float, z: float) -> list[list[float]]:
    matrix = [list(row) for row in _IDENTITY]
    matrix[0][3] = float(x)
    matrix[1][3] = float(y)
    matrix[2][3] = float(z)
    return matrix


def look_at_y_down(origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera frame +X right, +Y down, +Z forward, looking from ``origin`` toward ``target``."""
    forward = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4)
    matrix[:3, 0] = right
    matrix[:3, 1] = down
    matrix[:3, 2] = forward
    matrix[:3, 3] = np.asarray(origin, dtype=np.float64)
    return validate_rigid(matrix)


def yaw_matrix(yaw_rad: float) -> np.ndarray:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return np.array(
        [[cosine, -sine, 0.0, 0.0], [sine, cosine, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def resolve_under(root: Path, relative: str) -> Path:
    try:
        relative = normalize_payload_path(relative)
    except PackageError as exc:
        raise SceneManifestError(str(exc)) from exc
    base = Path(root).resolve()
    target = (base / relative).resolve()
    if base != target and base not in target.parents:
        raise SceneManifestError(f"reason=path_escape path={relative}")
    return target


def matrix_lists(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix, dtype=np.float64)]


def _positive(value: object, place: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
        raise SceneManifestError(f"reason=camera_parameter field={place}")
