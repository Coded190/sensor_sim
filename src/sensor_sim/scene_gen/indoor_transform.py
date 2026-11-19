"""Blender, asset-manifest, and Mitsuba transforms for indoor scenes.

Installed Mitsuba 3.9.1 was measured directly. A numpy 4x4 matrix passed to
``ScalarTransform4f`` uses the same layout as this module: row-major storage,
column-vector multiplication, translation in the last column, Z up, and meters.
The basis conversion is the identity. That is a measured result for this
install, not an assumption copied from newer documentation.

``M_world = M_parent @ M_local``. A unit or axis conversion is applied once,
here, and the measured conversion is identity.

Negative determinant is accepted. Mitsuba mirrors the positions and flips the
world normal. The report sets ``mirrored`` so the flip is not silent. Local
face order is not rewritten.

Shear is rejected. A parent non-uniform scale combined with a local rotation
that does not preserve orthogonal columns is shear and is not approximated.
Axis-aligned non-uniform scale, and rotation followed by that scale in the
local frame (``R @ S``), stay orthogonal and are accepted. Singular and
ill-conditioned matrices are rejected.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import PackageError

logger = logging.getLogger("sensor_sim.scene_gen.indoor_transform")

BASIS_CONVERSION = np.eye(4, dtype=np.float64)
COLUMN_ORTHOGONAL_TOLERANCE = 1e-6
DETERMINANT_MINIMUM = 1e-8
CONDITION_MAXIMUM = 1e8
TRANSFORM_CONVENTION = "row_major_column_vector"


def identity_matrix() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def serialize_matrix(matrix: np.ndarray) -> list[float]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4):
        raise PackageError("reason=transform_shape")
    return [float(value) for value in array.reshape(16)]


def deserialize_matrix(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(4, 4)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise PackageError("reason=transform_values")
    if abs(float(array[3, 0])) > 1e-12 or abs(float(array[3, 1])) > 1e-12 or abs(float(array[3, 2])) > 1e-12:
        raise PackageError("reason=transform_perspective")
    if abs(float(array[3, 3]) - 1.0) > 1e-8:
        raise PackageError("reason=transform_homogeneous")
    return array


def compose(parent: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Parent applied after local: ``M_world = M_parent @ M_local``."""
    return np.asarray(parent, dtype=np.float64) @ np.asarray(local, dtype=np.float64)


def translation_matrix(offset: list[float] | np.ndarray) -> np.ndarray:
    matrix = identity_matrix()
    matrix[:3, 3] = np.asarray(offset, dtype=np.float64)
    return matrix


def rotation_matrix(axis: str, radians: float) -> np.ndarray:
    cosine, sine = float(np.cos(radians)), float(np.sin(radians))
    matrix = identity_matrix()
    if axis == "x":
        matrix[:3, :3] = [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
    elif axis == "y":
        matrix[:3, :3] = [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    elif axis == "z":
        matrix[:3, :3] = [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    else:
        raise PackageError(f"reason=rotation_axis axis={axis}")
    return matrix


def scale_matrix(factors: list[float] | np.ndarray) -> np.ndarray:
    matrix = identity_matrix()
    matrix[:3, :3] = np.diag(np.asarray(factors, dtype=np.float64))
    return matrix


def apply_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate([array, np.ones((len(array), 1))], axis=1)
    return (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T[:, :3]


def apply_normals(matrix: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Inverse-transpose of the linear part, times the sign of its determinant.

    ``cross(A u, A v) = det(A) inv(A)^T cross(u, v)``. The determinant sign keeps
    a mirrored transform from reporting an unflipped normal.
    """
    linear = np.asarray(matrix, dtype=np.float64)[:3, :3]
    transformed = np.linalg.inv(linear).T @ np.asarray(normals, dtype=np.float64).T
    transformed *= np.sign(np.linalg.det(linear))
    lengths = np.linalg.norm(transformed, axis=0)
    return (transformed / lengths).T


def inspect_linear(matrix: np.ndarray) -> dict:
    """Accept an invertible shear-free transform. Reject the others."""
    array = deserialize_matrix(serialize_matrix(matrix))
    linear = array[:3, :3]
    determinant = float(np.linalg.det(linear))
    if abs(determinant) < DETERMINANT_MINIMUM:
        raise PackageError("reason=singular_transform")
    condition = float(np.linalg.cond(linear))
    if not np.isfinite(condition) or condition > CONDITION_MAXIMUM:
        raise PackageError(f"reason=ill_conditioned condition={condition}")
    columns = linear / np.linalg.norm(linear, axis=0)
    dots = columns.T @ columns
    shear = float(max(abs(dots[0, 1]), abs(dots[0, 2]), abs(dots[1, 2])))
    if shear > COLUMN_ORTHOGONAL_TOLERANCE:
        raise PackageError(f"reason=shear shear={shear}")
    return {
        "convention": TRANSFORM_CONVENTION,
        "basis_conversion": "identity",
        "matrix": serialize_matrix(array),
        "determinant": determinant,
        "mirrored": determinant < 0.0,
        "condition": condition,
        "shear": shear,
        "units": "m",
        "up_axis": "Z",
        "handedness": "right",
    }


def mitsuba_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the matrix Mitsuba should receive. The measured basis change is identity."""
    report = inspect_linear(matrix)
    converted = BASIS_CONVERSION @ deserialize_matrix(report["matrix"])
    return converted


def load_mitsuba_mesh(obj_path: Path, matrix: np.ndarray) -> dict:
    """Load one OBJ and return world positions and normals from the installed Mitsuba build."""
    import mitsuba as mi

    variant = "cuda_ad_spectral_polarized"
    if variant not in mi.variants():
        raise PackageError(f"reason=variant_missing variant={variant}")
    mi.set_variant(variant)
    world = mitsuba_matrix(matrix)
    scene = mi.load_dict(
        {
            "type": "scene",
            "mesh": {
                "type": "obj",
                "filename": str(obj_path),
                "to_world": mi.ScalarTransform4f(world),
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]}},
            },
        }
    )
    shapes = scene.shapes()
    if len(shapes) != 1:
        raise PackageError(f"reason=shape_count count={len(shapes)}")
    parameters = mi.traverse(shapes[0])
    positions = np.array(parameters["vertex_positions"], dtype=np.float64).reshape(-1, 3)
    normals = np.array(parameters["vertex_normals"], dtype=np.float64).reshape(-1, 3)
    logger.info(
        "indoor_transform event=load_mesh variant=%s vertices=%d mirrored=%s",
        variant,
        len(positions),
        inspect_linear(matrix)["mirrored"],
    )
    return {"positions": positions, "normals": normals, "variant": mi.variant()}


def load_mitsuba_camera(origin: list[float], target: list[float], up: list[float]) -> dict:
    """Load a perspective sensor and report its origin column and forward axis."""
    import mitsuba as mi

    variant = "cuda_ad_spectral_polarized"
    mi.set_variant(variant)
    scene = mi.load_dict(
        {
            "type": "scene",
            "sensor": {
                "type": "perspective",
                "fov": 45.0,
                "to_world": mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up),
                "film": {"type": "hdrfilm", "width": 4, "height": 4, "pixel_format": "rgb"},
            },
        }
    )
    sensor = scene.sensors()[0]
    matrix = np.array(sensor.world_transform().matrix, dtype=np.float64).reshape(4, 4)
    ray, _weight = sensor.sample_ray(mi.Float(0.0), mi.Float(0.5), mi.Point2f(0.5, 0.5), mi.Point2f(0.5, 0.5))
    direction = np.array(ray.d, dtype=np.float64).reshape(-1)[:3]
    direction = direction / np.linalg.norm(direction)
    return {
        "matrix": matrix,
        "origin": matrix[:3, 3].copy(),
        "forward": matrix[:3, 2].copy(),
        "camera_x": matrix[:3, 0].copy(),
        "camera_y": matrix[:3, 1].copy(),
        "center_direction": direction,
    }
