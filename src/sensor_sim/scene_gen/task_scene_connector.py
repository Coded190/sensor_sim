"""Join a scene manifest to verified catalog records.

Isaac and Mitsuba consume this output. They do not look up packages or
hard-code poses. The scene-resolution hash is SHA-256 of the canonical JSON
payload with the hash field removed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_library import AssetLibraryError, catalog_file_hash, load_catalog, validate_catalog
from sensor_sim.scene_gen.asset_package import canonical_json_bytes
from sensor_sim.scene_gen.scene_manifest import (
    apply_transform,
    canonical_to_mitsuba,
    load_scene_manifest,
    pinhole_camera_to_mitsuba,
    validate_manifest,
    validate_rigid,
)

SERIALIZATION = "canonical_json_v1"
COORDINATE_CONTRACT = {
    "handedness": "right",
    "up_axis": "z",
    "units": "m",
    "matrix": "4x4_row_lists_column_vectors",
    "multiplication": "T_W_I = T_W_A @ T_A_I",
    "world_from_asset": "maps an asset-local point into world coordinates",
    "quaternion": "wxyz",
    "camera_frame": "x_right_y_down_z_forward",
    "pixel_origin": "top_left",
    "pixel_x": "right",
    "pixel_y": "down",
    "isaac_simulation_frame": "z_up_meters_when_stage_upAxis_is_Z_and_metersPerUnit_is_1",
    "isaac_camera_convention": "ros",
    "mitsuba_camera": "pinhole_to_mitsuba_+Z_forward_+Y_up via diag(-1,-1,1)",
}


class ConnectorError(ValueError):
    """The manifest and catalog do not resolve to one scene."""


def resolve_scene(manifest: dict, catalog: dict, library_root: Path, *, manifest_sha256: str, catalog_sha256: str) -> dict:
    """Return the backend-neutral scene. ``manifest`` is not modified."""
    library_root = Path(library_root)
    try:
        validate_manifest(manifest, library_root)
    except Exception as exc:
        raise ConnectorError(str(exc)) from exc
    try:
        records = validate_catalog(catalog, library_root)
    except AssetLibraryError as exc:
        raise ConnectorError(str(exc)) from exc
    by_identity = {(item["asset_id"], item["package_version"], item["package_hash"]): item for item in records}
    instances = []
    seen = set()
    for instance in manifest["instances"]:
        instance_id = instance["instance_id"]
        if instance_id in seen:
            raise ConnectorError(f"reason=duplicate_instance_id id={instance_id}")
        seen.add(instance_id)
        key = (instance["asset_id"], instance["asset_version"], instance["package_sha256"])
        record = by_identity.get(key)
        if record is None:
            raise ConnectorError(f"reason=unknown_asset asset_id={instance['asset_id']}")
        _check_geometry_order(record)
        world = validate_rigid(instance["world_from_asset"])
        instances.append(
            {
                "instance_id": instance_id,
                "asset_id": record["asset_id"],
                "package_version": record["package_version"],
                "package_hash": record["package_hash"],
                "render_geometry": record["render_geometry"],
                "render_geometry_hash": record["render_geometry_hash"],
                "collision_geometry": record["collision_geometry"],
                "collision_geometry_hash": record["collision_geometry_hash"],
                "render_path": str((library_root / record["asset_id"] / record["package_version"] / record["render_geometry"]).resolve()),
                "collision_path": str((library_root / record["asset_id"] / record["package_version"] / record["collision_geometry"]).resolve()),
                "material_sources": list(record["material_sources"]),
                "world_from_asset": _matrix_lists(world),
                "parent": instance.get("parent"),
                "support": instance.get("support"),
                "semantic_role": "support" if record["support_role"] else "movable",
                "support_polygon_m": record.get("support_polygon_m"),
                "qualification": {
                    "status": record["qualification_status"],
                    "collision": record["collision_qualification"],
                    "material_fidelity": record["material_fidelity"],
                    "redistribution_status": record["redistribution_status"],
                    "manipulation_qualified": False,
                },
                "provenance": {
                    "source_generator": record["source_generator"],
                    "source_factory": record["source_factory"],
                    "source_commit": record["source_commit"],
                    "source_seed": record["source_seed"],
                },
                "omit": bool(instance.get("omit", False)),
            }
        )
    camera = _camera(manifest["cameras"][0])
    payload = {
        "serialization": SERIALIZATION,
        "scene_id": manifest["scene_id"],
        "manifest_sha256": manifest_sha256,
        "catalog_sha256": catalog_sha256,
        "coordinate_contract": COORDINATE_CONTRACT,
        "task_id": manifest["task_id"],
        "task_objects": manifest["task_objects"],
        "camera": camera,
        "lights": manifest["lights"],
        "instances": instances,
    }
    payload["scene_resolution_sha256"] = scene_resolution_hash(payload)
    return payload


def resolve_scene_files(manifest_path: Path, catalog_path: Path, library_root: Path) -> dict:
    manifest_path = Path(manifest_path)
    catalog_path = Path(catalog_path)
    manifest = load_scene_manifest(manifest_path.read_text(encoding="utf-8"))
    catalog = load_catalog(catalog_path)
    return resolve_scene(
        manifest,
        catalog,
        library_root,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        catalog_sha256=catalog_file_hash(catalog_path),
    )


def scene_resolution_hash(payload: dict) -> str:
    """Hash the resolved scene. The hash field itself is not part of the preimage."""
    body = {key: value for key, value in payload.items() if key != "scene_resolution_sha256"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def canonical_to_isaac_sim(matrix: np.ndarray) -> np.ndarray:
    """Pose in an Isaac stage whose up axis is Z and whose metersPerUnit is 1.

    This is not ``canonical_to_usd``. That helper targets a Y-up USD basis.
    ``MeshConverter`` sets the stage to Z-up meters. Applying the Y-up helper
    on top of that stage would move the asset. Callers must still read the
    stage up axis and stop if it is not Z.
    """
    return np.array(validate_rigid(matrix), copy=True)


def pose_wxyz(matrix: np.ndarray) -> tuple[list[float], list[float]]:
    """Return translation and a ``wxyz`` quaternion for ``write_root_pose_to_sim``."""
    rigid = canonical_to_isaac_sim(matrix)
    return [float(value) for value in rigid[:3, 3]], matrix_to_quat_wxyz(rigid)


def matrix_to_quat_wxyz(matrix: np.ndarray) -> list[float]:
    rotation = validate_rigid(matrix)[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return [float(value) for value in quat]


def quat_wxyz_to_matrix(quaternion: list[float], translation: list[float] | None = None) -> np.ndarray:
    w, x, y, z = [float(value) for value in quaternion]
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-5:
        raise ConnectorError(f"reason=quaternion_norm value={norm}")
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    if translation is not None:
        matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return validate_rigid(matrix)


def project_pinhole(world_from_camera: np.ndarray, point: np.ndarray, camera: dict) -> tuple[float, float]:
    """Project a world point. The camera frame is +X right, +Y down, +Z forward."""
    local = apply_transform(np.linalg.inv(validate_rigid(world_from_camera)), np.asarray(point, dtype=np.float64))
    if local[2] <= 1e-9:
        raise ConnectorError("reason=point_behind_camera")
    u = float(camera["fx"]) * float(local[0]) / float(local[2]) + float(camera["cx"])
    v = float(camera["fy"]) * float(local[1]) / float(local[2]) + float(camera["cy"])
    return u, v


def project_mitsuba(world_from_camera: np.ndarray, point: np.ndarray, camera: dict) -> tuple[float, float]:
    """Project through the Mitsuba camera matrix using +Y up in that camera frame.

    ``pinhole_camera_to_mitsuba`` post-multiplies ``diag(-1, -1, 1)``. A pinhole
    camera point ``(x, y_down, z)`` becomes ``(-x, -y_down, z)`` in the Mitsuba
    camera frame. Image +v is still down, so the pixel uses the negated Mitsuba y.
    """
    mitsuba_world = pinhole_camera_to_mitsuba(world_from_camera)
    local = apply_transform(np.linalg.inv(mitsuba_world), np.asarray(point, dtype=np.float64))
    if local[2] <= 1e-9:
        raise ConnectorError("reason=point_behind_camera")
    u = float(camera["cx"]) - float(camera["fx"]) * float(local[0]) / float(local[2])
    v = float(camera["cy"]) - float(camera["fy"]) * float(local[1]) / float(local[2])
    return u, v


def isaac_camera_is_representable(camera: dict) -> bool:
    """Isaac Lab 2.3.0 accepts square pixels and a centered principal point."""
    width = float(camera["width"])
    height = float(camera["height"])
    square = abs(float(camera["fx"]) - float(camera["fy"])) <= 1e-6
    centered = abs(float(camera["cx"]) - width / 2.0) <= 0.5 and abs(float(camera["cy"]) - height / 2.0) <= 0.5
    return square and centered and camera["projection"] == "pinhole"


def _camera(camera: dict) -> dict:
    required = ("camera_id", "width", "height", "projection", "fx", "fy", "cx", "cy", "world_from_camera", "camera_frame")
    missing = [key for key in required if key not in camera]
    if missing:
        raise ConnectorError(f"reason=camera_field fields={missing}")
    if camera["projection"] != "pinhole" or camera["camera_frame"] != "x_right_y_down_z_forward":
        raise ConnectorError("reason=camera_model")
    pose = validate_rigid(camera["world_from_camera"])
    resolved = dict(camera)
    resolved["world_from_camera"] = _matrix_lists(pose)
    resolved["mitsuba_to_world"] = _matrix_lists(canonical_to_mitsuba(pinhole_camera_to_mitsuba(pose)))
    resolved["distortion"] = camera.get("distortion") or {"model": "none"}
    resolved["isaac_representable"] = isaac_camera_is_representable(camera)
    if not resolved["isaac_representable"]:
        raise ConnectorError("reason=isaac_camera_intrinsics_not_representable")
    return resolved


def _check_geometry_order(record: dict) -> None:
    render = record["render_geometry"]
    collision = record["collision_geometry"]
    if render == collision or record["render_geometry_hash"] == record["collision_geometry_hash"]:
        raise ConnectorError("reason=swapped_or_identical_geometry")
    if not render.endswith("render.obj") or not collision.endswith("collision.obj"):
        raise ConnectorError("reason=geometry_role_path")


def _matrix_lists(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix, dtype=np.float64)]
