"""``scene_state_snapshot_v1`` records for one offline Isaac trajectory.

The content hash is SHA-256 of the canonical JSON payload with ``content_hash``
removed. The sequence hash is SHA-256 of the exact JSONL file bytes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import canonical_json_bytes, parse_strict_json
from sensor_sim.scene_gen.scene_manifest import validate_rigid
from sensor_sim.scene_gen.task_scene_connector import quat_wxyz_to_matrix

SCHEMA_VERSION = "scene_state_snapshot_v1"
SERIALIZATION = "canonical_json_v1"


class SceneStateError(ValueError):
    """A snapshot or a state stream is not usable."""


def content_hash(snapshot: dict) -> str:
    body = {key: value for key, value in snapshot.items() if key != "content_hash"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def stamp(snapshot: dict) -> dict:
    """Return a copy that includes a content hash of the other fields."""
    body = {key: value for key, value in snapshot.items() if key != "content_hash"}
    body["schema_version"] = SCHEMA_VERSION
    body["serialization"] = SERIALIZATION
    stamped = dict(body)
    stamped["content_hash"] = content_hash(body)
    return stamped


def validate_snapshot(snapshot: dict, *, expected_manifest: str | None = None, expected_catalog: str | None = None, expected_ids: set[str] | None = None) -> None:
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise SceneStateError("reason=schema_version")
    required = {
        "snapshot_id",
        "scene_id",
        "manifest_sha256",
        "catalog_sha256",
        "scene_resolution_sha256",
        "frame_index",
        "physics_step",
        "time_s",
        "timestep_s",
        "backend",
        "instances",
        "coordinate_convention",
        "content_hash",
    }
    missing = sorted(required - set(snapshot))
    if missing:
        raise SceneStateError(f"reason=missing_field fields={missing}")
    if snapshot["content_hash"] != content_hash(snapshot):
        raise SceneStateError("reason=content_hash_mismatch")
    if expected_manifest is not None and snapshot["manifest_sha256"] != expected_manifest:
        raise SceneStateError("reason=manifest_hash_mismatch")
    if expected_catalog is not None and snapshot["catalog_sha256"] != expected_catalog:
        raise SceneStateError("reason=catalog_hash_mismatch")
    convention = snapshot["coordinate_convention"]
    if convention.get("up") != "z" or convention.get("units") != "m" or convention.get("quaternion") != "wxyz":
        raise SceneStateError("reason=coordinate_convention")
    ids = []
    dynamic = 0
    for instance in snapshot["instances"]:
        for key in ("instance_id", "asset_id", "package_sha256", "world_from_asset", "linear_velocity_m_s", "angular_velocity_rad_s"):
            if key not in instance:
                raise SceneStateError(f"reason=instance_field field={key}")
        if instance["instance_id"] in ids:
            raise SceneStateError(f"reason=duplicate_instance_id id={instance['instance_id']}")
        ids.append(instance["instance_id"])
        validate_rigid(instance["world_from_asset"])
        _finite_vec(instance["linear_velocity_m_s"], 3, "linear_velocity")
        _finite_vec(instance["angular_velocity_rad_s"], 3, "angular_velocity")
        if instance.get("dynamic"):
            dynamic += 1
    if dynamic < 1:
        raise SceneStateError("reason=missing_dynamic_instance")
    if expected_ids is not None and set(ids) != set(expected_ids):
        raise SceneStateError(f"reason=instance_set got={ids} expected={sorted(expected_ids)}")
    for key in ("name", "isaaclab", "isaacsim", "physx"):
        if key not in snapshot["backend"]:
            raise SceneStateError(f"reason=backend_field field={key}")


def validate_sequence(snapshots: list[dict], **kwargs) -> None:
    if len(snapshots) < 3:
        raise SceneStateError("reason=sequence_too_short")
    previous_time = -1.0
    previous_frame = -1
    for snapshot in snapshots:
        validate_snapshot(snapshot, **kwargs)
        time_s = float(snapshot["time_s"])
        frame = int(snapshot["frame_index"])
        if time_s < previous_time or frame < previous_frame:
            raise SceneStateError("reason=time_not_monotonic")
        previous_time = time_s
        previous_frame = frame


def write_snapshot(path: Path, snapshot: dict) -> dict:
    stamped = stamp(snapshot)
    validate_snapshot(stamped)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(stamped)
    path.write_bytes(payload)
    loaded = parse_strict_json(path.read_text(encoding="utf-8"))
    if loaded["content_hash"] != content_hash(loaded):
        raise SceneStateError("reason=reload_hash_mismatch")
    return loaded


def write_jsonl(path: Path, snapshots: list[dict]) -> str:
    """Write one canonical snapshot per line. Return the SHA-256 of the file bytes."""
    lines = []
    for snapshot in snapshots:
        stamped = stamp(snapshot)
        validate_snapshot(stamped)
        lines.append(canonical_json_bytes(stamped).decode("utf-8").rstrip("\n"))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict], str]:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    snapshots = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        parsed = parse_strict_json(line)
        if not isinstance(parsed, dict):
            raise SceneStateError("reason=snapshot_not_object")
        validate_snapshot(parsed)
        snapshots.append(parsed)
    validate_sequence(snapshots)
    return snapshots, digest


def pose_to_world_matrix(position: list[float], quaternion_wxyz: list[float]) -> list[list[float]]:
    matrix = quat_wxyz_to_matrix(quaternion_wxyz, position)
    return [[float(value) for value in row] for row in matrix]


def translation_rotation_error(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Return translation error in meters and rotation error in degrees."""
    a = validate_rigid(left)
    b = validate_rigid(right)
    translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    degrees = float(np.degrees(np.arccos(cosine)))
    return translation, degrees


def _finite_vec(value: object, size: int, place: str) -> None:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise SceneStateError(f"reason=vector field={place}")
