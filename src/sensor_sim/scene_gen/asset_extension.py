"""Sidecar for one asset_package_v1 directory.

The sidecar is not a file inside the package. ``verify_package`` rejects an
unlisted file, and the package hash must stay stable while this document
points at it.
"""

from __future__ import annotations

import math
from pathlib import Path

from sensor_sim.scene_gen.asset_package import PackageError, canonical_json_bytes, normalize_payload_path, parse_strict_json

SCHEMA_VERSION = "asset_extension_v1"
PROVENANCE = frozenset({"measured", "sourced", "estimated", "default"})
FRAME_NAMES = frozenset({"asset_root", "base_contact", "support_surface"})
COLLISION_METHODS = frozenset({"convex_mesh", "convex_decomposition", "primitive_proxy", "other"})


class AssetExtensionError(ValueError):
    """The sidecar does not describe the package it names."""


def load_extension(text: str) -> dict:
    try:
        parsed = parse_strict_json(text)
    except PackageError as exc:
        raise AssetExtensionError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise AssetExtensionError("reason=extension_not_object")
    return parsed


def validate_extension(document: dict, package: dict) -> None:
    """Reject a sidecar whose identity or hash does not match ``package``."""
    _require(document, {"schema_version", "asset_id", "package_version", "package_sha256", "frames", "regions", "geometries"})
    if document["schema_version"] != SCHEMA_VERSION:
        raise AssetExtensionError("reason=schema_version")
    if document["asset_id"] != package.get("asset_id"):
        raise AssetExtensionError("reason=asset_id_mismatch")
    if document["package_version"] != package.get("version"):
        raise AssetExtensionError("reason=package_version_mismatch")
    if document["package_sha256"] != package.get("package_sha256"):
        raise AssetExtensionError("reason=package_sha256_mismatch")
    frames = document["frames"]
    if not isinstance(frames, list) or not frames:
        raise AssetExtensionError("reason=frames")
    names = set()
    for frame in frames:
        _require(frame, {"name", "purpose", "origin_m", "axes", "transform_to_asset", "units", "provenance"})
        if frame["name"] not in FRAME_NAMES and not str(frame["name"]).startswith("contact_"):
            raise AssetExtensionError(f"reason=frame_name name={frame['name']}")
        if frame["name"] in names:
            raise AssetExtensionError(f"reason=duplicate_frame name={frame['name']}")
        names.add(frame["name"])
        if frame["units"] != "m":
            raise AssetExtensionError("reason=frame_units")
        _provenance(frame["provenance"])
        _vec3(frame["origin_m"], "origin_m")
        _matrix(frame["transform_to_asset"])
    if "asset_root" not in names:
        raise AssetExtensionError("reason=asset_root_missing")
    for region in document["regions"]:
        _require(region, {"id", "region_type", "frame", "geometry", "normal", "units", "provenance", "method"})
        if region["frame"] not in names:
            raise AssetExtensionError(f"reason=region_frame id={region['id']}")
        if region["units"] != "m":
            raise AssetExtensionError("reason=region_units")
        _provenance(region["provenance"])
        _vec3(region["normal"], "normal")
    for geometry in document["geometries"]:
        _require(geometry, {"role", "path", "sha256", "frame", "units", "method", "validation"})
        if geometry["role"] not in {"render", "collision"}:
            raise AssetExtensionError(f"reason=geometry_role role={geometry['role']}")
        if geometry["role"] == "collision" and geometry["method"] not in COLLISION_METHODS:
            raise AssetExtensionError("reason=collision_method")
        normalize_payload_path(geometry["path"])
        if geometry["units"] != "m":
            raise AssetExtensionError("reason=geometry_units")
        if geometry["frame"] not in names:
            raise AssetExtensionError("reason=geometry_frame")
    render_paths = {item["path"] for item in document["geometries"] if item["role"] == "render"}
    collision_paths = {item["path"] for item in document["geometries"] if item["role"] == "collision"}
    if not render_paths or not collision_paths:
        raise AssetExtensionError("reason=render_collision_both_required")
    if render_paths & collision_paths:
        raise AssetExtensionError("reason=collision_reuses_render_path")
    for entry in document.get("physics") or []:
        _require(entry, {"name", "value", "units", "provenance", "method"})
        _provenance(entry["provenance"])
        if isinstance(entry["value"], bool) or not isinstance(entry["value"], (int, float)) or not math.isfinite(float(entry["value"])):
            raise AssetExtensionError(f"reason=physics_value name={entry['name']}")
    interaction = document.get("interaction") or {}
    for grasp in interaction.get("grasp_candidates") or []:
        if grasp.get("qualification") != "unqualified_geometry":
            raise AssetExtensionError("reason=grasp_not_robot_validated")


def extension_bytes(document: dict) -> bytes:
    return canonical_json_bytes(document)


def _require(value: object, keys: set[str]) -> None:
    if not isinstance(value, dict) or not keys <= set(value):
        raise AssetExtensionError(f"reason=missing_key keys={sorted(keys - set(value if isinstance(value, dict) else []))}")


def _provenance(value: object) -> None:
    if value not in PROVENANCE:
        raise AssetExtensionError(f"reason=provenance value={value}")


def _vec3(value: object, place: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise AssetExtensionError(f"reason=vector place={place}")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise AssetExtensionError(f"reason=nonfinite place={place}")


def _matrix(value: object) -> None:
    if not isinstance(value, list) or len(value) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in value):
        raise AssetExtensionError("reason=transform_shape")
    for row in value:
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise AssetExtensionError("reason=nonfinite_transform")


def geometry_path(package_root: Path, relative: str) -> Path:
    """Resolve a package-relative path and reject a target outside the package."""
    relative = normalize_payload_path(relative)
    root = package_root.resolve()
    target = (root / relative).resolve()
    if root != target and root not in target.parents:
        raise AssetExtensionError(f"reason=path_escape path={relative}")
    return target
