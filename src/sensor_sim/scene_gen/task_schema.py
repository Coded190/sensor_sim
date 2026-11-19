"""Strict schema for task-conditioned scenes.

Positions and dimensions are meters. Revolute values and angular tolerances are
radians. Prismatic values are meters. A local coordinate is never read as a
world point unless its frame says so.
"""

from __future__ import annotations

import hashlib
import re

from sensor_sim.scene_gen.asset_package import PackageError, canonical_json_bytes, parse_strict_json

SCHEMA_VERSION = 1
FRAMES = frozenset({"world", "instance_local", "support_surface_local", "container_interior_local"})
RELATIONS = frozenset(
    {
        "supported_on",
        "inside",
        "adjacent_to",
        "attached_to",
        "aligned_with",
        "facing",
        "near",
        "keep_clear_of",
        "reachable_from",
        "task_target",
    }
)
PRIOR_STATUS = frozenset({"sourced", "curated", "asset_declared", "unknown"})
_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


class TaskError(ValueError):
    """A task specification or resolved bundle failed closed."""


def load_task(text: str) -> dict:
    try:
        parsed = parse_strict_json(text)
    except PackageError as exc:
        raise TaskError(str(exc)) from exc
    validate_task(parsed)
    return parsed


def validate_task(document: dict) -> None:
    _require_keys(document, {"schema_version", "task"})
    if document["schema_version"] != SCHEMA_VERSION:
        raise TaskError("reason=schema_version")
    task = document["task"]
    _require_keys(
        task,
        {
            "task_id",
            "name",
            "description",
            "category",
            "required_objects",
            "contextual_objects",
            "robot",
            "workspace_m",
            "constraints",
            "initial_requirements",
            "goal_requirements",
            "seed",
        },
    )
    if _ID.fullmatch(str(task["task_id"])) is None:
        raise TaskError("reason=task_id")
    if not isinstance(task["seed"], int):
        raise TaskError("reason=seed")
    if task["robot"] is not None and not isinstance(task["robot"], dict):
        raise TaskError("reason=robot")
    _objects(task["required_objects"], required=True)
    _objects(task["contextual_objects"], required=False)
    _finite_box(task["workspace_m"])


def validate_bundle(document: dict) -> None:
    _require_keys(
        document,
        {
            "schema_version",
            "task",
            "task_sha256",
            "query_template",
            "retrieval",
            "proposal",
            "resolved",
            "validation",
        },
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise TaskError("reason=schema_version")
    validate_task({"schema_version": SCHEMA_VERSION, "task": document["task"]})
    if document["task_sha256"] != hashlib.sha256(canonical_json_bytes(document["task"])).hexdigest():
        raise TaskError("reason=task_hash")
    _selections(document["proposal"]["assets"], proposed=True)
    _selections(document["resolved"]["assets"], proposed=False)
    _poses(document["proposal"]["poses"])
    _poses(document["resolved"]["poses"])
    _relations(document["proposal"]["relations"])
    _relations(document["resolved"]["relations"])
    if not isinstance(document["resolved"]["collision_log"], dict):
        raise TaskError("reason=collision_log")


def document_hash(value: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _objects(items: object, *, required: bool) -> None:
    if not isinstance(items, list):
        raise TaskError("reason=objects")
    if required and not items:
        raise TaskError("reason=required_objects")
    seen = set()
    for item in items:
        _require_keys(item, {"instance_id", "query", "affordances", "role"})
        if _ID.fullmatch(item["instance_id"]) is None or item["instance_id"] in seen:
            raise TaskError("reason=instance_id")
        seen.add(item["instance_id"])
        if item["role"] not in {"required", "contextual"}:
            raise TaskError("reason=role")
        if not isinstance(item["affordances"], list):
            raise TaskError("reason=affordances")


def _selections(items: object, *, proposed: bool) -> None:
    if not isinstance(items, list):
        raise TaskError("reason=assets")
    seen = set()
    for item in items:
        required = {
            "instance_id",
            "semantic_name",
            "category",
            "role",
            "asset_id",
            "version",
            "package_sha256",
            "query",
            "score",
            "score_type",
            "license_status",
            "technical_status",
            "renderer_status",
            "visual_verification",
        }
        _require_keys(item, required)
        if _ID.fullmatch(item["instance_id"]) is None or item["instance_id"] in seen:
            raise TaskError("reason=instance_id")
        seen.add(item["instance_id"])
        if _HASH.fullmatch(item["package_sha256"]) is None:
            raise TaskError("reason=package_sha256")
        if item["visual_verification"] not in {"pending", "verified", "rejected"}:
            raise TaskError("reason=visual_verification")
        if proposed and item["renderer_status"] not in {"candidate", "technically_validated", "renderer_validated"}:
            raise TaskError("reason=renderer_status")


def _poses(items: object) -> None:
    if not isinstance(items, list):
        raise TaskError("reason=poses")
    for item in items:
        _require_keys(item, {"instance_id", "matrix", "scale", "articulation"})
        if len(item["matrix"]) != 16 or not all(_finite_number(value) for value in item["matrix"]):
            raise TaskError("reason=matrix")
        if len(item["scale"]) != 3 or any(float(value) <= 0.0 for value in item["scale"]):
            raise TaskError("reason=scale")
        if item["articulation"] != "not_applicable" and not isinstance(item["articulation"], dict):
            raise TaskError("reason=articulation")


def _relations(items: object) -> None:
    if not isinstance(items, list):
        raise TaskError("reason=relations")
    for item in items:
        _require_keys(item, {"type", "instances", "frame", "priority", "constraints"})
        if item["type"] not in RELATIONS:
            raise TaskError(f"reason=relation type={item['type']}")
        if item["frame"] not in FRAMES:
            raise TaskError("reason=frame")
        if item["priority"] not in {"required", "preferred"}:
            raise TaskError("reason=priority")
        if item["frame"] != "world" and "frame_instance" not in item:
            raise TaskError("reason=frame_instance")


def _require_keys(value: object, keys: set[str]) -> None:
    if not isinstance(value, dict):
        raise TaskError("reason=fields")
    missing = keys - set(value)
    if missing:
        raise TaskError(f"reason=fields missing={sorted(missing)}")
    allowed_extra = {
        "frame_instance",
        "surface",
        "measured_m",
        "passed",
        "reason",
        "gap_limit_m",
        "penetration_limit_m",
    }
    unknown = set(value) - keys - allowed_extra
    if unknown:
        raise TaskError(f"reason=unknown_field fields={sorted(unknown)}")


def _finite_box(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise TaskError("reason=workspace")
    _require_keys(value, {"minimum_m", "maximum_m"})
    for key in ("minimum_m", "maximum_m"):
        if len(value[key]) != 3 or not all(_finite_number(item) for item in value[key]):
            raise TaskError("reason=workspace")


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float("-inf") < float(value) < float("inf")
