"""A task request before it is resolved into a scene manifest.

Counts of one are what the first pick-and-place execution uses. The schema
still accepts a count range so a later task can name more than one object.
"""

from __future__ import annotations

import hashlib
import math

from sensor_sim.scene_gen.asset_package import PackageError, parse_strict_json

SCHEMA_VERSION = "task_scene_request_v1"
TASK_TYPES = frozenset({"pick_and_place"})
SEED_DOMAINS = ("selection", "layout", "appearance", "camera", "lighting")
UNSUPPORTED_RELATIONS = frozenset({"attached_to", "aligned_with", "facing", "reachable_from"})


class TaskRequestError(ValueError):
    """The request cannot be resolved as written."""


def load_request(text: str) -> dict:
    try:
        parsed = parse_strict_json(text)
    except PackageError as exc:
        raise TaskRequestError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise TaskRequestError("reason=request_not_object")
    return parsed


def validate_request(document: dict, *, known_categories: set[str] | None = None, known_assets: set[str] | None = None) -> None:
    required = {
        "schema_version",
        "task_type",
        "support",
        "objects",
        "target_region",
        "camera_profile",
        "light_profile",
        "seeds",
        "relations",
    }
    if not isinstance(document, dict) or not required <= set(document):
        missing = sorted(required - set(document if isinstance(document, dict) else []))
        raise TaskRequestError(f"reason=missing_key keys={missing}")
    if document["schema_version"] != SCHEMA_VERSION:
        raise TaskRequestError("reason=schema_version")
    if document["task_type"] not in TASK_TYPES:
        raise TaskRequestError(f"reason=task_type value={document['task_type']}")
    _selection(document["support"], known_categories, known_assets, "support")
    if not isinstance(document["objects"], list) or not document["objects"]:
        raise TaskRequestError("reason=objects")
    for item in document["objects"]:
        _selection(item, known_categories, known_assets, "object")
        _count(item.get("count"))
        _optional_positive(item.get("spacing_m"), "spacing_m")
    if document["task_type"] == "pick_and_place":
        if len(document["objects"]) != 1 or int(document["objects"][0]["count"]["minimum"]) != 1 or int(document["objects"][0]["count"]["maximum"]) != 1:
            raise TaskRequestError("reason=pick_and_place_requires_one_object")
        if document["support"].get("count", {}).get("minimum", 1) != 1:
            raise TaskRequestError("reason=pick_and_place_requires_one_support")
    region = document["target_region"]
    for key in ("frame", "origin_m", "size_m"):
        if key not in region:
            raise TaskRequestError(f"reason=target_region field={key}")
    if region["frame"] != "support_surface":
        raise TaskRequestError("reason=target_frame")
    _vec3(region["origin_m"], "target_origin")
    _vec3(region["size_m"], "target_size")
    if any(float(value) <= 0 for value in region["size_m"]):
        raise TaskRequestError("reason=target_size")
    _seeds(document["seeds"])
    for relation in document["relations"]:
        kind = relation.get("type")
        if kind in UNSUPPORTED_RELATIONS:
            raise TaskRequestError(f"reason=unsupported_relation relation={kind}")
        if kind not in {"supported_on", "keep_clear_of"}:
            raise TaskRequestError(f"reason=unsupported_relation relation={kind}")
    clutter = document.get("clutter", 0.0)
    if isinstance(clutter, bool) or not isinstance(clutter, (int, float)) or not 0.0 <= float(clutter) <= 1.0:
        raise TaskRequestError("reason=clutter")


def domain_seed(master: int, domain: str) -> int:
    """Deterministic domain seed. This does not use Python's process hash."""
    if domain not in SEED_DOMAINS:
        raise TaskRequestError(f"reason=seed_domain domain={domain}")
    if isinstance(master, bool) or not isinstance(master, int) or master < 0:
        raise TaskRequestError("reason=seed")
    digest = hashlib.sha256(f"{master}:{domain}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _selection(item: object, categories: set[str] | None, assets: set[str] | None, place: str) -> None:
    if not isinstance(item, dict):
        raise TaskRequestError(f"reason=selection place={place}")
    asset_id = item.get("asset_id")
    category = item.get("category")
    if asset_id is None and category is None:
        raise TaskRequestError(f"reason=missing_asset place={place}")
    if categories is not None and category is not None and category not in categories:
        raise TaskRequestError(f"reason=unknown_category category={category}")
    if assets is not None and asset_id is not None and asset_id not in assets:
        raise TaskRequestError(f"reason=missing_asset_id asset_id={asset_id}")
    for key in ("width_m", "depth_m", "height_m"):
        if key in item:
            _optional_positive(item[key], key)


def _count(value: object) -> None:
    if not isinstance(value, dict) or not {"minimum", "maximum"} <= set(value):
        raise TaskRequestError("reason=count")
    low = value["minimum"]
    high = value["maximum"]
    if isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, int) or not isinstance(high, int):
        raise TaskRequestError("reason=count")
    if low < 1 or high < low:
        raise TaskRequestError("reason=count")


def _seeds(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(SEED_DOMAINS):
        raise TaskRequestError("reason=seeds")
    for name, seed in value.items():
        domain_seed(seed, name)


def _optional_positive(value: object, place: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
        raise TaskRequestError(f"reason=dimension field={place}")


def _vec3(value: object, place: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise TaskRequestError(f"reason=vector place={place}")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise TaskRequestError(f"reason=nonfinite place={place}")
