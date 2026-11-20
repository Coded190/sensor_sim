"""Require a furnished room for a realistic scene.

A scene passes when it has a room shell and at least one other object.
The shell is a floor, a ceiling, and walls on the south, north, east, and west.
A ground plane or a rectangle under the subjects is not a floor.

``context_objects`` counts meshes that are not part of that shell and whose
role is not ``subject``. Subjects are the objects under test. They do not
satisfy the other-object requirement.

Dataset v1 ``generate.py`` and ``room_generator.py`` are hash-locked, so this check
is not edited into those files. Their saved bundles are still checked. New
scene assembly in the interior, semantic, tabletop, and material-approval
builders calls ``require_scene_context`` before it accepts a scene.

Room kinds are drawn from ``ROOM_DISTRIBUTION``. ``lab`` uses the workshop
shell until a separate lab shell exists. ``room`` uses the dining shell.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.indoor_transform import apply_points, deserialize_matrix
from sensor_sim.scene_gen.mesh_io import read_obj

logger = logging.getLogger("sensor_sim.scene_gen.scene_context")

INTERIOR_ROOT = Path("dataset/qualification/diverse_interior_review")
ROOM_DISTRIBUTION = (
    ("room", 1.0),
    ("kitchen", 1.0),
    ("dining", 1.0),
    ("living", 1.0),
    ("bedroom", 1.0),
    ("office", 1.0),
    ("workshop", 1.0),
    ("lab", 1.0),
)
GEOMETRY_KIND = {
    "room": "dining",
    "kitchen": "kitchen",
    "dining": "dining",
    "living": "living",
    "bedroom": "bedroom",
    "office": "office",
    "workshop": "workshop",
    "lab": "workshop",
}
SUPPORT_INSTANCE = {
    "kitchen": "island_1",
    "dining": "table_1",
    "living": "coffee_1",
    "bedroom": "bed_1",
    "office": "desk_1",
    "workshop": "workbench_1",
}
WALLS = ("south", "north", "east", "west")
MIN_CONTEXT_OBJECTS = 1
DEFINITION = (
    "Passed means floors>=1, ceilings>=1, walls are south,north,east,west, "
    f"and context_objects>={MIN_CONTEXT_OBJECTS}. "
    "context_objects excludes the shell and role=subject. "
    "A kind=plane or asset_id ground/plane is a bare plane, not a floor. "
    "Exemptions require a non-empty exemption string."
)


class SceneContextError(ValueError):
    """The scene is a bare plane, has no room shell, or has no other objects."""


def sample_room_kind(rng, distribution=ROOM_DISTRIBUTION) -> str:
    """Draw one room kind. ``rng.random()`` must return a float in [0, 1)."""
    return sample_room_kinds(rng, 1, distribution)[0]


def sample_room_kinds(rng, count: int, distribution=ROOM_DISTRIBUTION) -> list[str]:
    """Draw without replacement, using the distribution weights."""
    pool = [(str(name), float(weight)) for name, weight in distribution]
    if int(count) > len(pool) or int(count) < 1:
        raise SceneContextError(f"reason=sample_count count={count} available={len(pool)}")
    chosen = []
    for _ in range(int(count)):
        total = sum(weight for _name, weight in pool if weight > 0.0)
        if total <= 0.0:
            raise SceneContextError("reason=distribution_weight")
        pick = float(rng.random()) * total
        cursor = 0.0
        selected = None
        for index, (name, weight) in enumerate(pool):
            if weight <= 0.0:
                continue
            cursor += weight
            if pick <= cursor:
                selected = index
                break
        if selected is None:
            selected = len(pool) - 1
        chosen.append(pool.pop(selected)[0])
    logger.info("scene_context event=sample kinds=%s", ",".join(chosen))
    return chosen


def classify_instance(instance: dict) -> str:
    """Return floor, ceiling, wall_<direction>, bare_plane, subject, or context."""
    role = str(instance.get("role") or "").lower()
    asset = str(instance.get("asset_id") or instance.get("name") or instance.get("id") or "").lower()
    kind = str(instance.get("kind") or instance.get("family") or "").lower()
    if role == "subject" or instance.get("subject") is True:
        return "subject"
    if kind == "plane" or asset in {"plane", "ground"} or role in {"ground", "plane"}:
        return "bare_plane"
    for direction in WALLS:
        if f"wall_{direction}" in asset:
            return f"wall_{direction}"
    if asset.endswith("_ceiling") or asset == "room_ceiling" or role == "ceiling":
        return "ceiling"
    if asset.endswith("_floor") or asset == "room_floor" or role == "floor":
        return "floor"
    return "context"


def scene_context_report(instances: list[dict], *, room_kind: str) -> dict:
    """Count shell parts and other objects. This does not accept or reject."""
    if not str(room_kind).strip():
        raise SceneContextError("reason=missing_room_kind")
    counts = {"floor": 0, "ceiling": 0, "bare_plane": 0, "subject": 0, "context": 0}
    walls = set()
    for instance in instances:
        part = classify_instance(instance)
        if part.startswith("wall_"):
            walls.add(part.removeprefix("wall_"))
            continue
        counts[part] = counts.get(part, 0) + 1
    return {
        "room_kind": str(room_kind),
        "floors": counts["floor"],
        "ceilings": counts["ceiling"],
        "walls": [direction for direction in WALLS if direction in walls],
        "bare_planes": counts["bare_plane"],
        "context_objects": counts["context"],
        "subjects": counts["subject"],
        "min_context_objects": MIN_CONTEXT_OBJECTS,
        "definition": DEFINITION,
    }


def require_scene_context(
    instances: list[dict],
    *,
    room_kind: str,
    allow_bare_plane: bool = False,
    allow_without_context_objects: bool = False,
    exemption: str = "",
) -> dict:
    """Reject a bare plane or a room with no other objects.

    An exemption is valid only when ``exemption`` is a non-empty string.
    """
    report = scene_context_report(instances, room_kind=room_kind)
    reasons = []
    specified = bool(str(exemption).strip())
    if allow_bare_plane or allow_without_context_objects:
        if not specified:
            raise SceneContextError("reason=exemption_unspecified")
    if report["bare_planes"] and not (allow_bare_plane and specified):
        reasons.append("bare_plane")
    if report["floors"] < 1 or report["ceilings"] < 1 or report["walls"] != list(WALLS):
        if not (allow_bare_plane and specified):
            missing = [direction for direction in WALLS if direction not in report["walls"]]
            if report["floors"] < 1:
                reasons.append("missing_floor")
            if report["ceilings"] < 1:
                reasons.append("missing_ceiling")
            if missing:
                reasons.append("missing_wall:" + ",".join(missing))
    if report["context_objects"] < MIN_CONTEXT_OBJECTS and not (allow_without_context_objects and specified):
        reasons.append("missing_context_objects")
    report["exemption"] = str(exemption).strip()
    report["passed"] = not reasons
    if reasons:
        reason = "+".join(reasons)
        logger.info(
            "scene_context event=reject reason=%s room_kind=%s floors=%d ceilings=%d walls=%s context_objects=%d subjects=%d bare_planes=%d",
            reason,
            report["room_kind"],
            report["floors"],
            report["ceilings"],
            ",".join(report["walls"]),
            report["context_objects"],
            report["subjects"],
            report["bare_planes"],
        )
        raise SceneContextError(
            f"reason={reason} room_kind={report['room_kind']} "
            f"floors={report['floors']} ceilings={report['ceilings']} walls={','.join(report['walls'])} "
            f"context_objects={report['context_objects']} subjects={report['subjects']} bare_planes={report['bare_planes']}"
        )
    logger.info(
        "scene_context event=check room_kind=%s floors=%d ceilings=%d walls=%s context_objects=%d subjects=%d bare_planes=%d passed=true exemption=%s",
        report["room_kind"],
        report["floors"],
        report["ceilings"],
        ",".join(report["walls"]),
        report["context_objects"],
        report["subjects"],
        report["bare_planes"],
        report["exemption"] or "none",
    )
    return report


def load_furnished_scene(room_kind: str, root: Path = INTERIOR_ROOT) -> dict:
    """Load one saved furnished interior for ``room_kind`` and check it."""
    if room_kind not in GEOMETRY_KIND:
        raise SceneContextError(f"reason=unknown_room_kind kind={room_kind}")
    geometry = GEOMETRY_KIND[room_kind]
    matches = sorted((Path(root) / geometry).glob("*/bundle.json"))
    if not matches:
        raise SceneContextError(f"reason=missing_furnished_room room_kind={room_kind} geometry_kind={geometry}")
    bundle_path = matches[0]
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    report = require_scene_context(bundle["instances"], room_kind=room_kind)
    loaded = {
        "room_kind": room_kind,
        "geometry_kind": geometry,
        "lab_uses_workshop_shell": room_kind == "lab",
        "bundle_dir": str(bundle_path.parent),
        "bundle": bundle,
        "report": report,
    }
    logger.info(
        "scene_context event=furnished_room room_kind=%s geometry_kind=%s lab_uses_workshop_shell=%s path=%s context_objects=%d",
        room_kind,
        geometry,
        loaded["lab_uses_workshop_shell"],
        bundle_path.parent,
        report["context_objects"],
    )
    return loaded


def support_top(loaded: dict) -> list[float]:
    """World center of the packaged ``top`` support polygon.

    XY is the mean of that polygon after the instance matrix. Z is the maximum
    Z of the same polygon, in meters. Subjects sit on this height. If the
    package has no ``top`` polygon, Z is the maximum mesh Z and XY is the mean
    of vertices within 2 cm of that height.
    """
    geometry = loaded["geometry_kind"]
    instance_id = SUPPORT_INSTANCE[geometry]
    record = next(item for item in loaded["bundle"]["instances"] if item["instance_id"] == instance_id)
    package = Path(record["catalog"]) / record["asset_id"] / record["version"]
    manifest = json.loads((package / "asset.json").read_text(encoding="utf-8"))
    matrix = deserialize_matrix(record["matrix"])
    surfaces = manifest.get("support_surfaces") or []
    top = next((item for item in surfaces if item.get("id") == "top"), None)
    if top is not None:
        world = apply_points(matrix, np.asarray(top["polygon"], dtype=np.float64))
        source = "support_polygon"
        used = int(len(world))
    else:
        vertices, _faces = read_obj(package / "geometry" / "render.obj")
        world = apply_points(matrix, vertices)
        top_z = float(np.max(world[:, 2]))
        world = world[world[:, 2] >= top_z - 0.02]
        source = "mesh_top"
        used = int(len(world))
    center = world.mean(axis=0)
    point = [float(center[0]), float(center[1]), float(np.max(world[:, 2]))]
    logger.info(
        "scene_context event=support room_kind=%s instance=%s source=%s x=%.4f y=%.4f z=%.4f vertices=%d",
        loaded["room_kind"],
        instance_id,
        source,
        point[0],
        point[1],
        point[2],
        used,
    )
    return point


def compose_scene_records(room_kind: str, subjects: list[str], root: Path = INTERIOR_ROOT) -> dict:
    """Furnished room plus subject records. Subjects do not count as context objects."""
    loaded = load_furnished_scene(room_kind, root)
    records = [dict(item) for item in loaded["bundle"]["instances"]]
    for name in subjects:
        records.append({"instance_id": str(name), "asset_id": str(name), "role": "subject", "name": str(name)})
    loaded["records"] = records
    loaded["support_top_m"] = support_top(loaded)
    loaded["report"] = require_scene_context(records, room_kind=room_kind)
    return loaded
