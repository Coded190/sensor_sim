"""Reject a placement that puts one object inside another.

``clearance_m`` is the minimum triangle-to-triangle distance in meters.
A pair fails when that distance is below ``MINIMUM_CLEARANCE_M`` (4 mm), or
when one mesh's box sits inside the other and its centroid is inside that mesh.

The room shell is not a blocker: the floor, ceiling, and four walls enclose
the scene. The flat top of the support is not a blocker, because objects are
supposed to rest on it. Raised parts of that same mesh, such as pillows or a
headboard, are blockers. A support pair listed in ``allowed_pairs`` may touch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_validate import triangulate
from sensor_sim.scene_gen.indoor_transform import apply_points, deserialize_matrix
from sensor_sim.scene_gen.mesh_io import read_obj
from sensor_sim.scene_gen.narrow import contains_point, mesh_clearance
from sensor_sim.scene_gen.scene_context import SUPPORT_INSTANCE, classify_instance

logger = logging.getLogger("sensor_sim.scene_gen.placement_separation")

MINIMUM_CLEARANCE_M = 0.004
RAISED_ABOVE_SUPPORT_M = 0.015
DEFINITION = (
    "clearance_m is the minimum triangle-to-triangle distance. "
    f"A pair is inside another object when clearance_m < {MINIMUM_CLEARANCE_M} m, "
    "or when one axis-aligned box is inside the other and its centroid is inside that mesh. "
    "The room shell and the flat support top are excluded. Raised support geometry is included."
)


class PlacementSeparationError(ValueError):
    """An object was placed inside another object, or no clear support cell exists."""


def _aabb(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vertices, dtype=np.float64)
    return points.min(axis=0), points.max(axis=0)


def _aabb_gap(low_a, high_a, low_b, high_b) -> float:
    delta = np.maximum(np.maximum(low_a - high_b, low_b - high_a), 0.0)
    return float(np.linalg.norm(delta))


def _box_inside(low_a, high_a, low_b, high_b) -> bool:
    return bool(np.all(low_a >= low_b) and np.all(high_a <= high_b))


def _world_mesh(record: dict) -> dict:
    package = Path(record["catalog"]) / record["asset_id"] / record["version"]
    vertices, polygons = read_obj(package / "geometry" / "render.obj")
    world = apply_points(deserialize_matrix(record["matrix"]), vertices)
    return {
        "instance_id": str(record["instance_id"]),
        "asset_id": str(record["asset_id"]),
        "vertices": world,
        "faces": triangulate(polygons),
        "part": classify_instance(record),
    }


def _raised_faces(vertices: np.ndarray, faces: np.ndarray, support_z: float) -> np.ndarray:
    if len(faces) == 0:
        return faces
    centroids = vertices[faces].mean(axis=1)
    return faces[centroids[:, 2] > float(support_z) + RAISED_ABOVE_SUPPORT_M]


def support_frame(loaded: dict) -> dict:
    """World support polygon, its top Z, and the support mesh."""
    instance_id = SUPPORT_INSTANCE[loaded["geometry_kind"]]
    record = next(item for item in loaded["bundle"]["instances"] if item["instance_id"] == instance_id)
    package = Path(record["catalog"]) / record["asset_id"] / record["version"]
    manifest_polygon = None
    import json

    manifest = json.loads((package / "asset.json").read_text(encoding="utf-8"))
    for surface in manifest.get("support_surfaces") or []:
        if surface.get("id") == "top":
            manifest_polygon = np.asarray(surface["polygon"], dtype=np.float64)
            break
    mesh = _world_mesh(record)
    if manifest_polygon is None:
        polygon = np.asarray([mesh["vertices"].min(axis=0), mesh["vertices"].max(axis=0)])
        support_z = float(mesh["vertices"][:, 2].max())
    else:
        polygon = apply_points(deserialize_matrix(record["matrix"]), manifest_polygon)
        support_z = float(np.max(polygon[:, 2]))
    return {"instance_id": instance_id, "polygon": polygon, "z": support_z, "mesh": mesh}


def blockers_for(loaded: dict, frame: dict | None = None) -> list[dict]:
    """Furniture and props that a new object must not enter. The shell is omitted."""
    frame = frame or support_frame(loaded)
    blockers = []
    for record in loaded["bundle"]["instances"]:
        part = classify_instance(record)
        if part in {"floor", "ceiling"} or part.startswith("wall_"):
            continue
        mesh = _world_mesh(record)
        if record["instance_id"] == frame["instance_id"]:
            raised = _raised_faces(mesh["vertices"], mesh["faces"], frame["z"])
            if len(raised) == 0:
                continue
            blockers.append({**mesh, "instance_id": f"{frame['instance_id']}:raised", "faces": raised})
            continue
        blockers.append(mesh)
    logger.info(
        "placement_separation event=blockers room_kind=%s support=%s count=%d",
        loaded.get("room_kind"),
        frame["instance_id"],
        len(blockers),
    )
    return blockers


def separation_report(
    items: list[dict],
    *,
    existing: list[dict] | None = None,
    minimum_m: float = MINIMUM_CLEARANCE_M,
    allowed_pairs: set[frozenset] | None = None,
) -> dict:
    """Pairwise inside-object report. This does not move anything.

    ``existing`` objects are checked against ``items``, and ``items`` are
    checked against each other. Pairs that are only inside ``existing`` are
    not measured. Those objects were already in the room.
    """
    allowed = allowed_pairs or set()
    pairs: list[tuple[dict, dict]] = []
    if existing is None:
        for index, left in enumerate(items):
            pairs.extend((left, right) for right in items[index + 1 :])
    else:
        for left in items:
            pairs.extend((left, right) for right in existing)
        for index, left in enumerate(items):
            pairs.extend((left, right) for right in items[index + 1 :])
    failures = []
    checked = 0
    for left, right in pairs:
        pair = frozenset((str(left["instance_id"]), str(right["instance_id"])))
        if pair in allowed:
            continue
        low_a, high_a = _aabb(left["vertices"])
        low_b, high_b = _aabb(right["vertices"])
        gap = _aabb_gap(low_a, high_a, low_b, high_b)
        nested = _box_inside(low_a, high_a, low_b, high_b) or _box_inside(low_b, high_b, low_a, high_a)
        if gap >= float(minimum_m) and not nested:
            continue
        checked += 1
        clearance = float(mesh_clearance(left, right)["clearance_m"])
        centroid_a = np.asarray(left["vertices"], dtype=np.float64).mean(axis=0)
        centroid_b = np.asarray(right["vertices"], dtype=np.float64).mean(axis=0)
        inside = bool(
            nested
            and (
                contains_point(centroid_a, right["vertices"], right["faces"])
                or contains_point(centroid_b, left["vertices"], left["faces"])
            )
        )
        if clearance < float(minimum_m) or inside:
            reason = "centroid_inside" if inside and clearance >= float(minimum_m) else "clearance"
            failures.append(
                {
                    "instances": sorted(pair),
                    "clearance_m": clearance,
                    "aabb_gap_m": gap,
                    "reason": reason,
                }
            )
    return {
        "passed": not failures,
        "failures": failures,
        "pairs_measured": checked,
        "minimum_clearance_m": float(minimum_m),
        "definition": DEFINITION,
    }


def require_outside(
    items: list[dict],
    *,
    existing: list[dict] | None = None,
    minimum_m: float = MINIMUM_CLEARANCE_M,
    allowed_pairs: set[frozenset] | None = None,
) -> dict:
    """Raise when any measured pair is inside another object."""
    report = separation_report(items, existing=existing, minimum_m=minimum_m, allowed_pairs=allowed_pairs)
    if report["failures"]:
        first = report["failures"][0]
        logger.info(
            "placement_separation event=reject reason=%s instances=%s clearance_m=%.6f minimum_m=%.4f failures=%d",
            first["reason"],
            ",".join(first["instances"]),
            first["clearance_m"],
            float(minimum_m),
            len(report["failures"]),
        )
        raise PlacementSeparationError(
            f"reason=object_inside pair={'+'.join(first['instances'])} "
            f"detail={first['reason']} clearance_m={first['clearance_m']:.6f} "
            f"minimum_m={float(minimum_m):.4f} failures={len(report['failures'])}"
        )
    logger.info(
        "placement_separation event=check passed=true items=%d pairs_measured=%d minimum_m=%.4f",
        len(items),
        report["pairs_measured"],
        float(minimum_m),
    )
    return report


def _world_box(local: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = _aabb(local)
    return low + translation, high + translation


def _raised_boxes(vertices: np.ndarray, faces: np.ndarray, step: float = 0.04) -> list[tuple[np.ndarray, np.ndarray]]:
    """One box per 4 cm cell that contains raised support triangles.

    A single box around every pillow and the headboard would mark the whole
    mattress occupied. The cell boxes leave the flat top free.
    """
    if len(faces) == 0:
        return []
    triangles = vertices[np.asarray(faces, dtype=np.int64)]
    low = triangles.min(axis=1)
    high = triangles.max(axis=1)
    origin_x = float(low[:, 0].min())
    origin_y = float(low[:, 1].min())
    cells: dict[tuple[int, int], list[float]] = {}
    for index in range(len(low)):
        x0 = int(np.floor((low[index, 0] - origin_x) / step))
        x1 = int(np.floor((high[index, 0] - origin_x) / step))
        y0 = int(np.floor((low[index, 1] - origin_y) / step))
        y1 = int(np.floor((high[index, 1] - origin_y) / step))
        z0 = float(low[index, 2])
        z1 = float(high[index, 2])
        for x_index in range(x0, x1 + 1):
            for y_index in range(y0, y1 + 1):
                key = (x_index, y_index)
                if key in cells:
                    cells[key][0] = min(cells[key][0], z0)
                    cells[key][1] = max(cells[key][1], z1)
                else:
                    cells[key] = [z0, z1]
    boxes = []
    for (x_index, y_index), (z0, z1) in cells.items():
        box_low = np.array([origin_x + x_index * step, origin_y + y_index * step, z0])
        box_high = np.array([origin_x + (x_index + 1) * step, origin_y + (y_index + 1) * step, z1])
        boxes.append((box_low, box_high))
    return boxes


def _blocker_boxes(blockers: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    boxes = []
    for blocker in blockers:
        if str(blocker["instance_id"]).endswith(":raised"):
            boxes.extend(_raised_boxes(blocker["vertices"], blocker["faces"]))
            continue
        boxes.append(_aabb(blocker["vertices"]))
    return boxes


def _blocked(low, high, boxes, minimum_m: float) -> bool:
    for other_low, other_high in boxes:
        grown_low = other_low - minimum_m
        grown_high = other_high + minimum_m
        if np.all(low < grown_high) and np.all(high > grown_low):
            return True
    return False


def place_clear(
    subjects: list[dict],
    blockers: list[dict],
    support_polygon: np.ndarray,
    *,
    offsets: dict[str, tuple[float, float]] | None = None,
    minimum_m: float = MINIMUM_CLEARANCE_M,
) -> dict:
    """Put each subject on the support, at least ``minimum_m`` from every blocker.

    ``subjects`` carry local ``vertices`` and ``faces``. The returned translation
    moves the local origin. Z puts the lowest vertex on the support top.
    ``offsets`` are XY shifts from one shared anchor, so a row or triangle is
    kept when the group fits. Otherwise each subject is placed on its own cell.
    """
    polygon = np.asarray(support_polygon, dtype=np.float64)
    support_z = float(np.max(polygon[:, 2]))
    pad = float(minimum_m)
    span_min = polygon[:, :2].min(axis=0) + pad
    span_max = polygon[:, :2].max(axis=0) - pad
    center = polygon[:, :2].mean(axis=0)
    blocker_boxes = _blocker_boxes(blockers)
    group = _place_pattern(subjects, blocker_boxes, span_min, span_max, center, support_z, offsets or {}, minimum_m)
    if group is None:
        group = _place_each(subjects, blocker_boxes, span_min, span_max, center, support_z, minimum_m)
        group["pattern_kept"] = False
    else:
        group["pattern_kept"] = True
    placed = []
    for subject, translation in zip(subjects, group["translations"], strict=True):
        world = np.asarray(subject["vertices"], dtype=np.float64) + np.asarray(translation, dtype=np.float64)
        placed.append({"instance_id": subject["name"], "vertices": world, "faces": subject["faces"]})
    report = require_outside(placed, existing=blockers, minimum_m=minimum_m)
    group["report"] = report
    group["support_z_m"] = support_z
    logger.info(
        "placement_separation event=placed subjects=%d pattern_kept=%s minimum_m=%.4f",
        len(subjects),
        group["pattern_kept"],
        float(minimum_m),
    )
    return group


def _candidate_order(lo, hi, center, step: float):
    if np.any(hi < lo):
        return []
    xs = np.arange(float(lo[0]), float(hi[0]) + 1e-9, step)
    ys = np.arange(float(lo[1]), float(hi[1]) + 1e-9, step)
    if len(xs) == 0 or len(ys) == 0:
        xs = np.array([float(lo[0])]) if hi[0] >= lo[0] else np.array([])
        ys = np.array([float(lo[1])]) if hi[1] >= lo[1] else np.array([])
    points = [(float(x), float(y)) for x in xs for y in ys]
    points.sort(key=lambda point: (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
    return points


def _place_pattern(subjects, blocker_boxes, span_min, span_max, center, support_z, offsets, minimum_m):
    if not offsets:
        return None
    locals_ = []
    for subject in subjects:
        local = np.asarray(subject["vertices"], dtype=np.float64)
        offset = np.asarray([offsets.get(subject["name"], (0.0, 0.0))[0], offsets.get(subject["name"], (0.0, 0.0))[1], 0.0])
        locals_.append((subject, local, offset, float(local[:, 2].min())))
    anchor_lo = span_min.copy()
    anchor_hi = span_max.copy()
    for _subject, local, offset, _bottom in locals_:
        low, high = _aabb(local)
        anchor_lo = np.maximum(anchor_lo, span_min - low[:2] - offset[:2])
        anchor_hi = np.minimum(anchor_hi, span_max - high[:2] - offset[:2])
    for anchor_x, anchor_y in _candidate_order(anchor_lo, anchor_hi, center, 0.02):
        boxes = list(blocker_boxes)
        translations = []
        fits = True
        for _subject, local, offset, bottom in locals_:
            translation = np.array([anchor_x + offset[0], anchor_y + offset[1], support_z - bottom])
            low, high = _world_box(local, translation)
            if _blocked(low, high, boxes, minimum_m):
                fits = False
                break
            boxes.append((low, high))
            translations.append(translation.tolist())
        if fits:
            return {"translations": translations, "anchor_m": [anchor_x, anchor_y]}
    return None


def _place_each(subjects, blocker_boxes, span_min, span_max, center, support_z, minimum_m):
    boxes = list(blocker_boxes)
    translations = []
    for subject in subjects:
        local = np.asarray(subject["vertices"], dtype=np.float64)
        low, high = _aabb(local)
        bottom = float(local[:, 2].min())
        lo = span_min - low[:2]
        hi = span_max - high[:2]
        found = None
        for x_value, y_value in _candidate_order(lo, hi, center, 0.02):
            translation = np.array([x_value, y_value, support_z - bottom])
            world_low, world_high = _world_box(local, translation)
            if _blocked(world_low, world_high, boxes, minimum_m):
                continue
            found = translation
            boxes.append((world_low, world_high))
            break
        if found is None:
            raise PlacementSeparationError(f"reason=no_free_cell subject={subject['name']} support_z={support_z:.4f}")
        translations.append(found.tolist())
    return {"translations": translations, "anchor_m": None}
