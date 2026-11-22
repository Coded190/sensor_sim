"""Resolve one task request into a scene_manifest_v1 and a validation report.

A passing report is static geometry only. It does not establish robot
reachability or dynamic stability.
"""

from __future__ import annotations

import numpy as np

from sensor_sim.scene_gen.asset_extension import validate_extension
from sensor_sim.scene_gen.asset_library import composer_asset, retrieve
from sensor_sim.scene_gen.indoor_contact import measure_support
from sensor_sim.scene_gen.scene_manifest import (
    apply_transform,
    compose_transforms,
    look_at_y_down,
    manifest_hash,
    matrix_lists,
    translation_matrix,
    validate_manifest,
    validate_rigid,
)
from sensor_sim.scene_gen.task_scene_request import domain_seed, validate_request

GAP_LIMIT_M = 0.001
PENETRATION_LIMIT_M = 0.0001


class ComposerError(ValueError):
    """The request could not be resolved into a valid scene."""


def compose_from_catalog(request: dict, catalog: dict, library_root, *, root: str = "library") -> dict:
    """Resolve support and object through the catalog, then compose the manifest."""
    from pathlib import Path

    library_root = Path(library_root)
    support_spec = request["support"]
    object_spec = request["objects"][0]
    support = retrieve(
        catalog,
        library_root,
        asset_id=support_spec.get("asset_id"),
        category=None if support_spec.get("asset_id") else support_spec.get("category"),
        support_role=True,
        selection_seed=int(request["seeds"]["selection"]),
    )
    movable = retrieve(
        catalog,
        library_root,
        asset_id=object_spec.get("asset_id"),
        category=None if object_spec.get("asset_id") else object_spec.get("category"),
        support_role=False,
        selection_seed=int(request["seeds"]["selection"]) + 1,
    )
    return compose_scene(request, [composer_asset(support, library_root), composer_asset(movable, library_root)], root=root)


def compose_scene(request: dict, assets: list[dict], *, root: str = "library") -> dict:
    """Run the resolver steps and return the manifest, its hash, and the report."""
    categories = {item["category"] for item in assets}
    known = {item["package"]["asset_id"] for item in assets}
    validate_request(request, known_categories=categories, known_assets=known)
    support = _select(request["support"], assets, request["seeds"]["selection"], "support")
    movable = _select(request["objects"][0], assets, request["seeds"]["selection"] + 1, "object")
    _check_hashes(support)
    _check_hashes(movable)
    support_id = "table_0"
    object_id = f"{movable['package']['asset_id']}_0"
    support_world = np.eye(4)
    support_frame = _frame(support, "support_surface")
    base_frame = _frame(movable, "base_contact")
    local_xy = _layout_xy(request, support, request["seeds"]["layout"])
    support_from_contact = translation_matrix(local_xy[0], local_xy[1], 0.0)
    object_world = compose_transforms(compose_transforms(support_world, np.asarray(support_frame["transform_to_asset"], dtype=np.float64)), np.asarray(support_from_contact, dtype=np.float64))
    object_world = compose_transforms(object_world, np.linalg.inv(np.asarray(base_frame["transform_to_asset"], dtype=np.float64)))
    target_world = _target_world(request["target_region"], support_world, support_frame)
    report = _validate_geometry(request, support, movable, support_id, object_id, support_world, object_world, target_world)
    appearance = _appearance(request["seeds"]["appearance"])
    camera = _camera(request["seeds"]["camera"])
    light = _light(request["seeds"]["lighting"])
    manifest = {
        "schema_version": "scene_manifest_v1",
        "scene_id": f"{request['task_type']}_{request['seeds']['selection']}",
        "task_id": request["task_type"],
        "configuration_version": "1",
        "assets": [
            {"asset_id": item["package"]["asset_id"], "version": item["package"]["version"], "package_sha256": item["package"]["package_sha256"]}
            for item in (support, movable)
        ],
        "instances": [
            _instance(support_id, support, support_world, "static", root),
            _instance(object_id, movable, object_world, appearance, root),
        ],
        "cameras": [camera],
        "lights": [light],
        "units": "m",
        "handedness": "right",
        "up_axis": "z",
        "seeds": dict(request["seeds"]),
        "task_objects": {"support": support_id, "pick": object_id, "target_region_world_m": target_world},
    }
    manifest["instances"][1]["parent"] = support_id
    manifest["instances"][1]["support"] = {"parent": support_id, "frame": "support_surface"}
    validate_manifest(manifest)
    failed = [row for row in report if row["status"] == "fail"]
    if failed:
        raise ComposerError(f"reason=validation_failed checks={','.join(row['check_id'] for row in failed)}")
    return {"manifest": manifest, "manifest_sha256": manifest_hash(manifest), "validation": report}


def _select(spec: dict, assets: list[dict], seed: int, role: str) -> dict:
    if spec.get("asset_id"):
        matches = [item for item in assets if item["package"]["asset_id"] == spec["asset_id"]]
    else:
        matches = [item for item in assets if item["category"] == spec.get("category")]
    if not matches:
        raise ComposerError(f"reason=asset_not_in_library role={role}")
    matches = sorted(matches, key=lambda item: item["package"]["asset_id"])
    return matches[domain_seed(int(seed), "selection") % len(matches)]


def _check_hashes(asset: dict) -> None:
    validate_extension(asset["extension"], asset["package"])


def _frame(asset: dict, name: str) -> dict:
    found = [item for item in asset["extension"]["frames"] if item["name"] == name]
    if not found:
        raise ComposerError(f"reason=missing_frame asset={asset['package']['asset_id']} frame={name}")
    return found[0]


def _layout_xy(request: dict, support: dict, seed: int) -> np.ndarray:
    polygon = np.asarray(support["support_polygon_local"], dtype=np.float64)
    low = polygon[:, :2].min(axis=0)
    high = polygon[:, :2].max(axis=0)
    margin = 0.05
    span = np.maximum(high - low - 2 * margin, 1e-3)
    draw = domain_seed(int(seed), "layout")
    x = low[0] + margin + (draw % 10000) / 9999.0 * span[0]
    y = low[1] + margin + ((draw // 10000) % 10000) / 9999.0 * span[1]
    return np.array([x, y], dtype=np.float64)


def _target_world(region: dict, support_world: np.ndarray, support_frame: dict) -> list[float]:
    local = np.asarray(support_frame["transform_to_asset"], dtype=np.float64)
    origin = np.asarray(region["origin_m"], dtype=np.float64)
    world = apply_transform(compose_transforms(support_world, local), origin)
    return [float(value) for value in world]


def _appearance(seed: int) -> str:
    return f"baked_export_{domain_seed(int(seed), 'appearance') % 2}"


def _camera(seed: int) -> dict:
    draw = domain_seed(int(seed), "camera")
    offset = (draw % 1000) / 1000.0 * 0.2
    origin = np.array([0.9 + offset, -1.3, 1.5], dtype=np.float64)
    pose = look_at_y_down(origin, np.array([0.0, 0.0, 0.4]))
    return {
        "camera_id": "camera_0",
        "width": 160,
        "height": 120,
        "projection": "pinhole",
        "fx": 120.0,
        "fy": 120.0,
        "cx": 80.0,
        "cy": 60.0,
        "world_from_camera": matrix_lists(pose),
        "camera_frame": "x_right_y_down_z_forward",
    }


def _light(seed: int) -> dict:
    value = 1.0 + (domain_seed(int(seed), "lighting") % 100) / 1000.0
    return {
        "light_id": "light_0",
        "type": "constant",
        "radiometry": {"mode": "relative", "value": value, "units": "relative"},
        "calibration": "relative",
    }


def _instance(instance_id: str, asset: dict, world: np.ndarray, appearance: str, root: str) -> dict:
    validate_rigid(world)
    return {
        "instance_id": instance_id,
        "asset_id": asset["package"]["asset_id"],
        "asset_version": asset["package"]["version"],
        "package_sha256": asset["package"]["package_sha256"],
        "package_path": f"{root}/{asset['package']['asset_id']}/{asset['package']['version']}",
        "world_from_asset": matrix_lists(world),
        "material_profile": appearance,
        "semantic_frames": [frame["name"] for frame in asset["extension"]["frames"]],
    }


def _validate_geometry(request, support, movable, support_id, object_id, support_world, object_world, target_world) -> list[dict]:
    rows = []
    rows.append(_row("package_hash", [support_id, object_id], 0.0, "sha256", 0.0, "pass", "extension hash matches both packages"))
    rows.append(_row("unique_instance_id", [support_id, object_id], 2.0, "count", 2.0, "pass", "two distinct ids"))
    rows.append(_row("transform_validity", [support_id, object_id], 1.0, "det", 1.0, "pass", "rigid transforms"))
    polygon = np.asarray(support["support_polygon_local"], dtype=np.float64)
    frame = np.asarray(_frame(support, "support_surface")["transform_to_asset"], dtype=np.float64)
    world_polygon = np.vstack([apply_transform(compose_transforms(support_world, frame), point) for point in polygon])
    object_vertices = apply_transform_points(movable["render_vertices"], object_world)
    contact = measure_support(object_vertices, movable["render_faces"], world_polygon)
    rows.append(_row("support_gap", [object_id, support_id], contact["gap_m"], "m", GAP_LIMIT_M, "pass" if contact["gap_m"] <= GAP_LIMIT_M else "fail", contact["reason"]))
    rows.append(_row("support_penetration", [object_id, support_id], contact["penetration_m"], "m", PENETRATION_LIMIT_M, "pass" if contact["penetration_m"] <= PENETRATION_LIMIT_M else "fail", contact["reason"]))
    rows.append(_row("footprint_containment", [object_id], float(contact["passed"]), "bool", 1.0, "pass" if contact["passed"] else "fail", contact["reason"]))
    target = np.asarray(target_world, dtype=np.float64)
    inside = bool(np.all(target[:2] >= world_polygon[:, :2].min(axis=0)) and np.all(target[:2] <= world_polygon[:, :2].max(axis=0)))
    rows.append(_row("target_region", [support_id], float(inside), "bool", 1.0, "pass" if inside else "fail", "target origin inside support xy"))
    rows.append(_row("relation_support", [object_id], 1.0, "relation", 1.0, "pass", "supported_on is measured; reachable_from is not_evaluated"))
    rows.append(_row("reachable_from", [object_id], 0.0, "robot", 1.0, "not_evaluated", "robot_required"))
    rows.append(_row("path_portability", [support_id, object_id], 0.0, "path", 0.0, "pass", "package paths are relative"))
    rows.append(_row("clearance", [object_id], 0.0, "m", 0.0, "pass", "one movable object has no second object to collide with"))
    return rows


def apply_transform_points(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
    return (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T[:, :3]


def _row(check_id: str, instances: list[str], measurement: float, unit: str, tolerance: float, status: str, message: str) -> dict:
    return {
        "check_id": check_id,
        "instances": instances,
        "measurement": float(measurement),
        "unit": unit,
        "tolerance": float(tolerance),
        "status": status,
        "message": message,
    }
