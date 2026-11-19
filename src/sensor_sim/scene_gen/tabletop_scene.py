"""Room-context tabletop scenes.

The room shell is the existing floor, ceiling, and four wall packages. Furniture
poses are explicit. Tabletop objects are pinned by asset id and placed by
``tabletop_layout``. Replay uses ``task_replay.replay`` and does not retrieve.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import canonical_json_bytes, verify_package
from sensor_sim.scene_gen.asset_validate import triangulate
from sensor_sim.scene_gen.indoor_contact import measure_support, unintended_intersection, world_vertices
from sensor_sim.scene_gen.indoor_transform import identity_matrix, serialize_matrix, translation_matrix
from sensor_sim.scene_gen.tabletop_layout import load_mesh, place_tabletop, support_pose
from sensor_sim.scene_gen.task_replay import replay
from sensor_sim.scene_gen.task_retrieve import catalog_snapshot_hash, embedding_backend
from sensor_sim.scene_gen.task_schema import SCHEMA_VERSION, document_hash, validate_bundle
from sensor_sim.scene_gen.task_size import accept_scale, mesh_bounds

logger = logging.getLogger("sensor_sim.scene_gen.tabletop_scene")

ROOT = Path("dataset/qualification/tabletop_scenes")
VERSION = "1.0.0"
SHELL = (
    "room_floor",
    "room_ceiling",
    "room_wall_south",
    "room_wall_north",
    "room_wall_west",
    "room_wall_east",
)
PRIMARY_SEEDS = (17101, 17102, 17103)
RESERVED_SEEDS = (17104, 17105, 17106)
COMPARISON_SEED = 17121
ROOM_CAMERA = {
    "name": "room_overview",
    "origin": [0.55, 0.48, 1.72],
    "target": [2.35, 2.15, 0.75],
    "up": [0.0, 0.0, 1.0],
    "fov_deg": 58.0,
    "resolution": [512, 384],
}
# Translations put the furniture origin on the floor. The floor support is z=0.05.
ARRANGEMENTS = {
    "south_open": {"table_rect": [1.50, 1.35, 0.05], "shelf_two": [2.90, 2.50, 0.05], "prop_chair": [0.45, 1.45, 0.05]},
    "east_work": {"table_rect": [0.70, 1.60, 0.05], "shelf_two": [2.80, 0.50, 0.05], "prop_chair": [2.10, 1.70, 0.05]},
    "west_shelf": {"table_rect": [1.90, 1.10, 0.05], "shelf_two": [0.40, 2.40, 0.05], "prop_chair": [3.30, 1.20, 0.05]},
}
RECIPES = {
    "tabletop_polarization_controlled": {
        "arrangement": "south_open",
        "separation": "controlled",
        "description": "Eight tabletop objects in a room, spaced for material and geometry inspection.",
        "objects": [
            ("prop_bowl__1", "prop_bowl", "left"),
            ("prop_bowl_glossy__1", "prop_bowl_glossy", "right"),
            ("prop_plate__1", "prop_plate", "left"),
            ("prop_bottle__1", "prop_bottle", "left"),
            ("prop_can__1", "prop_can", "right"),
            ("prop_can_rough__1", "prop_can_rough", "right"),
            ("prop_spoon__1", "prop_spoon", "left"),
            ("prop_phone__1", "prop_phone", "right"),
        ],
    },
    "tabletop_mixed_objects": {
        "arrangement": "east_work",
        "separation": "mixed",
        "description": "Ten tabletop geometry types in eating and work groupings inside the room.",
        "objects": [
            ("prop_bowl__1", "prop_bowl", "eating"),
            ("prop_plate__1", "prop_plate", "eating"),
            ("prop_bottle__1", "prop_bottle", "eating"),
            ("prop_spoon__1", "prop_spoon", "eating"),
            ("prop_apple__1", "prop_apple", "eating"),
            ("prop_can__1", "prop_can", "work"),
            ("prop_marker__1", "prop_marker", "work"),
            ("prop_phone__1", "prop_phone", "work"),
            ("prop_tape_roll__1", "prop_tape_roll", "work"),
            ("prop_small_box__1", "prop_small_box", "work"),
        ],
    },
    "tabletop_dense_clutter": {
        "arrangement": "west_shelf",
        "separation": "clutter",
        "description": "Sixteen tabletop instances in closer groupings. Instance count is not a density claim.",
        "objects": [
            ("prop_bowl__1", "prop_bowl", "left"),
            ("prop_bowl__2", "prop_bowl", "left"),
            ("prop_plate__1", "prop_plate", "left"),
            ("prop_bottle__1", "prop_bottle", "left"),
            ("prop_spoon__1", "prop_spoon", "left"),
            ("prop_spoon__2", "prop_spoon", "right"),
            ("prop_apple__1", "prop_apple", "left"),
            ("prop_apple__2", "prop_apple", "right"),
            ("prop_can__1", "prop_can", "right"),
            ("prop_can__2", "prop_can", "right"),
            ("prop_can_rough__1", "prop_can_rough", "right"),
            ("prop_marker__1", "prop_marker", "right"),
            ("prop_marker__2", "prop_marker", "left"),
            ("prop_tape_roll__1", "prop_tape_roll", "right"),
            ("prop_small_box__1", "prop_small_box", "left"),
            ("prop_phone__1", "prop_phone", "right"),
        ],
    },
}


def generate_pilot(catalog: Path, output: Path | None = None) -> list[dict]:
    catalog = Path(catalog)
    output = Path(output or ROOT)
    output.mkdir(parents=True, exist_ok=True)
    reserved = list(RESERVED_SEEDS)
    written = []
    for task_id, recipe in RECIPES.items():
        seeds = list(PRIMARY_SEEDS)
        produced = 0
        while produced < 3:
            if not seeds:
                raise RuntimeError(f"{task_id} exhausted placement seeds")
            seed = seeds.pop(0)
            result = _generate_recipe(catalog, output, task_id, recipe, seed)
            written.append(result)
            if result["valid"]:
                produced += 1
                logger.info("tabletop_scene event=pilot_saved task=%s seed=%d path=%s", task_id, seed, result["path"])
            else:
                nxt = reserved.pop(0)
                logger.info(
                    "tabletop_scene event=pilot_failed task=%s seed=%d next_seed=%d reason=%s",
                    task_id,
                    seed,
                    nxt,
                    result["failure"],
                )
                seeds.insert(0, nxt)
    written.extend(_comparison_bundles(catalog, output))
    return written


def generate_recipe(catalog: Path, output: Path, task_id: str, seed: int) -> dict:
    return _generate_recipe(catalog, output, task_id, RECIPES[task_id], seed)


def _generate_recipe(catalog: Path, output: Path, task_id: str, recipe: dict, seed: int) -> dict:
    room = _room(catalog, recipe["arrangement"])
    if not room["valid"]:
        path = _write_failure(output, task_id, seed, room)
        return {"valid": False, "path": str(path), "failure": room["failure"], "task_id": task_id, "seed": seed}
    placed = place_tabletop(
        catalog,
        room["table_matrix"],
        room["table_support_local"],
        _requests(recipe, room["clusters"]),
        seed=seed,
        minimum_separation_m=room["separation"][recipe["separation"]],
    )
    if not placed["valid"]:
        path = _write_failure(output, task_id, seed, placed)
        return {"valid": False, "path": str(path), "failure": placed["collision_log"]["termination"], "task_id": task_id, "seed": seed}
    bundle = _bundle(catalog, task_id, recipe["description"], seed, room, placed["poses"], placed["relations"], placed["collision_log"])
    path = _publish(output, task_id, bundle)
    replay(path, catalog)
    return {"valid": True, "path": str(path), "failure": "", "task_id": task_id, "seed": seed, "occupancy": placed["collision_log"]["occupancy"]}


def _comparison_bundles(catalog: Path, output: Path) -> list[dict]:
    room = _room(catalog, "south_open")
    if not room["valid"]:
        raise RuntimeError(f"comparison room failed: {room['failure']}")
    support = room["table_support_world"]
    center = (float(support[:, 0].mean()), float(support[:, 1].mean()))
    yaw = 0.4
    pairs = [
        ("comparison_bowl_diffuse", "prop_bowl", "prop_bowl"),
        ("comparison_bowl_glossy", "prop_bowl_glossy", "prop_bowl"),
        ("comparison_can_polished", "prop_can", "prop_can"),
        ("comparison_can_rough", "prop_can_rough", "prop_can"),
        ("comparison_geometry_plate", "prop_plate", "prop_plate"),
        ("comparison_geometry_apple", "prop_apple", "prop_apple"),
    ]
    written = []
    for task_id, asset_id, geometry_source in pairs:
        matrix = support_pose(room["table_matrix"], room["table_support_local"], yaw, center)
        mesh = load_mesh(catalog, asset_id)
        world = world_vertices(mesh["vertices"], matrix)
        contact = measure_support(world, mesh["faces"], support)
        if not contact["passed"]:
            raise RuntimeError(f"{task_id} comparison object is not supported")
        instance_id = f"{asset_id}__1"
        relation = {
            "type": "supported_on",
            "instances": [instance_id, "table_1"],
            "frame": "support_surface_local",
            "frame_instance": "table_1",
            "surface": "top",
            "priority": "required",
            "constraints": {},
            "passed": True,
            "reason": contact["reason"],
            "measured_m": float(contact["gap_m"]),
            "gap_limit_m": float(contact["gap_limit_m"]),
            "penetration_limit_m": float(contact["penetration_limit_m"]),
        }
        log = {
            "termination": "specified_pose",
            "comparison": True,
            "changed_asset": asset_id,
            "geometry_source": geometry_source,
            "yaw_rad": yaw,
            "support_xy_m": [center[0], center[1]],
            "occupancy": {"instance_count": 1, "note": "single-object pair; density is not claimed"},
        }
        description = f"Controlled comparison scene using {asset_id}. Only the named variable differs from its pair."
        bundle = _bundle(catalog, task_id, description, COMPARISON_SEED, room, {instance_id: serialize_matrix(matrix)}, [relation], log)
        path = _publish(output, task_id, bundle)
        replay(path, catalog)
        written.append({"valid": True, "path": str(path), "task_id": task_id, "seed": COMPARISON_SEED, "failure": ""})
        logger.info("tabletop_scene event=comparison_saved task=%s asset=%s path=%s", task_id, asset_id, path)
    return written


def _room(catalog: Path, arrangement: str) -> dict:
    from sensor_sim.scene_gen.tabletop_layout import SEPARATION

    poses = {asset_id: identity_matrix() for asset_id in SHELL}
    for asset_id, translation in ARRANGEMENTS[arrangement].items():
        poses[asset_id] = translation_matrix(translation)
    floor = verify_package(Path(catalog) / "room_floor" / VERSION)
    floor_support = np.asarray(floor["support_surfaces"][0]["polygon"], dtype=np.float64)
    table = verify_package(Path(catalog) / "table_rect" / VERSION)
    table_support = np.asarray(next(item["polygon"] for item in table["support_surfaces"] if item["id"] == "top"), dtype=np.float64)
    contacts = []
    valid = True
    failure = ""
    furniture = ("table_rect", "shelf_two", "prop_chair")
    for asset_id in furniture:
        mesh = load_mesh(catalog, asset_id)
        world = world_vertices(mesh["vertices"], poses[asset_id])
        contact = measure_support(world, mesh["faces"], floor_support)
        contacts.append({"instance_id": _furniture_id(asset_id), "asset_id": asset_id, "passed": contact["passed"], "reason": contact["reason"], "gap_m": contact["gap_m"]})
        if not contact["passed"]:
            valid = False
            failure = f"floor_support:{asset_id}:{contact['reason']}"
    bodies = []
    for asset_id, matrix in poses.items():
        mesh = load_mesh(catalog, asset_id)
        bodies.append((asset_id, {"vertices": world_vertices(mesh["vertices"], matrix), "faces": triangulate(mesh["polygons"])}))
    intersections = []
    for index, (left_id, left) in enumerate(bodies):
        for right_id, right in bodies[index + 1 :]:
            if _skip_shell_contact(left_id, right_id):
                continue
            found = unintended_intersection(left, right)
            if found["unintended"]:
                intersections.append({"pair": [left_id, right_id], "clearance_m": found["clearance_m"]})
                valid = False
                failure = failure or f"intersection:{left_id}:{right_id}"
    table_world = world_vertices(np.asarray(table_support, dtype=np.float64), poses["table_rect"])
    x0, y0 = float(table_world[:, 0].min()) + 0.015, float(table_world[:, 1].min()) + 0.015
    x1, y1 = float(table_world[:, 0].max()) - 0.015, float(table_world[:, 1].max()) - 0.015
    mid_x = 0.5 * (x0 + x1)
    clusters = {
        "left": (x0, y0, mid_x - 0.005, y1),
        "right": (mid_x + 0.005, y0, x1, y1),
        "eating": (x0, y0, mid_x - 0.005, y1),
        "work": (mid_x + 0.005, y0, x1, y1),
    }
    logger.info(
        "tabletop_scene event=room arrangement=%s valid=%s contacts=%d intersections=%d tabletop_z=%.4f",
        arrangement,
        valid,
        len(contacts),
        len(intersections),
        float(table_world[:, 2].mean()),
    )
    return {
        "valid": valid,
        "failure": failure,
        "arrangement": arrangement,
        "poses": {key: serialize_matrix(value) for key, value in poses.items()},
        "matrices": poses,
        "table_matrix": poses["table_rect"],
        "table_support_local": table_support,
        "table_support_world": table_world,
        "contacts": contacts,
        "intersections": intersections,
        "clusters": clusters,
        "separation": SEPARATION,
        "floor_support": floor_support,
    }


def _requests(recipe: dict, clusters: dict) -> list[dict]:
    requests = []
    for instance_id, asset_id, cluster_name in recipe["objects"]:
        requests.append(
            {
                "instance_id": instance_id,
                "asset_id": asset_id,
                "version": VERSION,
                "cluster": clusters[cluster_name],
                "cluster_name": cluster_name,
                "support_parent": "table_1",
                "support_surface": "top",
            }
        )
    return requests


def _bundle(catalog, task_id, description, seed, room, object_poses, object_relations, collision_log, cameras=None) -> dict:
    task = _task(task_id, description, seed, room, object_poses)
    selections = []
    poses = []
    relations = []
    for asset_id in SHELL:
        instance_id = asset_id
        selections.append(_selection(catalog, instance_id, asset_id, "contextual"))
        poses.append(_pose(instance_id, room["poses"][asset_id]))
    for asset_id in ("table_rect", "shelf_two", "prop_chair"):
        instance_id = _furniture_id(asset_id)
        selections.append(_selection(catalog, instance_id, asset_id, "contextual"))
        poses.append(_pose(instance_id, room["poses"][asset_id]))
        relations.append(
            {
                "type": "supported_on",
                "instances": [instance_id, "room_floor"],
                "frame": "support_surface_local",
                "frame_instance": "room_floor",
                "surface": "interior",
                "priority": "required",
                "constraints": {},
                "passed": True,
                "reason": "contact",
                "measured_m": 0.0,
            }
        )
    for instance_id, matrix in object_poses.items():
        asset_id = instance_id.rsplit("__", 1)[0]
        selections.append(_selection(catalog, instance_id, asset_id, "required"))
        poses.append(_pose(instance_id, matrix))
    relations.extend(object_relations)
    sizes = [accept_scale(mesh_bounds(catalog, item["asset_id"], item["version"]), [1.0, 1.0, 1.0]) for item in selections]
    backend = embedding_backend()
    collision_log = dict(collision_log)
    collision_log["room_arrangement"] = room["arrangement"]
    collision_log["room_contacts"] = room["contacts"]
    collision_log["room_intersections"] = room["intersections"]
    collision_log["cameras"] = cameras or {"room": ROOM_CAMERA, "tabletop": _table_camera(room["table_support_world"])}
    collision_log["geometry_counts"] = _geometry_counts(selections)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "task_sha256": document_hash(task),
        "query_template": {"version": backend["query_template"], "sha256": backend["query_template_sha256"], "mode": "explicit_asset_v1"},
        "retrieval": {
            "backend": backend,
            "catalog_sha256": catalog_snapshot_hash(catalog),
            "top_k": 0,
            "seed": int(seed),
            "rankings": [],
            "selection": "explicit_asset_v1",
        },
        "proposal": {"assets": selections, "sizes": sizes, "poses": poses, "relations": relations},
        "resolved": {
            "assets": selections,
            "sizes": sizes,
            "poses": poses,
            "relations": relations,
            "collision_log": collision_log,
        },
        "validation": {"valid": True, "visual_verification": "pending", "admitted": False},
    }
    validate_bundle(bundle)
    return bundle


def _task(task_id: str, description: str, seed: int, room: dict, object_poses: dict) -> dict:
    required = []
    for instance_id in object_poses:
        asset_id = instance_id.rsplit("__", 1)[0]
        required.append({"instance_id": instance_id, "query": asset_id.replace("_", " "), "affordances": ["rest"], "role": "required"})
    contextual = [{"instance_id": asset_id, "query": asset_id.replace("_", " "), "affordances": ["structure"], "role": "contextual"} for asset_id in SHELL]
    contextual.append({"instance_id": "table_1", "query": "table rect", "affordances": ["support"], "role": "contextual"})
    contextual.append({"instance_id": "shelf_1", "query": "shelf two", "affordances": ["support"], "role": "contextual"})
    contextual.append({"instance_id": "chair_1", "query": "chair", "affordances": ["rest"], "role": "contextual"})
    constraints = [
        {
            "type": "supported_on",
            "instances": [instance_id, "table_1"],
            "frame": "support_surface_local",
            "frame_instance": "table_1",
            "surface": "top",
            "priority": "required",
            "constraints": {},
        }
        for instance_id in object_poses
    ]
    return {
        "task_id": task_id,
        "name": description,
        "description": description,
        "category": "tabletop",
        "required_objects": required,
        "contextual_objects": contextual,
        "robot": None,
        "workspace_m": None,
        "constraints": constraints,
        "initial_requirements": ["rigid_pose", "explicit_asset_v1"],
        "goal_requirements": [],
        "seed": int(seed),
    }


def _selection(catalog: Path, instance_id: str, asset_id: str, role: str) -> dict:
    manifest = verify_package(Path(catalog) / asset_id / VERSION)
    index_status = _index_status(catalog, asset_id)
    return {
        "instance_id": instance_id,
        "semantic_name": asset_id.replace("_", " "),
        "category": str((manifest.get("geometry") or {}).get("category", asset_id)),
        "role": role,
        "asset_id": asset_id,
        "version": VERSION,
        "package_sha256": manifest["package_sha256"],
        "query": asset_id.replace("_", " "),
        "score": 1.0,
        "score_type": "explicit_asset_v1",
        "license_status": manifest["provenance"]["license_status"],
        "technical_status": "technically_validated",
        "renderer_status": index_status,
        "visual_verification": "pending",
    }


def _pose(instance_id: str, matrix: list[float]) -> dict:
    return {"instance_id": instance_id, "matrix": list(matrix), "scale": [1.0, 1.0, 1.0], "articulation": "not_applicable"}


def _index_status(catalog: Path, asset_id: str) -> str:
    from sensor_sim.scene_gen.asset_package import index_entry, read_index

    entry = index_entry(read_index(Path(catalog) / "index.json"), asset_id, VERSION)
    if entry is None:
        return "candidate"
    return str(entry["status"])


def _furniture_id(asset_id: str) -> str:
    return {"table_rect": "table_1", "shelf_two": "shelf_1", "prop_chair": "chair_1"}[asset_id]


def _skip_shell_contact(left: str, right: str) -> bool:
    """The qualified shell pieces meet on purpose. Furniture-to-floor contact is support, not a collision."""
    pair = {left, right}
    if pair <= set(SHELL):
        return True
    return "room_floor" in pair and bool(pair & {"table_rect", "shelf_two", "prop_chair"})


def _table_camera(support_world: np.ndarray) -> dict:
    """Front-elevated camera inside the room, south of the table and north of the wall."""
    center = support_world.mean(axis=0)
    target = [float(center[0]), float(center[1]), float(center[2]) + 0.08]
    near_y = float(support_world[:, 1].min())
    origin_y = min(max(0.55, near_y - 0.85), near_y - 0.28)
    origin_x = float(np.clip(center[0], 0.7, 3.5))
    origin = [origin_x, float(origin_y), 1.45]
    width = float(support_world[:, 0].max() - support_world[:, 0].min()) + 0.35
    distance = max(abs(target[1] - origin[1]), 0.45)
    fov = float(np.clip(2.0 * np.degrees(np.arctan((width * 0.5) / distance)) * 1.12, 48.0, 78.0))
    return {
        "name": "tabletop",
        "origin": origin,
        "target": target,
        "up": [0.0, 0.0, 1.0],
        "fov_deg": fov,
        "resolution": [1024, 768],
        "framing": "inside the room, south of the table, field of view set to cover the support polygon",
    }


def _geometry_counts(selections: list[dict]) -> dict:
    tabletop = [item for item in selections if item["role"] == "required"]
    geometries = []
    materials = []
    for item in tabletop:
        geometries.append(item["asset_id"].removesuffix("_glossy").removesuffix("_rough"))
        materials.append(item["asset_id"])
    return {
        "tabletop_instances": len(tabletop),
        "unique_geometry_ids": len(set(geometries)),
        "unique_asset_ids": len({item["asset_id"] for item in tabletop}),
        "repeated_asset_ids": sorted(asset for asset in set(item["asset_id"] for item in tabletop) if sum(row["asset_id"] == asset for row in tabletop) > 1),
    }


def _publish(output: Path, task_id: str, bundle: dict) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "task_sha256": bundle["task_sha256"],
        "backend": bundle["retrieval"]["backend"]["selected"],
        "seed": bundle["task"]["seed"],
        "top_k": bundle["retrieval"]["top_k"],
        "task_id": task_id,
    }
    run_id = hashlib.sha256(canonical_json_bytes(config)).hexdigest()[:16]
    destination = output / task_id / run_id
    scene = _scene(bundle)
    scene_bytes = canonical_json_bytes(scene)
    bundle_bytes = canonical_json_bytes(bundle)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (destination / "bundle.json").read_bytes() == bundle_bytes and (destination / "scene.json").read_bytes() == scene_bytes:
            return destination
        raise RuntimeError(f"bundle_exists path={destination}")
    staging = Path(tempfile.mkdtemp(prefix=".tabletop-scene-", dir=output))
    try:
        (staging / "bundle.json").write_bytes(bundle_bytes)
        (staging / "scene.json").write_bytes(scene_bytes)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _scene(bundle: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": bundle["task"]["task_id"],
        "task_sha256": bundle["task_sha256"],
        "qualification": True,
        "visual_review": "pending",
        "admitted": False,
        "seed": bundle["task"]["seed"],
        "instances": [
            {
                "instance_id": asset["instance_id"],
                "asset_id": asset["asset_id"],
                "version": asset["version"],
                "package_sha256": asset["package_sha256"],
                "matrix": next(item["matrix"] for item in bundle["resolved"]["poses"] if item["instance_id"] == asset["instance_id"]),
            }
            for asset in bundle["resolved"]["assets"]
        ],
        "relations": bundle["resolved"]["relations"],
        "valid": bundle["validation"]["valid"],
    }


def _write_failure(output: Path, task_id: str, seed: int, payload: dict) -> Path:
    destination = Path(output) / task_id / f"failed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    record = {"task_id": task_id, "seed": seed, "valid": False, "failure": payload.get("failure") or payload.get("collision_log", {})}
    (destination / "failure.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination

