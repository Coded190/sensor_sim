"""Deterministic placement on top of the existing contact checks.

Corrections stay in the support region or the container interior. A required
constraint that cannot be met marks the scene invalid.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_validate import triangulate
from sensor_sim.scene_gen.indoor_contact import (
    MAX_PENETRATION_M,
    MAX_SUPPORT_GAP_M,
    measure_support,
    support_graph_ok,
    unintended_intersection,
    world_vertices,
)
from sensor_sim.scene_gen.indoor_transform import identity_matrix, serialize_matrix, translation_matrix
from sensor_sim.scene_gen.mesh_io import read_obj
from sensor_sim.scene_gen.narrow import mesh_clearance

logger = logging.getLogger("sensor_sim.scene_gen.task_layout")

SOLVER_VERSION = "task_layout_1"
MAX_ITERATIONS = 40
STEP_M = 0.02


def resolve(catalog: Path, selections: list[dict], relations: list[dict], *, seed: int) -> dict:
    catalog = Path(catalog)
    meshes = {item["instance_id"]: _load(catalog, item) for item in selections}
    proposed = _propose(meshes, relations, seed)
    resolved = {key: value.copy() for key, value in proposed.items()}
    log = {
        "solver_version": SOLVER_VERSION,
        "seed": int(seed),
        "step_m": STEP_M,
        "max_iterations": MAX_ITERATIONS,
        "initial_matrices": {key: serialize_matrix(value) for key, value in proposed.items()},
        "corrections": [],
        "termination": "",
    }
    relations_out = []
    for iteration in range(MAX_ITERATIONS):
        relations_out = [_evaluate(meshes, resolved, relation) for relation in relations]
        graph = [(relation["instances"][1], relation["instances"][0]) for relation in relations_out if relation["type"] == "supported_on"]
        if graph and not support_graph_ok(graph):
            log["termination"] = "cycle"
            return _done(resolved, relations_out, log, False)
        pairs = _collision_pairs(meshes, resolved, relations)
        blocked = [pair for pair in pairs if pair["unintended"]]
        if all(item["passed"] for item in relations_out if item["priority"] == "required") and not blocked:
            log["termination"] = "converged"
            log["iterations"] = iteration
            return _done(resolved, relations_out, log, True)
        if not blocked:
            log["termination"] = "constraint"
            log["iterations"] = iteration
            return _done(resolved, relations_out, log, False)
        pair = blocked[0]
        moved = _slide(meshes, resolved, pair["second"], relations)
        log["corrections"].append({"iteration": iteration, "pair": [pair["first"], pair["second"]], "moved": moved, "clearance_m": pair["clearance_m"]})
        if not moved:
            log["termination"] = "no_valid_correction"
            log["iterations"] = iteration + 1
            relations_out = [_evaluate(meshes, resolved, relation) for relation in relations]
            return _done(resolved, relations_out, log, False)
    log["termination"] = "iteration_limit"
    log["iterations"] = MAX_ITERATIONS
    return _done(resolved, relations_out, log, False)


def _propose(meshes: dict, relations: list[dict], seed: int) -> dict[str, np.ndarray]:
    poses = {key: identity_matrix() for key in meshes}
    nudge = (int(seed) % 5) * 0.01
    for relation in relations:
        if relation["type"] != "supported_on":
            continue
        child, parent = relation["instances"]
        polygon = _surface(meshes[parent], relation["surface"], poses[parent])
        low = polygon.min(axis=0)
        height = float(polygon[0][2]) - float(meshes[child]["vertices"].min(axis=0)[2])
        poses[child] = translation_matrix([float(low[0]) + 0.02 + nudge, float(low[1]) + 0.02, height])
    for relation in relations:
        if relation["type"] != "inside":
            continue
        child, parent = relation["instances"]
        interior = meshes[parent]["manifest"]["geometry"]["interior"]
        origin = interior["origin_m"]
        margin = float(interior["margin_m"])
        local = translation_matrix([float(origin[0]) + margin, float(origin[1]) + margin, float(interior["bottom_z_m"])])
        poses[child] = poses[parent] @ local
    for relation in relations:
        if relation["type"] == "near":
            child = relation["instances"][0]
            current = poses[child][:3, 3]
            minimum = float(relation["constraints"]["minimum_m"])
            poses[child] = translation_matrix([float(current[0]) + minimum + 0.01, float(current[1]), float(current[2])])
    return poses


def _evaluate(meshes, poses, relation: dict) -> dict:
    result = dict(relation)
    result["passed"] = False
    result["measured_m"] = None
    kind = relation["type"]
    if kind == "supported_on":
        child, parent = relation["instances"]
        polygon = _surface(meshes[parent], relation["surface"], poses[parent])
        world = world_vertices(meshes[child]["vertices"], poses[child])
        contact = measure_support(world, meshes[child]["polygons"], polygon)
        result["measured_m"] = contact["gap_m"]
        result["passed"] = bool(contact["passed"])
        result["reason"] = contact["reason"]
        result["gap_limit_m"] = MAX_SUPPORT_GAP_M
        result["penetration_limit_m"] = MAX_PENETRATION_M
    elif kind == "inside":
        child, parent = relation["instances"]
        result["passed"] = _inside(meshes[child], poses[child], meshes[parent])
        result["reason"] = "inside" if result["passed"] else "outside_interior"
    elif kind in {"near", "adjacent_to"}:
        left, right = relation["instances"]
        distance = _distance(meshes[left], poses[left], meshes[right], poses[right])
        limits = relation["constraints"]
        result["measured_m"] = distance
        result["passed"] = bool(float(limits["minimum_m"]) <= distance <= float(limits["maximum_m"]))
        result["reason"] = "distance"
    elif kind == "keep_clear_of":
        left, right = relation["instances"]
        distance = _distance(meshes[left], poses[left], meshes[right], poses[right])
        result["measured_m"] = distance
        result["passed"] = bool(distance + 1e-9 >= float(relation["constraints"]["minimum_m"]))
        result["reason"] = "clearance"
    elif kind == "reachable_from":
        result["passed"] = False
        result["reason"] = "robot_required"
    elif kind == "task_target":
        result["passed"] = True
        result["reason"] = "semantic_reference"
    elif kind in {"attached_to", "aligned_with", "facing"}:
        result["passed"] = False
        result["reason"] = "unsupported_numeric_relation"
    else:
        result["passed"] = False
        result["reason"] = "unsupported_relation"
    return result


def _inside(mesh, pose, container) -> bool:
    interior = container["manifest"]["geometry"].get("interior")
    if not interior:
        return False
    world = world_vertices(mesh["vertices"], pose)
    origin = np.asarray(interior["origin_m"], dtype=np.float64)
    size = np.asarray(interior["size_m"], dtype=np.float64)
    margin = float(interior["margin_m"])
    low = origin + np.array([margin, margin, 0.0])
    high = origin + size - np.array([margin, margin, 0.0])
    high[2] = origin[2] + size[2] - margin
    return bool(np.all(world >= low - 1e-9) and np.all(world <= high + 1e-9))


def _collision_pairs(meshes, poses, relations) -> list[dict]:
    exempt = set()
    for relation in relations:
        if relation["type"] in {"supported_on", "inside"}:
            exempt.add(tuple(sorted(relation["instances"])))
    names = sorted(meshes)
    pairs = []
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if tuple(sorted((first, second))) in exempt:
                continue
            left = {"vertices": world_vertices(meshes[first]["vertices"], poses[first]), "faces": triangulate(meshes[first]["polygons"])}
            right = {"vertices": world_vertices(meshes[second]["vertices"], poses[second]), "faces": triangulate(meshes[second]["polygons"])}
            found = unintended_intersection(left, right)
            found["first"] = first
            found["second"] = second
            pairs.append(found)
    pairs.sort(key=lambda item: (item["first"], item["second"]))
    return pairs


def _slide(meshes, poses, instance_id, relations) -> bool:
    support = next((relation for relation in relations if relation["type"] == "supported_on" and relation["instances"][0] == instance_id), None)
    if support is None:
        return False
    parent = support["instances"][1]
    polygon = _surface(meshes[parent], support["surface"], poses[parent])
    trial = poses[instance_id].copy()
    trial[:3, 3] = trial[:3, 3] + np.array([STEP_M, 0.0, 0.0])
    world = world_vertices(meshes[instance_id]["vertices"], trial)
    contact = measure_support(world, meshes[instance_id]["polygons"], polygon)
    if not contact["passed"]:
        return False
    poses[instance_id] = trial
    return True


def _surface(mesh, surface_id, pose) -> np.ndarray:
    for surface in mesh["manifest"].get("support_surfaces") or []:
        if surface["id"] == surface_id:
            return world_vertices(np.asarray(surface["polygon"], dtype=np.float64), pose)
    raise KeyError(surface_id)


def _distance(left, left_pose, right, right_pose) -> float:
    first = {"vertices": world_vertices(left["vertices"], left_pose), "faces": triangulate(left["polygons"])}
    second = {"vertices": world_vertices(right["vertices"], right_pose), "faces": triangulate(right["polygons"])}
    return float(mesh_clearance(first, second)["clearance_m"])


def _load(catalog: Path, selection: dict) -> dict:
    from sensor_sim.scene_gen.asset_package import verify_package

    root = catalog / selection["asset_id"] / selection["version"]
    manifest = verify_package(root)
    vertices, polygons = read_obj(root / "geometry" / "render.obj")
    return {"vertices": vertices, "polygons": polygons, "manifest": manifest, "selection": selection}


def _done(poses, relations, log, valid: bool) -> dict:
    logger.info("task_layout event=resolve valid=%s termination=%s corrections=%d", valid, log["termination"], len(log["corrections"]))
    return {
        "valid": valid,
        "poses": {key: serialize_matrix(value) for key, value in poses.items()},
        "relations": relations,
        "collision_log": log,
    }
