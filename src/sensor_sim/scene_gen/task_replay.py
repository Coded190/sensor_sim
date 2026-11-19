"""Build and replay task bundles without a second retrieval pass.

Generation is offline. Query text comes from ``deterministic_local_template_v1``.
Replay reads the saved selection and does not rank the catalog again.
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
from sensor_sim.scene_gen.indoor_transform import deserialize_matrix
from sensor_sim.scene_gen.mesh_io import read_obj
from sensor_sim.scene_gen.task_layout import resolve
from sensor_sim.scene_gen.task_retrieve import catalog_snapshot_hash, embedding_backend, retrieve
from sensor_sim.scene_gen.task_schema import SCHEMA_VERSION, TaskError, document_hash, validate_bundle, validate_task
from sensor_sim.scene_gen.task_size import accept_scale, fits_interior, fits_support, mesh_bounds

logger = logging.getLogger("sensor_sim.scene_gen.task_replay")

ROOT = Path("dataset/qualification/task_scenes")


def generate(task: dict, catalog: Path, output: Path | None = None, *, top_k: int = 10, seed: int | None = None) -> dict:
    validate_task({"schema_version": SCHEMA_VERSION, "task": task})
    catalog = Path(catalog)
    output = Path(output or ROOT)
    if task["task_id"] == "reachable_without_robot" or any(item.get("type") == "reachable_from" for item in task.get("constraints", [])):
        raise TaskError("reason=robot_required")
    seed = int(task["seed"] if seed is None else seed)
    backend = embedding_backend()
    rankings = []
    selections = []
    for spec in [*task["required_objects"], *task["contextual_objects"]]:
        found = retrieve(catalog, spec["query"], top_k=top_k, require_renderer=True)
        rankings.append(found)
        chosen = _choose(catalog, found["results"], spec["affordances"])
        if chosen is None:
            raise TaskError(f"reason=no_match instance={spec['instance_id']} query={spec['query']}")
        chosen.update({"instance_id": spec["instance_id"], "semantic_name": spec["query"], "category": spec["query"], "role": spec["role"], "query": spec["query"]})
        selections.append(chosen)
    sized = []
    for selection in selections:
        bounds = mesh_bounds(catalog, selection["asset_id"], selection["version"])
        sized.append(accept_scale(bounds, [1.0, 1.0, 1.0]))
    relations = _relations_for(task, selections)
    _check_sizes(catalog, selections, sized, relations)
    proposal_relations = [dict(item) for item in relations]
    solved = resolve(catalog, selections, relations, seed=seed)
    if task.get("workspace_m") and solved["valid"]:
        if not _inside_workspace(catalog, selections, solved["poses"], task["workspace_m"]):
            solved["valid"] = False
            solved["collision_log"]["termination"] = "workspace"
    bundle = _bundle(task, backend, rankings, selections, sized, proposal_relations, solved, catalog, seed, top_k)
    validate_bundle(bundle)
    path = _publish(output, task["task_id"], bundle)
    logger.info("task_replay event=generate task_id=%s valid=%s path=%s", task["task_id"], solved["valid"], path)
    return {"path": str(path), "valid": solved["valid"], "bundle": bundle}


def replay(bundle_dir: Path, catalog: Path) -> dict:
    """Rebuild the resolved manifest from the saved bundle. No retrieval and no model call."""
    bundle_dir = Path(bundle_dir)
    catalog = Path(catalog)
    original = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    validate_bundle(original)
    for asset in original["resolved"]["assets"]:
        manifest = verify_package(catalog / asset["asset_id"] / asset["version"])
        if manifest["package_sha256"] != asset["package_sha256"]:
            raise TaskError(f"reason=package_hash asset={asset['asset_id']}")
    scene = _scene(original)
    if canonical_json_bytes(scene) != (bundle_dir / "scene.json").read_bytes():
        raise TaskError("reason=replay_mismatch")
    return {"valid": original["validation"]["valid"], "scene": scene}


def _choose(catalog: Path, results: list[dict], affordances: list[str]) -> dict | None:
    for result in results:
        manifest = verify_package(catalog / result["asset_id"] / result["version"])
        if "contain" in affordances and "interior" not in (manifest.get("geometry") or {}):
            continue
        if "support" in affordances and not manifest.get("support_surfaces"):
            continue
        return dict(result)
    return None


def _relations_for(task: dict, selections: list[dict]) -> list[dict]:
    by_query = {}
    for item in selections:
        by_query.setdefault(item["query"], item["instance_id"])
    relations = []
    for constraint in task["constraints"]:
        relation = dict(constraint)
        relation["instances"] = [by_query.get(name, name) for name in constraint["instances"]]
        relations.append(relation)
    return relations


def _check_sizes(catalog, selections, sized, relations) -> None:
    by_id = {item["instance_id"]: item for item in selections}
    for relation in relations:
        if relation["type"] != "inside":
            continue
        child, parent = relation["instances"]
        parent_manifest = verify_package(catalog / by_id[parent]["asset_id"] / by_id[parent]["version"])
        interior = (parent_manifest.get("geometry") or {}).get("interior")
        child_size = next(item["accepted_size_m"] for item, selection in zip(sized, selections) if selection["instance_id"] == child)
        fit = fits_interior(child_size, interior)
        if not fit["fits"]:
            raise TaskError("reason=interior")
        parent_mesh = parent_manifest.get("support_surfaces") or []
        surface = next(item for item in parent_mesh if item["id"] == "interior_bottom")
        if not fits_support(child_size[:2], np.asarray(surface["polygon"], dtype=np.float64)):
            raise TaskError("reason=support_fit")


def _inside_workspace(catalog, selections, poses, workspace: dict) -> bool:
    low = np.asarray(workspace["minimum_m"], dtype=np.float64)
    high = np.asarray(workspace["maximum_m"], dtype=np.float64)
    for selection in selections:
        vertices, _polygons = read_obj(catalog / selection["asset_id"] / selection["version"] / "geometry" / "render.obj")
        matrix = deserialize_matrix(poses[selection["instance_id"]])
        homogeneous = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
        world = (matrix @ homogeneous.T).T[:, :3]
        if np.any(world < low - 1e-8) or np.any(world > high + 1e-8):
            return False
    return True


def _bundle(task, backend, rankings, selections, sized, proposal_relations, solved, catalog, seed, top_k) -> dict:
    proposal_assets = [dict(item) for item in selections]
    resolved_assets = [dict(item) for item in selections]
    proposal_poses = [{"instance_id": key, "matrix": value, "scale": [1.0, 1.0, 1.0], "articulation": "not_applicable"} for key, value in solved["collision_log"]["initial_matrices"].items()]
    resolved_poses = [{"instance_id": key, "matrix": value, "scale": [1.0, 1.0, 1.0], "articulation": "not_applicable"} for key, value in solved["poses"].items()]
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "task_sha256": document_hash(task),
        "query_template": {"version": backend["query_template"], "sha256": backend["query_template_sha256"], "mode": "deterministic_local_template_v1"},
        "retrieval": {
            "backend": backend,
            "catalog_sha256": catalog_snapshot_hash(catalog),
            "top_k": top_k,
            "seed": seed,
            "rankings": rankings,
        },
        "proposal": {"assets": proposal_assets, "sizes": sized, "poses": proposal_poses, "relations": proposal_relations},
        "resolved": {
            "assets": resolved_assets,
            "sizes": sized,
            "poses": resolved_poses,
            "relations": solved["relations"],
            "collision_log": solved["collision_log"],
        },
        "validation": {"valid": solved["valid"], "visual_verification": "pending", "admitted": False},
    }


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


def _publish(output: Path, task_id: str, bundle: dict) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    config = {"task_sha256": bundle["task_sha256"], "backend": bundle["retrieval"]["backend"]["selected"], "seed": bundle["task"]["seed"], "top_k": bundle["retrieval"]["top_k"]}
    run_id = hashlib.sha256(canonical_json_bytes(config)).hexdigest()[:16]
    destination = output / task_id / run_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    scene_bytes = canonical_json_bytes(_scene(bundle))
    bundle_bytes = canonical_json_bytes(bundle)
    if destination.exists():
        if (destination / "bundle.json").read_bytes() == bundle_bytes and (destination / "scene.json").read_bytes() == scene_bytes:
            return destination
        raise TaskError("reason=bundle_exists")
    staging = Path(tempfile.mkdtemp(prefix=".task-", dir=output))
    try:
        (staging / "bundle.json").write_bytes(bundle_bytes)
        (staging / "scene.json").write_bytes(scene_bytes)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
