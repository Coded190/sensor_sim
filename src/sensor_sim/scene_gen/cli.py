"""Command line for one scene or a small batch.

``--num-scenes`` is the number of samples that pass validation. Failed seeds
are recorded and do not count.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from sensor_sim.scene_gen.dataset import record_failure
from sensor_sim.scene_gen.dataset_contract import PRODUCTION_DATASET_ID, refuse_production_generation
from sensor_sim.scene_gen.generate import child_seeds, generate_sample, prototype_config, random_config
from sensor_sim.scene_gen.run import (
    expand_schedule,
    manifest_path,
    resume_manifest,
    sample_id,
    sample_is_valid,
    save_manifest,
    stream_seed,
)
from sensor_sim.scene_gen.splits import write_splits

logger = logging.getLogger("sensor_sim.scene_gen")


def _load(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return data


def _build_config(spec: dict, seed: int, render_seed: int):
    if spec.get("mode") == "prototype":
        config = prototype_config(seed, tuple(spec.get("resolution", [128, 128])), int(spec.get("spp", 16)))
        if "segments" in spec:
            config["objects"][0]["segments"] = int(spec["segments"])
            config["objects"][0]["rings"] = int(spec["rings"])
    else:
        spec = dict(spec)
        spec["seed"] = seed
        config = random_config(spec, seed)
        config["resolution"] = [int(value) for value in spec.get("resolution", config["resolution"])]
        config["spp"] = int(spec.get("spp", config["spp"]))
    config["render_seed"] = int(render_seed)
    config["config_hash"] = spec.get("config_hash")
    return config


def _refuse_production_output(spec: dict, output: Path) -> None:
    """Qualification runs cannot publish into the Dataset v1 directory."""
    production = (Path.cwd() / "dataset" / "dataset_v1").resolve()
    resolved = Path(output).resolve()
    inside = resolved == production or production in resolved.parents
    if inside and spec.get("dataset_id") != PRODUCTION_DATASET_ID:
        raise RuntimeError("refusing to write a qualification run into the Dataset v1 production directory")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a Blender scene and render a Stokes sample")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-scenes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("dataset/samples"))
    parser.add_argument("--resolution", type=int, nargs=2)
    parser.add_argument("--spp", type=int)
    parser.add_argument("--split-mode", default=None)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    spec = _load(args.config)
    file_resolution = list(spec.get("resolution") or [])
    file_spp = spec.get("spp")
    if args.resolution is not None:
        spec["resolution"] = [int(args.resolution[0]), int(args.resolution[1])]
    if args.spp is not None:
        spec["spp"] = int(args.spp)
    from sensor_sim.scene_gen.production_schedule import bind_schedule

    if spec.get("dataset_id") == PRODUCTION_DATASET_ID:
        target = spec.get("successful_scenes")
        overridden = args.resolution is not None or args.spp is not None
        count_override = isinstance(target, int) and int(args.num_scenes) != int(target)
        if overridden or count_override or list(spec.get("resolution") or []) != file_resolution or spec.get("spp") != file_spp:
            raise RuntimeError("refusing Dataset v1 generation before export: scene-count or render override")
    refuse_production_generation(spec)
    bind_schedule(spec)
    _refuse_production_output(spec, args.output)
    base = int(spec.get("seed", 0) if args.seed is None else args.seed)
    profiles = {}
    profile_path = spec.get("profile_file")
    if profile_path:
        profiles = json.loads(Path(profile_path).read_text())
    spec["profiles"] = profiles
    requested = int(args.num_scenes)
    manifest_file = manifest_path(args.output)
    manifest = resume_manifest(
        manifest_file,
        spec,
        requested=requested,
        base_seed=base,
        split_mode=args.split_mode,
        split_seed=int(args.split_seed),
        sample_root=args.output,
    )
    spec["config_hash"] = manifest["config_hash"]
    completed = list(manifest["completed_ids"])
    schedule = expand_schedule(spec.get("strata"), requested)
    manifest["schedule"] = schedule
    manifest.setdefault("stratum_success", {})
    manifest.setdefault("stratum_failures", {})
    budget = int(args.max_attempts) if args.max_attempts is not None else 8 * max(requested - len(completed), 0)
    attempt = 0
    while len(completed) < requested and attempt < budget:
        slot = len(completed)
        stratum = schedule[slot]
        scene_seed = stream_seed(base, slot, attempt, "scene")
        render_seed = stream_seed(base, slot, attempt, "render")
        geometry_seed, material_seed = child_seeds(scene_seed)
        attempt += 1
        manifest["attempted_seeds"].append(int(scene_seed))
        scene_id = sample_id(manifest["run_id"], slot)
        draw = dict(spec)
        draw["scene_id"] = scene_id
        if stratum is not None:
            draw["forced_family"] = stratum.get("family")
            draw["forced_material"] = stratum.get("material")
            if stratum.get("preset") is not None:
                draw["forced_preset"] = stratum["preset"]
            if stratum.get("objects") is not None:
                draw["forced_objects"] = stratum["objects"]
            if stratum.get("require_partial_occlusion"):
                draw["require_partial_occlusion"] = True
            if stratum.get("count_objects") is not None:
                draw["forced_count"] = int(stratum["count_objects"])
            if stratum.get("repeated_asset"):
                draw["repeated_asset"] = True
            if stratum.get("require_sample_supervision") or spec.get("require_sample_supervision"):
                draw["require_sample_supervision"] = True
        failure = {
            "run_id": manifest["run_id"],
            "sample_slot": slot,
            "attempt_index": attempt - 1,
            "scene_seed": int(scene_seed),
            "geometry_seed": int(geometry_seed),
            "material_seed": int(material_seed),
            "render_seed": int(render_seed),
            "stratum": stratum,
            "failure_stage": "build",
        }
        try:
            config = _build_config(draw, scene_seed, render_seed)
            if spec.get("dataset_id") == PRODUCTION_DATASET_ID:
                config["dataset_contract"] = {
                    key: value for key, value in spec.items() if key not in ("profiles", "dataset_contract")
                }
                config["dataset_id"] = PRODUCTION_DATASET_ID
                config["generation_authorized"] = spec.get("generation_authorized")
            config["scene_id"] = scene_id
            config["run_id"] = manifest["run_id"]
            config["sample_slot"] = slot
            config["attempt_index"] = attempt - 1
            config["stratum"] = stratum
            failure["geometry_seed"] = int(config.get("geometry_seed", geometry_seed))
            failure["material_seed"] = int(config.get("material_seed", material_seed))
            scene_id = config["scene_id"]
            path = generate_sample(config, args.output)
        except RuntimeError as exc:
            failure["failure_stage"] = "generate"
            failure["exception_type"] = type(exc).__name__
            key = json.dumps(stratum, sort_keys=True, default=str)
            manifest["stratum_failures"][key] = manifest["stratum_failures"].get(key, 0) + 1
            record_failure(args.output.parent / "failures", scene_id, str(exc), **failure)
            manifest["failures"].append({**failure, "scene_id": scene_id, "error": str(exc)})
            logger.error("scene_gen seed=%s status=failed error=%s", scene_seed, exc)
            save_manifest(manifest_file, manifest)
            continue
        if not sample_is_valid(path, manifest["config_hash"]):
            message = f"{path} failed validation after publish"
            failure["failure_stage"] = "validate"
            failure["exception_type"] = "RuntimeError"
            record_failure(args.output.parent / "failures", scene_id, message, **failure)
            manifest["failures"].append({**failure, "scene_id": scene_id, "error": message})
            logger.error("scene_gen seed=%s status=failed error=%s", scene_seed, message)
            save_manifest(manifest_file, manifest)
            continue
        if scene_id not in completed:
            completed.append(scene_id)
        key = json.dumps(stratum, sort_keys=True, default=str)
        manifest["stratum_success"][key] = manifest["stratum_success"].get(key, 0) + 1
        manifest["completed_ids"] = completed
        manifest["remaining_quota"] = requested - len(completed)
        save_manifest(manifest_file, manifest)
        logger.info("scene_gen status=counted completed=%d requested=%d attempts=%d", len(completed), requested, attempt)
    manifest["completed_ids"] = completed
    if len(completed) < requested:
        manifest["status"] = "incomplete"
        save_manifest(manifest_file, manifest)
        logger.error(
            "scene_gen status=incomplete completed=%d requested=%d attempts=%d",
            len(completed),
            requested,
            attempt,
        )
        sys.exit(1)
    manifest["status"] = "complete"
    manifest["completed"] = manifest.get("completed") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    save_manifest(manifest_file, manifest)
    if args.split_mode:
        records = []
        for scene_id in completed:
            metadata = json.loads((args.output / scene_id / "metadata.json").read_text())
            records.append({"scene_id": metadata["scene_id"], "objects": metadata["objects"]})
        write_splits(records, args.split_mode, int(args.split_seed), args.output.parent / "splits")


if __name__ == "__main__":
    main()
