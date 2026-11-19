"""Closed-loop evaluation. Every 5 Hz step re-renders the filmed scene.

Success rate is the fraction of episodes whose task predicate passes. The
headline effect is the paired difference, same policy and seed, second-image
checkpoint minus the RGB checkpoint. The cone is reported on its own.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from sensor_sim.film.sfp import OUTPUT, SEED
from sensor_sim.film.shape_mesh import shape_catalog
from sensor_sim.policy.arm import ArmEnv
from sensor_sim.policy.collect import _record, choose_distance
from sensor_sim.policy.observe import _apply, modality_images, pose_matrix, render_measurement
from sensor_sim.policy.stokes_checkpoint import load_stokes
from sensor_sim.policy.tasks import scripted_episode, success
from sensor_sim.policy.train import CKPT, load_index, load_policy

logger = logging.getLogger("sensor_sim.policy")

REPORT = OUTPUT / "policy" / "results.json"
TASKS = ("slot", "tip", "lift")


def _tensor(image: np.ndarray, mean: np.ndarray, std: np.ndarray, device: str) -> torch.Tensor:
    scaled = (image.astype(np.float32) - mean) / std
    return torch.tensor(np.transpose(scaled, (2, 0, 1))[None], dtype=torch.float32, device=device)


def rollout(task: str, record: dict, xy: tuple[float, float], yaw: float, model, stats: dict, stokes, mean, std, device: str) -> bool:
    """One episode. The policy sees a fresh Mitsuba frame after every action."""
    radius = float(record["radius_m"])
    tilt = 1.2 if task == "tip" else 0.0
    env = ArmEnv()
    env.reset(xy, yaw, tilt, radius)
    # The camera distance is chosen on the expert's motion so the film stays in frame.
    preview = scripted_episode(task, record["family"], xy, yaw, radius)
    camera, _fraction = choose_distance(record, preview)
    env.reset(xy, yaw, tilt, radius)
    poses = [env.object_pose()]
    modality = model_modality(model)
    for _step in range(40):
        pose = env.object_pose()
        matrix = pose_matrix(record["floor_xy"], record["yaw_rad"], pose["position"][:2], pose["yaw"], pose["tilt"], radius)
        measurement = render_measurement(
            record["mesh_dir"],
            matrix,
            env.link_segments(),
            camera,
            _apply(matrix, record["probe"]),
            fallback=np.asarray(pose["position"], dtype=np.float64),
        )
        if modality == "rgb":
            from sensor_sim.policy.observe import downsample

            images = {"rgb": downsample(measurement["rgb"]).astype(np.float32)}
        else:
            images = modality_images(measurement, stokes, mean, std, device)
        rgb = _tensor(np.asarray(images["rgb"], dtype=np.float32), stats["rgb_mean"], stats["rgb_std"], device)
        extra = None
        if modality != "rgb":
            extra = _tensor(np.asarray(images[modality], dtype=np.float32), stats["extra_mean"], stats["extra_std"], device)
        state = torch.tensor(env.state()[None].astype(np.float32), dtype=torch.float32, device=device)
        with torch.no_grad():
            action = model.act(rgb, state, extra)[0].detach().cpu().numpy()
        action = action * stats["action_std"] + stats["action_mean"]
        env.step(action)
        poses.append(env.object_pose())
    return success(task, record["family"], radius, poses)


def model_modality(model) -> str:
    return "rgb" if model.extra is None else getattr(model, "modality_name", "normals")


def _validation_success(model, stats, stokes, mean, std, device: str) -> float:
    """Closed-loop success on the five validation episodes of each task. Those shapes were in the training pool."""
    flags = []
    for item in load_index():
        if item["split"] != "val":
            continue
        record = _record("train", item["folder"])
        flags.append(
            rollout(item["task"], record, tuple(item["xy"]), float(item["yaw"]), model, stats, stokes, mean, std, device)
        )
        logger.info(
            "policy stage=val checkpoint=%s task=%s folder=%s success=%s",
            model.modality_name,
            item["task"],
            item["folder"],
            flags[-1],
        )
    return float(np.mean(flags))


def evaluate(device: str = "cuda", kinds: set[str] | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    stokes, mean, std = load_stokes(device)
    catalog = {spec["folder"]: spec for spec in shape_catalog(SEED)}
    test = [_record("test", spec["folder"]) for spec in shape_catalog(SEED) if spec["split"] == "test"]
    rng = np.random.default_rng(1)
    cases = []
    for record in test:
        for task in TASKS:
            cases.append(
                {
                    "task": task,
                    "record": record,
                    "xy": (float(rng.uniform(-0.04, 0.04)), float(0.02 + rng.uniform(-0.04, 0.04))),
                    "yaw": float(rng.uniform(0.0, 2.0 * np.pi)),
                }
            )
    rows = []
    if kinds is not None and REPORT.is_file():
        previous = json.loads(REPORT.read_text())
        rows = [row for row in previous.get("results", []) if row["kind"] not in kinds]
    for path in sorted(CKPT.glob("*.pt")):
        if kinds is not None and not any(path.name.startswith(f"{kind}_") for kind in kinds):
            continue
        model, stats = load_policy(path, device)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.modality_name = payload["modality"]
        successes = []
        for case in cases:
            try:
                flag = rollout(
                    case["task"], case["record"], case["xy"], case["yaw"], model, stats, stokes, mean, std, device
                )
                error = None
            except RuntimeError as exc:
                flag = False
                error = str(exc)
                logger.info("policy stage=eval_error checkpoint=%s folder=%s reason=%s", path.name, case["record"]["view"], exc)
            successes.append(
                {
                    "task": case["task"],
                    "folder": case["record"]["view"],
                    "family": case["record"]["family"],
                    "success": bool(flag),
                    "error": error,
                }
            )
            logger.info(
                "policy stage=eval checkpoint=%s task=%s folder=%s success=%s",
                path.name,
                case["task"],
                case["record"]["view"],
                flag,
            )
        per_task = {}
        for task in TASKS:
            chosen = [item for item in successes if item["task"] == task]
            per_task[task] = float(np.mean([item["success"] for item in chosen]))
        cone = [item for item in successes if item["family"] == "cone"]
        validation = None
        if payload["modality"] == "rgb":
            validation = _validation_success(model, stats, stokes, mean, std, device)
        rows.append(
            {
                "checkpoint": path.name,
                "kind": payload["kind"],
                "modality": payload["modality"],
                "seed": payload["seed"],
                "success_rate": float(np.mean([item["success"] for item in successes])),
                "per_task": per_task,
                "cone_success_rate": float(np.mean([item["success"] for item in cone])) if cone else None,
                "validation_success_rate": validation,
                "episodes": successes,
            }
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        REPORT.write_text(json.dumps({"partial": True, "results": rows}, indent=2) + "\n")
        logger.info("policy stage=eval_partial checkpoints=%d", len(rows))
    paired = []
    for row in rows:
        if row["modality"] == "rgb":
            continue
        baseline = next(
            item for item in rows if item["kind"] == row["kind"] and item["seed"] == row["seed"] and item["modality"] == "rgb"
        )
        paired.append(
            {
                "kind": row["kind"],
                "seed": row["seed"],
                "modality": row["modality"],
                "success_rate": row["success_rate"],
                "rgb_success_rate": baseline["success_rate"],
                "paired_difference": row["success_rate"] - baseline["success_rate"],
                "cone_success_rate": row["cone_success_rate"],
                "cone_rgb_success_rate": baseline["cone_success_rate"],
            }
        )
    report = {
        "definition": (
            "Success rate is the fraction of closed-loop episodes that meet the task predicate. "
            "An episode is one binary trial. The headline is the unweighted mean over slot, tip, and lift, "
            "and over the 10 held-out shapes. The paired difference is that rate minus the RGB policy "
            "with the same architecture and seed. Cone success uses only the cone episodes. "
            "Tip-up tilt decreases only with the upward motion the end effector actually achieves, "
            "and 2 cm of that rise clears the tilt."
        ),
        "control_hz": 5,
        "frames_per_episode": 40,
        "catalog_note": "Eval poses are drawn from a fixed RNG seed, inside ±4 cm of the workspace center.",
        "results": rows,
        "paired": paired,
        "test_shapes": [spec["folder"] for spec in catalog.values() if spec["split"] == "test"],
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("policy stage=eval_done path=%s", REPORT)
    return report
