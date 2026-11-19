"""Script demonstrations and cache Mitsuba observations.

The viewing direction is fixed, off the arm's reach. Distance is shortened
until the film covers at least 0.15 of the frame, because a 3 cm object does
not fill that fraction from the distance that frames an 8 cm object.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from sensor_sim.film.lighting import display_srgb
from sensor_sim.film.sfp import OUTPUT, SEED
from sensor_sim.film.shape_mesh import MIN_FILM_FRACTION, shape_catalog
from sensor_sim.policy.observe import (
    WORKSPACE_CENTER,
    _apply,
    _camera,
    film_fraction_at,
    modality_images,
    pose_matrix,
    render_measurement,
)
from sensor_sim.policy.stokes_checkpoint import load_stokes
from sensor_sim.policy.tasks import scripted_episode

logger = logging.getLogger("sensor_sim.policy")

DATA = OUTPUT / "policy" / "demos"
VIEW_AZIMUTH = 1.8
VIEW_ELEVATION = 0.7
DISTANCES = (0.22, 0.18, 0.15, 0.12, 0.10, 0.08)
TASKS = ("slot", "tip", "lift")


def viewing_camera(distance: float, elevation: float = VIEW_ELEVATION) -> dict:
    origin = WORKSPACE_CENTER + float(distance) * np.array(
        [
            np.cos(VIEW_AZIMUTH) * np.cos(elevation),
            np.sin(VIEW_AZIMUTH) * np.cos(elevation),
            np.sin(elevation),
        ]
    )
    return _camera(origin, WORKSPACE_CENTER)


def _record(split: str, folder: str) -> dict:
    path = OUTPUT / "02_heldout_shapes" / split / folder
    record = json.loads((path / "metadata.json").read_text())
    record["mesh_dir"] = path / "meshes"
    record["probe"] = np.asarray(record["shapes"]["film-shell"]["point"], dtype=np.float64)
    return record


def choose_distance(record: dict, episode: dict) -> tuple[dict, float]:
    """Shorten the fixed view until the start, middle, and last frames clear 0.15."""
    frames = [episode["frames"][index] for index in (0, 20, 39)]
    best_camera = None
    best_fraction = -1.0
    for elevation in (VIEW_ELEVATION, 0.4):
        for distance in DISTANCES:
            camera = viewing_camera(distance, elevation)
            fractions = []
            for frame in frames:
                pose = frame["object"]
                matrix = pose_matrix(
                    record["floor_xy"],
                    record["yaw_rad"],
                    pose["position"][:2],
                    pose["yaw"],
                    pose["tilt"],
                    record["radius_m"],
                )
                fractions.append(
                    film_fraction_at(
                        record["mesh_dir"],
                        matrix,
                        frame["segments"],
                        camera,
                        _apply(matrix, record["probe"]),
                        fallback=np.asarray(pose["position"], dtype=np.float64),
                    )
                )
            worst = float(min(fractions))
            logger.info(
                "policy stage=dolly folder=%s elevation=%.2f distance=%.2f film_fraction=%.3f,%.3f,%.3f",
                record["view"],
                elevation,
                distance,
                fractions[0],
                fractions[1],
                fractions[2],
            )
            if worst > best_fraction:
                best_camera = camera
                best_fraction = worst
            if worst >= MIN_FILM_FRACTION:
                return camera, worst
    if best_camera is None:
        raise RuntimeError(f"{record['view']} produced no camera measurement")
    logger.info(
        "policy stage=dolly_best folder=%s film_fraction=%.3f below_gate=%.2f",
        record["view"],
        best_fraction,
        MIN_FILM_FRACTION,
    )
    return best_camera, best_fraction


def render_episode(record: dict, episode: dict, camera: dict, stokes, mean, std, device: str) -> dict:
    frames = {key: [] for key in ("rgb", "normals", "gt_normals", "dolp", "state", "action", "film_fraction")}
    for frame in episode["frames"]:
        pose = frame["object"]
        matrix = pose_matrix(
            record["floor_xy"],
            record["yaw_rad"],
            pose["position"][:2],
            pose["yaw"],
            pose["tilt"],
            record["radius_m"],
        )
        measurement = render_measurement(
            record["mesh_dir"],
            matrix,
            frame["segments"],
            camera,
            _apply(matrix, record["probe"]),
            fallback=np.asarray(pose["position"], dtype=np.float64),
        )
        images = modality_images(measurement, stokes, mean, std, device)
        for key in ("rgb", "normals", "gt_normals", "dolp", "film_fraction"):
            frames[key].append(images[key])
        frames["state"].append(frame["state"])
        frames["action"].append(frame["action"])
    return {
        "rgb": np.stack(frames["rgb"]),
        "normals": np.stack(frames["normals"]),
        "gt_normals": np.stack(frames["gt_normals"]),
        "dolp": np.stack(frames["dolp"]),
        "state": np.stack(frames["state"]).astype(np.float32),
        "action": np.stack(frames["action"]).astype(np.float32),
        "film_fraction": np.asarray(frames["film_fraction"], dtype=np.float32),
    }


def collect(n_per_task: int = 30, device: str = "cuda") -> None:
    """Write ``n_per_task`` expert episodes for slot, tip, and lift."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    catalog = [spec for spec in shape_catalog(SEED) if spec["split"] == "train"]
    stokes, mean, std = load_stokes(device)
    rng = np.random.default_rng(0)
    DATA.mkdir(parents=True, exist_ok=True)
    index = []
    for task in TASKS:
        for episode_id in range(n_per_task):
            saved = None
            for attempt in range(8):
                spec = catalog[(episode_id * 3 + TASKS.index(task) + attempt) % len(catalog)]
                record = _record("train", spec["folder"])
                xy = (float(rng.uniform(-0.04, 0.04)), float(0.02 + rng.uniform(-0.04, 0.04)))
                yaw = float(rng.uniform(0.0, 2.0 * np.pi))
                episode = scripted_episode(task, spec["family"], xy, yaw, float(record["radius_m"]))
                if not episode["success"]:
                    logger.info("policy stage=expert_retry task=%s folder=%s", task, spec["folder"])
                    continue
                try:
                    camera, fraction = choose_distance(record, episode)
                except RuntimeError:
                    logger.info("policy stage=dolly_give_up task=%s folder=%s", task, spec["folder"])
                    continue
                saved = render_episode(record, episode, camera, stokes, mean, std, device)
                saved_path = DATA / task / f"{episode_id:03d}.npz"
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    saved_path,
                    rgb=saved["rgb"],
                    normals=saved["normals"],
                    gt_normals=saved["gt_normals"],
                    dolp=saved["dolp"],
                    state=saved["state"],
                    action=saved["action"],
                )
                split = "val" if episode_id >= n_per_task - 5 else "train"
                index.append(
                    {
                        "task": task,
                        "episode": episode_id,
                        "split": split,
                        "folder": spec["folder"],
                        "family": spec["family"],
                        "path": str(saved_path),
                        "xy": xy,
                        "yaw": yaw,
                        "radius_m": float(record["radius_m"]),
                        "expert_success": True,
                        "first_frame_film_fraction": fraction,
                        "min_film_fraction": float(saved["film_fraction"].min()),
                    }
                )
                logger.info(
                    "policy stage=demo task=%s episode=%d folder=%s split=%s min_film_fraction=%.3f",
                    task,
                    episode_id,
                    spec["folder"],
                    split,
                    float(saved["film_fraction"].min()),
                )
                break
            if saved is None:
                raise RuntimeError(f"expert failed six times for {task} {spec['folder']}")
    (DATA / "index.json").write_text(json.dumps({"episodes": index, "pose_box_m": 0.04, "pose_box_reason": "The catalog box of ±0.15 m put small shapes under the 0.15 film-fraction gate for a camera that also sees the arm. Episodes stay inside ±0.04 m of (0, 0.02). The view direction is fixed at azimuth 1.8 rad and elevation 0.7 rad. Distance is shortened per episode until the film fraction clears 0.15.", "keys": [
        "observation.images.rgb",
        "observation.images.normals",
        "observation.images.gt_normals",
        "observation.images.dolp",
        "observation.state",
        "action",
    ]}, indent=2) + "\n")
    _contact_sheet(DATA / "slot" / "000.npz", OUTPUT / "policy" / "demo_contact.png")
    logger.info("policy stage=demos_done episodes=%d", len(index))


def _contact_sheet(episode_path: Path, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episode = np.load(episode_path)
    figure, axes = plt.subplots(2, 4, figsize=(10, 5))
    for column, frame in enumerate((0, 12, 24, 39)):
        rgb = np.clip(display_srgb(episode["rgb"][frame].astype(np.float64)), 0.0, 1.0)
        normals = np.clip(episode["normals"][frame].astype(np.float64) * 0.5 + 0.5, 0.0, 1.0)
        axes[0, column].imshow(rgb)
        axes[0, column].set_title(f"rgb t={frame}")
        axes[0, column].axis("off")
        axes[1, column].imshow(normals)
        axes[1, column].set_title("stokes normals")
        axes[1, column].axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(out_path, dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    collect()
