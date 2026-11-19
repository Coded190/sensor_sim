"""Score Stokes, analyzer, and RGB normals on the policy evaluation poses.

The room, the arm cubes, the camera rule, and the object meshes are the ones
``evaluate`` renders. The grasped mesh is already ``film.obj`` with a smooth
dielectric shell. This module does not build a second room or a second film.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from sensor_sim.film.heldout import _fit_model
from sensor_sim.film.sfp import OUTPUT, SEED, _channels
from sensor_sim.film.sfp_math import angular_error_summary
from sensor_sim.film.shape_mesh import shape_catalog
from sensor_sim.policy.evaluate import TASKS
from sensor_sim.policy.stokes_checkpoint import load_records

logger = logging.getLogger("sensor_sim.policy")

REPORT_DIR = OUTPUT / "policy" / "scene_normals"
MODELS = ("stokes", "analyzer", "rgb")
DEFINITION = (
    "Mean angular error is the mean, over film pixels with S0 > 1e-4 and a unit "
    "shading normal, of arccos(clip(dot(prediction, truth), -1, 1)) in degrees. "
    "The truth is the Mitsuba shading normal from the policy scene's 1-sample AOV, "
    "rotated into the camera. A shape score is the unweighted mean of its slot, tip, "
    "and lift frames. The headline is the unweighted mean of the ten test shapes."
)


def evaluation_poses() -> list[dict]:
    """The 30 poses ``evaluate`` draws before the first action. RNG seed is 1."""
    specs = [spec for spec in shape_catalog(SEED) if spec["split"] == "test"]
    rng = np.random.default_rng(1)
    poses = []
    for spec in specs:
        for task in TASKS:
            poses.append(
                {
                    "folder": spec["folder"],
                    "family": spec["family"],
                    "task": task,
                    "xy": (float(rng.uniform(-0.04, 0.04)), float(0.02 + rng.uniform(-0.04, 0.04))),
                    "yaw": float(rng.uniform(0.0, 2.0 * np.pi)),
                }
            )
    if len(poses) != 30:
        raise RuntimeError(f"expected 30 evaluation poses, got {len(poses)}")
    return poses


def require_filmed_policy_scene(scene: dict) -> None:
    """The grasped object stays the dielectric shell, and the arm is seven diffuse cubes."""
    shell = scene.get("film-shell")
    solid = scene.get("solid")
    if not isinstance(shell, dict) or not isinstance(solid, dict):
        raise RuntimeError("policy scene is missing film-shell or solid")
    if shell["bsdf"]["type"] != "dielectric" or float(shell["bsdf"]["int_ior"]) != 1.5:
        raise RuntimeError(f"film-shell BSDF is not the IOR-1.5 dielectric: {shell['bsdf']}")
    if not str(shell["filename"]).endswith("film.obj"):
        raise RuntimeError(f"film-shell is not film.obj: {shell['filename']}")
    if solid["bsdf"]["type"] != "diffuse":
        raise RuntimeError(f"inner solid BSDF is not diffuse: {solid['bsdf']}")
    arms = [key for key in scene if str(key).startswith("arm-")]
    if len(arms) != 7:
        raise RuntimeError(f"expected 7 arm cubes, got {arms}")
    for name in arms:
        if scene[name]["bsdf"]["type"] != "diffuse":
            raise RuntimeError(f"{name} BSDF is not diffuse")
    wrapped = scene["integrator"]
    if wrapped["type"] != "stokes" or wrapped["integrator"]["type"] != "path":
        raise RuntimeError(f"policy render integrator is not stokes around path: {wrapped}")


def score_mask(measurement: dict) -> np.ndarray:
    """Film pixels with S0 above 1e-4 and a unit shading normal."""
    film = np.rint(measurement["shape_index"]).astype(np.int64) == int(measurement["film_index"])
    unit = np.linalg.norm(measurement["normal"], axis=-1) > 0.5
    return film & (np.asarray(measurement["s0"], dtype=np.float64) > 1e-4) & unit


def predict_normal(model, measurement: dict, mean: np.ndarray, std: np.ndarray, model_name: str, device: str) -> np.ndarray:
    """Unit camera-frame normal from one trained U-Net."""
    import torch

    image = _channels(measurement, model_name)
    standardized = (image - mean) / std
    tensor = torch.tensor(np.transpose(standardized, (2, 0, 1))[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = model(tensor)[0]
        prediction = prediction / prediction.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return np.transpose(prediction.detach().cpu().numpy(), (1, 2, 0))


def _load_model(path: Path, device: str):
    import torch

    from sensor_sim.film.unet import FilmUNet

    payload = torch.load(path, map_location=device, weights_only=False)
    model = FilmUNet(int(payload["in_channels"]), out_channels=3).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, np.asarray(payload["mean"], dtype=np.float64), np.asarray(payload["std"], dtype=np.float64), payload


def fit_models(device: str = "cuda") -> dict[str, Path]:
    """Train Stokes, analyzer, and RGB on the 50 static training shapes and save the weights."""
    train = load_records("train")
    test = load_records("test")
    if len(train) != 50 or len(test) != 10:
        raise RuntimeError(f"expected 50 train and 10 test measurements, got {len(train)} and {len(test)}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    static_rows = []
    for model_name in MODELS:
        path = REPORT_DIR / f"{model_name}.pt"
        logger.info("policy stage=scene_normals_train model=%s", model_name)
        predictions, info = _fit_model(train, test, model_name, device, crops_per_image=1, checkpoint_path=path)
        per_shape = []
        for record, predicted in zip(test, predictions, strict=True):
            from sensor_sim.film.sfp import _loss_mask

            mask = _loss_mask(record["measurement"], set(record["object_indices"]))
            summary = angular_error_summary(predicted, record["measurement"]["normal"], mask)
            per_shape.append({"view": record["view"], "mae_deg": summary["mae_deg"], "pixels": summary["pixels"]})
        static_rows.append(
            {
                "model": model_name,
                "unweighted_mean_mae_deg": float(np.mean([row["mae_deg"] for row in per_shape])),
                "shapes": per_shape,
                "final_epoch_loss": info["final_epoch_loss"],
            }
        )
        paths[model_name] = path
        logger.info(
            "policy stage=scene_normals_static model=%s mae_deg=%.4f",
            model_name,
            static_rows[-1]["unweighted_mean_mae_deg"],
        )
        del predictions
    (REPORT_DIR / "static_test.json").write_text(
        json.dumps(
            {
                "definition": (
                    "Same angular error as experiment 02, on the static held-out renders with no arm. "
                    "This checks the retrained weights against summary.json before the policy-scene score."
                ),
                "models": static_rows,
            },
            indent=2,
        )
        + "\n"
    )
    return paths


def _first_frame(task: str, record: dict, xy: tuple[float, float], yaw: float) -> dict:
    """The Mitsuba measurement evaluate renders before the first action."""
    from sensor_sim.policy.arm import ArmEnv
    from sensor_sim.policy.collect import choose_distance
    from sensor_sim.policy.observe import _apply, pose_matrix, render_measurement, scene_builder
    from sensor_sim.policy.tasks import scripted_episode

    radius = float(record["radius_m"])
    tilt = 1.2 if task == "tip" else 0.0
    env = ArmEnv()
    env.reset(xy, yaw, tilt, radius)
    preview = scripted_episode(task, record["family"], xy, yaw, radius)
    camera, fraction = choose_distance(record, preview)
    env.reset(xy, yaw, tilt, radius)
    pose = env.object_pose()
    matrix = pose_matrix(
        record["floor_xy"],
        record["yaw_rad"],
        pose["position"][:2],
        pose["yaw"],
        pose["tilt"],
        radius,
    )
    segments = env.link_segments()
    build, _probe = scene_builder(record["mesh_dir"], matrix, segments, camera, record["probe"])
    require_filmed_policy_scene(build(polarized=True, spp=64, rfilter="gaussian"))
    measurement = render_measurement(
        record["mesh_dir"],
        matrix,
        segments,
        camera,
        _apply(matrix, record["probe"]),
        fallback=np.asarray(pose["position"], dtype=np.float64),
    )
    measurement["camera_film_fraction"] = float(fraction)
    return measurement


def evaluate_policy_scene(device: str = "cuda") -> dict:
    """Fit the three U-Nets, then score the 30 evaluation frames."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from sensor_sim.policy.collect import _record

    paths = fit_models(device)
    loaded = {name: _load_model(paths[name], device) for name in MODELS}
    rows = []
    examples = []
    for pose in evaluation_poses():
        record = _record("test", pose["folder"])
        measurement = _first_frame(pose["task"], record, pose["xy"], pose["yaw"])
        mask = score_mask(measurement)
        scores = {}
        predictions = {}
        for model_name in MODELS:
            model, mean, std, _payload = loaded[model_name]
            predicted = predict_normal(model, measurement, mean, std, model_name, device)
            predictions[model_name] = predicted
            summary = angular_error_summary(predicted, measurement["normal"], mask)
            scores[model_name] = summary
        row = {
            "folder": pose["folder"],
            "family": pose["family"],
            "task": pose["task"],
            "xy": list(pose["xy"]),
            "yaw": pose["yaw"],
            "pixels": int(mask.sum()),
            "film_fraction": float(measurement["film_fraction"]),
            "mae_deg": {name: scores[name]["mae_deg"] for name in MODELS},
        }
        rows.append(row)
        if pose["task"] == "slot" and pose["folder"] in {"001_ellipsoid", "003_rounded_box", "010_cone"}:
            examples.append((pose, measurement, mask, predictions, scores))
        logger.info(
            "policy stage=scene_normals_frame folder=%s task=%s pixels=%d stokes=%.3f analyzer=%.3f rgb=%.3f",
            pose["folder"],
            pose["task"],
            int(mask.sum()),
            scores["stokes"]["mae_deg"],
            scores["analyzer"]["mae_deg"],
            scores["rgb"]["mae_deg"],
        )
        del measurement
    by_shape = []
    for folder in sorted({row["folder"] for row in rows}):
        chosen = [row for row in rows if row["folder"] == folder]
        by_shape.append(
            {
                "folder": folder,
                "family": chosen[0]["family"],
                "mae_deg": {
                    name: float(np.mean([row["mae_deg"][name] for row in chosen]))
                    for name in MODELS
                },
            }
        )
    headline = {name: float(np.mean([row["mae_deg"][name] for row in by_shape])) for name in MODELS}
    report = {
        "definition": DEFINITION,
        "scene": (
            "Same gray room, dielectric film.obj, diffuse solid.obj, and seven diffuse arm cubes "
            "as policy evaluation. Frame 0 of each of the 30 scored episodes, before any policy action. "
            "The film was already on the grasped object."
        ),
        "headline_unweighted_mean_mae_deg": headline,
        "shapes": by_shape,
        "frames": rows,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_figure(examples, by_shape, headline)
    logger.info(
        "policy stage=scene_normals_done stokes=%.4f analyzer=%.4f rgb=%.4f",
        headline["stokes"],
        headline["analyzer"],
        headline["rgb"],
    )
    return report


def _write_bars(by_shape: list[dict], headline: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.2, 6.4))
    labels = [row["folder"] for row in by_shape]
    positions = np.arange(len(by_shape))
    height = 0.24
    for offset, name in enumerate(MODELS):
        values = [row["mae_deg"][name] for row in by_shape]
        axis.barh(positions + (offset - 1) * height, values, height=height, label=name)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("mean angular error over slot, tip, and lift (deg)")
    axis.set_title(
        "Policy evaluation poses.\n"
        f"Unweighted mean of the ten shapes: Stokes {headline['stokes']:.2f}°, "
        f"analyzer {headline['analyzer']:.2f}°, RGB {headline['rgb']:.2f}°."
    )
    axis.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    axis.invert_yaxis()
    figure.tight_layout()
    path = REPORT_DIR / "means.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    logger.info("policy stage=scene_normals_bars path=%s", path)


def _write_frames(examples: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sensor_sim.film.lighting import display_srgb
    from sensor_sim.film.sfp_math import angular_error_deg

    figure, axes = plt.subplots(len(examples), 5, figsize=(14.2, 8.6), squeeze=False)
    figure.subplots_adjust(left=0.02, right=0.90, top=0.90, bottom=0.08, wspace=0.12, hspace=0.32)
    error_cmap = plt.cm.magma.copy()
    error_cmap.set_bad("white")
    error_artist = None
    for row_id, (pose, measurement, mask, predictions, scores) in enumerate(examples):
        rgb = np.clip(display_srgb(measurement["rgb"]), 0.0, 1.0)
        film = np.rint(measurement["shape_index"]).astype(np.int64) == int(measurement["film_index"])
        unit = np.linalg.norm(measurement["normal"], axis=-1) > 0.5
        normal = np.clip(measurement["normal"] * 0.5 + 0.5, 0.0, 1.0)
        normal[~(film & unit)] = 0.0
        panels = [rgb, normal]
        titles = [f"RGB\n{pose['folder']} slot", "Ground-truth normal"]
        for model_name in MODELS:
            panels.append(np.where(mask, angular_error_deg(predictions[model_name], measurement["normal"]), np.nan))
            titles.append(f"{model_name}\n{scores[model_name]['mae_deg']:.2f}°")
        rows, cols = np.nonzero(film)
        top, bottom = max(0, int(rows.min()) - 12), min(film.shape[0], int(rows.max()) + 13)
        left, right = max(0, int(cols.min()) - 12), min(film.shape[1], int(cols.max()) + 13)
        window = (slice(top, bottom), slice(left, right))
        for column, (image, title) in enumerate(zip(panels, titles, strict=True)):
            shown = image[window]
            if column >= 2:
                error_artist = axes[row_id, column].imshow(shown, vmin=0.0, vmax=90.0, cmap=error_cmap)
            else:
                axes[row_id, column].imshow(shown)
            axes[row_id, column].set_title(title, fontsize=9)
            axes[row_id, column].axis("off")
    if error_artist is not None:
        figure.colorbar(error_artist, ax=axes[:, 2:], fraction=0.03, pad=0.02).set_label("angular error (deg)")
    figure.suptitle("Frame 0 of the slot episode in the policy evaluation scene.", fontsize=12)
    figure.text(
        0.01,
        0.015,
        "The object is the dielectric film.obj already used at evaluation, plus the seven diffuse arm cubes. "
        "White pixels have S0 <= 1e-4 or are not the film. RGB is the U-Net that does not read polarization.",
        fontsize=8,
    )
    path = REPORT_DIR / "frames.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    logger.info("policy stage=scene_normals_frames path=%s", path)


def _write_figure(examples: list, by_shape: list[dict], headline: dict) -> None:
    _write_bars(by_shape, headline)
    _write_frames(examples)
