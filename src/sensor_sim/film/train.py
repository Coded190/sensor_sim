"""Train Model A (RGB) and Model B (RGB + four analyzers) on the CPU.

Both models share this loop. The checkpoint with the best validation mean
IoU is the one scored on the test split. The threshold is the configured
0.5 and is not chosen from the test images.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sensor_sim.film import ROOT
from sensor_sim.film.capture import load_measurement
from sensor_sim.film.images import read_exr
from sensor_sim.film.config import train_config
from sensor_sim.film.dataset import DATASET_ROOT
from sensor_sim.film.metrics import mean_std, scores
from sensor_sim.film.unet import FilmUNet, parameter_count
from sensor_sim.film.lighting import display_srgb
from sensor_sim.render import configure_logging

logger = logging.getLogger("sensor_sim.film")

EVAL_ROOT = ROOT / "outputs" / "film" / "eval"
CHANNEL_NAMES = {
    3: ["R", "G", "B"],
    7: ["R", "G", "B", "I0", "I45", "I90", "I135"],
}


def _load_split(split: str) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    manifest = json.loads((DATASET_ROOT / "manifest.json").read_text())
    chosen = [sample for sample in manifest["samples"] if sample["split"] == split]
    images, labels, metas = [], [], []
    mi.set_variant("scalar_rgb")
    for sample in chosen:
        image, mask = load_measurement(DATASET_ROOT / sample["sample_id"])
        images.append(image.astype(np.float64))
        labels.append(mask.astype(np.float64))
        metas.append(sample)
    if not images:
        raise RuntimeError(f"split {split} is empty")
    return images, labels, metas


def standardization(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and NumPy standard deviation (divide by N) on training pixels."""
    columns = np.concatenate([image.reshape(-1, image.shape[-1]) for image in images], axis=0)
    mean = columns.mean(axis=0)
    scale = columns.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean.astype(np.float64), scale.astype(np.float64)


def positive_weight(labels: list[np.ndarray], floor: float) -> float | None:
    """n_negative / n_positive when the film fraction is below ``floor``."""
    total = sum(int(label.size) for label in labels)
    positive = sum(int(label.sum()) for label in labels)
    fraction = positive / total if total else 0.0
    logger.info("film train n_pixels=%d n_film_pixels=%d fraction=%.6f", total, positive, fraction)
    if fraction == 0.0:
        raise RuntimeError("training masks have no film pixels")
    if fraction < floor:
        weight = (total - positive) / positive
        logger.info("film train pos_weight=%.6f", weight)
        return float(weight)
    logger.info("film train pos_weight=none")
    return None


def _tensors(images, labels, mean, std, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = [image[:, :, :channels] for image in images]
    arrays = [((image - mean[:channels]) / std[:channels]).transpose(2, 0, 1) for image in chosen]
    inputs = torch.from_numpy(np.stack(arrays).astype(np.float32))
    targets = torch.from_numpy(np.stack(labels).astype(np.float32)[:, None, :, :])
    return inputs, targets


def _loss(logits, target, pos_weight, bce_weight, dice_weight, smooth) -> torch.Tensor:
    weight = None if pos_weight is None else torch.tensor([pos_weight], dtype=logits.dtype)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).sum() + smooth
    denominator = probability.sum() + target.sum() + smooth
    dice = 1.0 - numerator / denominator
    return bce_weight * bce + dice_weight * dice


def _predict(model, inputs, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            logits = model(inputs[start : start + batch_size])
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _mean_iou(probabilities, labels, threshold: float) -> float:
    values = [
        scores(labels[index], probabilities[index, 0], threshold)["iou"]
        for index in range(probabilities.shape[0])
    ]
    return float(np.mean(values))


def train_one(name: str, channels: int, train_images, train_labels, val_images, val_labels, mean, std, cfg, out_dir: Path) -> dict:
    """Train one model from seed 0 and return the best-validation test-ready checkpoint path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    model = FilmUNet(channels, base_channels=int(cfg["base_channels"]))
    count = parameter_count(model)
    logger.info("film train model=%s in_channels=%d n_parameters=%d device=cpu", name, channels, count)
    inputs, targets = _tensors(train_images, train_labels, mean, std, channels)
    val_inputs, val_targets = _tensors(val_images, val_labels, mean, std, channels)
    weight = positive_weight(train_labels, float(cfg["pos_fraction_floor"]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        betas=(float(cfg["betas"][0]), float(cfg["betas"][1])),
        weight_decay=float(cfg["weight_decay"]),
    )
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=int(cfg["batch_size"]), shuffle=True)
    best_iou = -1.0
    best_epoch = 0
    threshold = float(cfg["probability_threshold"])
    history = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        started = time.perf_counter()
        losses = []
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = _loss(
                logits,
                batch_targets,
                weight,
                float(cfg["bce_weight"]),
                float(cfg["dice_weight"]),
                float(cfg["dice_smooth"]),
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        probabilities = _predict(model, val_inputs, int(cfg["batch_size"]))
        val_iou = _mean_iou(probabilities, val_labels, threshold)
        elapsed = time.perf_counter() - started
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": mean_loss, "val_iou": val_iou, "elapsed_s": elapsed})
        logger.info(
            "film train model=%s epoch=%d loss=%.6f val_iou=%.6f elapsed_s=%.3f",
            name,
            epoch,
            mean_loss,
            val_iou,
            elapsed,
        )
        state = {
            "model": model.state_dict(),
            "in_channels": channels,
            "base_channels": int(cfg["base_channels"]),
            "epoch": epoch,
            "val_iou": val_iou,
            "n_parameters": count,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "threshold": threshold,
            "pos_weight": weight,
        }
        torch.save(state, out_dir / "last.pt")
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            torch.save(state, out_dir / "best.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    logger.info("film train model=%s status=ok best_epoch=%d best_val_iou=%.6f n_parameters=%d", name, best_epoch, best_iou, count)
    return {"name": name, "best_epoch": best_epoch, "best_val_iou": best_iou, "n_parameters": count, "checkpoint": str(out_dir / "best.pt")}


def _load_model(path: Path) -> tuple[FilmUNet, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = FilmUNet(int(state["in_channels"]), base_channels=int(state["base_channels"]))
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def _rows(model, state, images, labels, metas, channels: int) -> list[dict]:
    mean = np.asarray(state["mean"], dtype=np.float64)
    std = np.asarray(state["std"], dtype=np.float64)
    inputs, _targets = _tensors(images, labels, mean, std, channels)
    probabilities = _predict(model, inputs, batch_size=1)
    rows = []
    for index, meta in enumerate(metas):
        scored = scores(labels[index], probabilities[index, 0], float(state["threshold"]))
        rows.append(
            {
                **{key: meta[key] for key in ("sample_id", "scene_id", "geometry", "coverage", "thickness", "camera_name", "light_name")},
                "iou": scored["iou"],
                "precision": scored["precision"],
                "recall": scored["recall"],
                "probability": probabilities[index, 0],
            }
        )
    return rows


def _breakdown(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {
        name: {metric: mean_std([row[metric] for row in items]) for metric in ("iou", "precision", "recall")}
        for name, items in groups.items()
    }


def _bars(path: Path, summary: dict) -> None:
    metrics = ["iou", "precision", "recall"]
    labels = ["A RGB", "B RGB+analyzers"]
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(metrics))
    width = 0.35
    for offset, (name, label) in enumerate(zip(("rgb", "rgb_polar"), labels)):
        values = [summary[name][metric]["mean"] for metric in metrics]
        ax.bar(positions + (offset - 0.5) * width, values, width=width, label=label)
    ax.set_xticks(positions, ["IoU", "Precision", "Recall"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mean over test images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _qualitative(path: Path, image: np.ndarray, label: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, dolp: np.ndarray, aolp: np.ndarray, threshold: float) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    panels = [
        (axes[0, 0], np.clip(display_srgb(image[:, :, :3]), 0, 1), None, "RGB"),
        (axes[0, 1], image[:, :, 3], "gray", "I0"),
        (axes[0, 2], image[:, :, 4], "gray", "I45"),
        (axes[0, 3], image[:, :, 5], "gray", "I90"),
        (axes[1, 0], image[:, :, 6], "gray", "I135"),
        (axes[1, 1], dolp, "magma", "DoLP"),
        (axes[1, 2], aolp, "twilight", "AoLP"),
        (axes[1, 3], label, "gray", "mask"),
        (axes[2, 0], (pred_a > threshold).astype(float), "gray", "A prediction"),
        (axes[2, 1], (pred_b > threshold).astype(float), "gray", "B prediction"),
        (axes[2, 2], np.abs((pred_a > threshold).astype(float) - label), "magma", "A error"),
        (axes[2, 3], np.abs((pred_b > threshold).astype(float) - label), "magma", "B error"),
    ]
    for axis, panel, cmap, title in panels:
        if cmap is None:
            axis.imshow(panel)
        else:
            axis.imshow(panel, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(cfg: dict) -> dict:
    test_images, test_labels, test_meta = _load_split("test")
    threshold = float(cfg["probability_threshold"])
    summary = {}
    stored = {}
    for name, channels in (("rgb", 3), ("rgb_polar", 7)):
        model, state = _load_model(ROOT / "outputs" / "film" / "train" / name / "best.pt")
        rows = _rows(model, state, test_images, test_labels, test_meta, channels)
        stored[name] = rows
        summary[name] = {
            "iou": mean_std([row["iou"] for row in rows]),
            "precision": mean_std([row["precision"] for row in rows]),
            "recall": mean_std([row["recall"] for row in rows]),
            "n_parameters": state["n_parameters"],
            "best_epoch": state["epoch"],
            "by_geometry": _breakdown(rows, "geometry"),
            "by_coverage": _breakdown(rows, "coverage"),
            "by_thickness": _breakdown(rows, "thickness"),
            "by_camera": _breakdown(rows, "camera_name"),
            "by_light": _breakdown(rows, "light_name"),
            "images": [{key: row[key] for key in row if key != "probability"} for row in rows],
        }
        logger.info(
            "film eval model=%s iou_mean=%.6f iou_std=%.6f precision_mean=%.6f recall_mean=%.6f n=%d",
            name,
            summary[name]["iou"]["mean"],
            summary[name]["iou"]["std"],
            summary[name]["precision"]["mean"],
            summary[name]["recall"]["mean"],
            summary[name]["iou"]["n"],
        )
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    _bars(EVAL_ROOT / "comparison.png", summary)
    by_id = {row["sample_id"]: row for row in stored["rgb"]}
    polar_by_id = {row["sample_id"]: row for row in stored["rgb_polar"]}
    for meta, image, label in zip(test_meta, test_images, test_labels, strict=True):
        if meta["geometry"] != "sphere" and meta["scene_id"] != "ellipsoid_cov45":
            continue
        if meta["scene_id"] not in ("sphere_cov15", "ellipsoid_cov45"):
            continue
        sample_dir = DATASET_ROOT / meta["sample_id"] / "polarization"
        dolp = read_exr(sample_dir / "dolp.exr")
        aolp = read_exr(sample_dir / "aolp.exr")
        dolp_image = next(iter(dolp.values()))
        aolp_image = next(iter(aolp.values()))
        _qualitative(
            EVAL_ROOT / f"{meta['sample_id']}_{meta['scene_id']}.png",
            image,
            label,
            by_id[meta["sample_id"]]["probability"],
            polar_by_id[meta["sample_id"]]["probability"],
            dolp_image[:, :, 0] if dolp_image.ndim == 3 else dolp_image,
            aolp_image[:, :, 0] if aolp_image.ndim == 3 else aolp_image,
            threshold,
        )
    (EVAL_ROOT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def train_models() -> dict:
    configure_logging()
    if torch.cuda.is_available():
        raise RuntimeError("CUDA is visible; this training run is CPU-only")
    cfg = train_config()
    train_images, train_labels, _train_meta = _load_split("train")
    val_images, val_labels, _val_meta = _load_split("validation")
    mean, std = standardization(train_images)
    stats_dir = ROOT / "outputs" / "film" / "train"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "standardize.json").write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist(), "channels": CHANNEL_NAMES[7]}, indent=2) + "\n"
    )
    results = {}
    for name, channels in (("rgb", 3), ("rgb_polar", 7)):
        results[name] = train_one(
            name,
            channels,
            train_images,
            train_labels,
            val_images,
            val_labels,
            mean,
            std,
            cfg,
            stats_dir / name,
        )
    summary = evaluate(cfg)
    return {"train": results, "test": summary}


def main() -> None:
    train_models()


if __name__ == "__main__":
    main()
