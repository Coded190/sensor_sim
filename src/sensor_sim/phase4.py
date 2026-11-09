"""Film-segmentation probe on the Phase 3 frames.

``uv run phase4`` reads ``outputs/phase3`` and writes ``outputs/phase4``.
It does not render. Four methods are scored by mean frame IoU:

- ``color_threshold``: ``R / (R+G+B)`` above a threshold chosen on the
  training frames.
- ``rgb``, ``rgb_dolp``, ``rgb_dolp_cube``: balanced logistic regression
  on the linear measurements. A pixel is film when the probability is
  above 0.5.

IoU is intersection / union per frame. Two empty masks score 1. The
headline number is the mean over the eight test frames.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from sklearn.linear_model import LogisticRegression

from sensor_sim.render import configure_logging, tonemap_srgb
from sensor_sim.spectral import spectral_bands
from sensor_sim.stokes import S0_MIN, polarization_maps

logger = logging.getLogger("phase4")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE3_ROOT = ROOT / "outputs" / "phase3"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "phase4"

TRAIN_FILM = list(range(0, 20))
TRAIN_PPLASTIC = list(range(24, 28))
TEST_FILM = list(range(20, 24))
TEST_PPLASTIC = list(range(28, 32))
TRAIN_IDS = TRAIN_FILM + TRAIN_PPLASTIC
TEST_IDS = TEST_FILM + TEST_PPLASTIC
FILM_IDS = set(range(0, 24))
OVERLAY_IDS = (20, 28)

PROBABILITY_THRESHOLD = 0.5
COLOR_SUM_MIN = 1e-4
COLOR_THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)
METHOD_ORDER = ("color_threshold", "rgb", "rgb_dolp", "rgb_dolp_cube")

IOU_DEFINITION = (
    "Per frame, intersection / union of the predicted film pixels and mask.png. "
    "If both are empty, IoU is 1. Pixel accuracy is not used."
)
COLOR_DEFINITION = (
    "R / (R+G+B) where R+G+B > 1e-4, else 0. "
    "Prediction is feature >= threshold. "
    "The threshold is the smallest value in {0, 0.01, ..., 1} that maximizes "
    "mean training-frame IoU. Test frames are not used."
)
DOLP_DEFINITION = "sqrt(S1^2 + S2^2) / S0 where S0 > 1e-4, else 0. Spectral copies are averaged."


@dataclass
class Frame:
    index: int
    rgb: np.ndarray
    dolp: np.ndarray
    cube: np.ndarray
    mask: np.ndarray


def feature_columns() -> dict[str, tuple[str, ...]]:
    bands = tuple(band.name for band in spectral_bands())
    rgb = ("R", "G", "B")
    return {
        "rgb": rgb,
        "rgb_dolp": rgb + ("DoLP",),
        "rgb_dolp_cube": rgb + ("DoLP",) + bands,
    }


def frame_iou(prediction: np.ndarray, mask: np.ndarray) -> float:
    """Intersection over union. Both empty scores 1."""
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(mask, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction shape {pred.shape} != mask shape {truth.shape}")
    union = int(np.logical_or(pred, truth).sum())
    if union == 0:
        return 1.0
    intersection = int(np.logical_and(pred, truth).sum())
    return float(intersection / union)


def redness(rgb: np.ndarray) -> np.ndarray:
    """R / (R+G+B), or 0 where the sum is at most ``COLOR_SUM_MIN``."""
    total = np.asarray(rgb, dtype=np.float64).sum(axis=2)
    red = np.zeros(total.shape, dtype=np.float64)
    np.divide(rgb[:, :, 0], total, out=red, where=total > COLOR_SUM_MIN)
    return red


def stack_features(frame: Frame) -> dict[str, np.ndarray]:
    """HxWxC stacks. Column order is ``feature_columns``."""
    rgb = np.asarray(frame.rgb, dtype=np.float64)
    dolp = np.asarray(frame.dolp, dtype=np.float64)[:, :, None]
    cube = np.asarray(frame.cube, dtype=np.float64)
    columns = feature_columns()
    stacked = {
        "rgb": rgb,
        "rgb_dolp": np.concatenate([rgb, dolp], axis=2),
        "rgb_dolp_cube": np.concatenate([rgb, dolp, cube], axis=2),
    }
    for name, image in stacked.items():
        expected = len(columns[name])
        if image.shape[2] != expected:
            raise ValueError(f"{name} has {image.shape[2]} columns, expected {expected}")
    return stacked


def _bitmap_channels(path: Path) -> dict[str, np.ndarray]:
    bitmap = mi.Bitmap(str(path))
    return {name: np.array(image, dtype=np.float64) for name, image in bitmap.split()}


def load_frame(phase3_root: Path, index: int) -> Frame:
    """Linear RGB, DoLP, cube, and film mask for one Phase 3 frame."""
    mi.set_variant("scalar_rgb")
    frame_dir = Path(phase3_root) / "frames" / f"{index:03d}"
    stokes = _bitmap_channels(frame_dir / "stokes" / "render.exr")
    if "<root>" not in stokes or "S0" not in stokes:
        raise RuntimeError(f"frame {index:03d} Stokes EXR is missing RGB or S0")
    maps = polarization_maps(stokes["S0"], stokes["S1"], stokes["S2"], stokes["S3"])
    cube_bitmap = mi.Bitmap(str(frame_dir / "cube" / "render.exr"))
    cube = np.array(cube_bitmap, dtype=np.float64)
    names = [field.name for field in cube_bitmap.struct_()]
    expected = [band.name for band in spectral_bands()]
    if names != expected:
        raise RuntimeError(f"frame {index:03d} cube channels {names} != {expected}")
    mask_image = plt.imread(frame_dir / "mask.png")
    if mask_image.ndim == 3:
        mask_image = mask_image[:, :, 0]
    mask = np.asarray(mask_image) > 0.5
    return Frame(
        index=index,
        rgb=np.array(stokes["<root>"], dtype=np.float64),
        dolp=maps["dolp"],
        cube=cube,
        mask=mask,
    )


def load_phase3(phase3_root: Path) -> list[Frame]:
    mi.set_variant("scalar_rgb")
    return [load_frame(phase3_root, index) for index in range(32)]


def choose_color_threshold(
    frames: list[Frame],
    train_ids: list[int],
) -> tuple[float, float]:
    """Smallest grid threshold that maximizes mean training-frame IoU.

    ``frames`` may contain test frames. Only ``train_ids`` are scored.
    """
    by_index = {frame.index: frame for frame in frames}
    train = [by_index[index] for index in train_ids]
    features = [redness(frame.rgb) for frame in train]
    best_threshold = float(COLOR_THRESHOLDS[0])
    best_iou = -1.0
    for threshold in COLOR_THRESHOLDS:
        values = [frame_iou(feature >= threshold, frame.mask) for feature, frame in zip(features, train)]
        score = float(np.mean(values))
        if score > best_iou:
            best_iou = score
            best_threshold = float(threshold)
    return best_threshold, best_iou


def _group_means(frame_scores: dict[int, float], ids: list[int], film_ids: set[int]) -> dict:
    film = [index for index in ids if index in film_ids]
    opaque = [index for index in ids if index not in film_ids]

    def mean(group: list[int]) -> float | None:
        if not group:
            return None
        return float(np.mean([frame_scores[index] for index in group]))

    return {
        "iou": mean(ids),
        "film_iou": mean(film),
        "pplastic_iou": mean(opaque),
        "frame_iou": {str(index): frame_scores[index] for index in ids},
    }


def _fit_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Training-pixel mean and standard deviation. Zero-variance columns use scale 1."""
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean, scale


def _predict_logistic(frame: Frame, key: str, mean: np.ndarray, scale: np.ndarray, model) -> np.ndarray:
    features = stack_features(frame)[key].reshape(-1, mean.shape[0])
    standardized = (features - mean) / scale
    classes = list(model.classes_)
    if 1 not in classes:
        probability = np.zeros(features.shape[0], dtype=np.float64)
    else:
        probability = model.predict_proba(standardized)[:, classes.index(1)]
    height, width = frame.mask.shape
    return probability.reshape(height, width) > PROBABILITY_THRESHOLD


def _train_matrix(frames: list[Frame], train_ids: list[int], key: str) -> tuple[np.ndarray, np.ndarray]:
    by_index = {frame.index: frame for frame in frames}
    features = []
    labels = []
    for index in train_ids:
        frame = by_index[index]
        image = stack_features(frame)[key]
        features.append(image.reshape(-1, image.shape[2]))
        labels.append(frame.mask.reshape(-1))
    return np.concatenate(features), np.concatenate(labels)


def run_probe(
    frames: list[Frame],
    output_dir: Path,
    train_ids: list[int] | None = None,
    test_ids: list[int] | None = None,
    film_ids: set[int] | None = None,
    overlay_ids: tuple[int, ...] = OVERLAY_IDS,
) -> dict:
    """Fit the four methods and write metrics, the bar chart, and overlays."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if train_ids is None:
        train_ids = TRAIN_IDS
    if test_ids is None:
        test_ids = TEST_IDS
    if film_ids is None:
        film_ids = FILM_IDS
    if set(train_ids) & set(test_ids):
        raise ValueError("train and test frames overlap")

    by_index = {frame.index: frame for frame in frames}
    threshold, train_color_iou = choose_color_threshold(frames, train_ids)
    logger.info(
        "phase4 method=color_threshold threshold=%.2f train_iou_at_selection=%.6f",
        threshold,
        train_color_iou,
    )

    predictions: dict[str, dict[int, np.ndarray]] = {}
    scales: dict[str, dict[str, list[float]]] = {}
    color_pred = {}
    for frame in frames:
        color_pred[frame.index] = redness(frame.rgb) >= threshold
    predictions["color_threshold"] = color_pred

    for key in ("rgb", "rgb_dolp", "rgb_dolp_cube"):
        features, labels = _train_matrix(frames, train_ids, key)
        mean, scale = _fit_scale(features)
        model = LogisticRegression(
            class_weight="balanced",
            random_state=0,
            solver="lbfgs",
            max_iter=500,
        )
        model.fit((features - mean) / scale, labels.astype(np.int8))
        iterations = int(np.max(model.n_iter_))
        logger.info(
            "phase4 method=%s status=fit n_iter=%d n_train_pixels=%d n_train_film_pixels=%d",
            key,
            iterations,
            int(labels.size),
            int(labels.sum()),
        )
        if iterations >= 500:
            logger.warning("phase4 method=%s status=not_converged n_iter=%d", key, iterations)
        predictions[key] = {
            frame.index: _predict_logistic(frame, key, mean, scale, model) for frame in frames
        }
        scales[key] = {"mean": mean.tolist(), "std": scale.tolist()}

    methods = {}
    for name in METHOD_ORDER:
        frame_scores = {
            frame.index: frame_iou(predictions[name][frame.index], frame.mask) for frame in frames
        }
        train_summary = _group_means(frame_scores, train_ids, film_ids)
        test_summary = _group_means(frame_scores, test_ids, film_ids)
        methods[name] = {"train": train_summary, "test": test_summary}
        logger.info(
            "phase4 method=%s split=test iou=%.6f film_iou=%s pplastic_iou=%s",
            name,
            test_summary["iou"],
            test_summary["film_iou"],
            test_summary["pplastic_iou"],
        )
        logger.info(
            "phase4 method=%s split=train iou=%.6f film_iou=%s pplastic_iou=%s",
            name,
            train_summary["iou"],
            train_summary["film_iou"],
            train_summary["pplastic_iou"],
        )

    n_train_pixels = int(sum(by_index[index].mask.size for index in train_ids))
    n_train_film = int(sum(int(by_index[index].mask.sum()) for index in train_ids))
    payload = {
        "definition_iou": IOU_DEFINITION,
        "definition_dolp": DOLP_DEFINITION,
        "definition_color": COLOR_DEFINITION,
        "s0_min": S0_MIN,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "class_weight": "balanced",
        "standardize": "training-pixel mean and NumPy std (divide by N). Zero-variance columns use scale 1.",
        "split": {
            "train": list(train_ids),
            "test": list(test_ids),
            "train_film": [index for index in train_ids if index in film_ids],
            "train_pplastic": [index for index in train_ids if index not in film_ids],
            "test_film": [index for index in test_ids if index in film_ids],
            "test_pplastic": [index for index in test_ids if index not in film_ids],
        },
        "features": {name: list(columns) for name, columns in feature_columns().items()},
        "color_threshold": threshold,
        "n_train_pixels": n_train_pixels,
        "n_train_film_pixels": n_train_film,
        "train_film_fraction": float(n_train_film / n_train_pixels) if n_train_pixels else 0.0,
        "standardize_params": scales,
        "methods": methods,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_comparison(output_dir / "comparison.png", methods)
    present = [index for index in overlay_ids if index in by_index]
    if present:
        _write_overlays(output_dir / "overlays.png", [by_index[index] for index in present], predictions)
    return payload


def _write_comparison(path: Path, methods: dict) -> None:
    labels = ["test", "test film", "test pplastic"]
    keys = ("iou", "film_iou", "pplastic_iou")
    series = []
    for label, key in zip(labels, keys):
        values = [methods[name]["test"][key] for name in METHOD_ORDER]
        if all(value is not None for value in values):
            series.append((label, values))
    x = np.arange(len(METHOD_ORDER))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for offset, (label, values) in enumerate(series):
        shift = (offset - (len(series) - 1) / 2.0) * width
        ax.bar(x + shift, values, width, label=label)
    ax.set_xticks(x, METHOD_ORDER, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean frame IoU")
    ax.set_title("Test IoU. Both-empty frames score 1.")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_overlays(path: Path, frames: list[Frame], predictions: dict[str, dict[int, np.ndarray]]) -> None:
    columns = (
        ("RGB", None),
        ("mask", "mask"),
        ("color_threshold", "color_threshold"),
        ("rgb", "rgb"),
        ("rgb_dolp", "rgb_dolp"),
        ("rgb_dolp_cube", "rgb_dolp_cube"),
    )
    fig, axes = plt.subplots(len(frames), len(columns), figsize=(2.4 * len(columns), 2.6 * len(frames)))
    if len(frames) == 1:
        axes = np.array([axes])
    for row, frame in enumerate(frames):
        for col, (title, source) in enumerate(columns):
            axis = axes[row, col]
            if source is None:
                axis.imshow(tonemap_srgb(frame.rgb))
            elif source == "mask":
                axis.imshow(frame.mask, cmap="gray", vmin=0, vmax=1)
            else:
                axis.imshow(predictions[source][frame.index], cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"{frame.index:03d} {title}", fontsize=9)
            axis.axis("off")
    fig.suptitle("Display RGB, ground-truth mask, and predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Probe film segmentation on the Phase 3 frames.")
    parser.add_argument("--phase3-root", type=Path, default=DEFAULT_PHASE3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if not (args.phase3_root / "frames" / "000" / "stokes" / "render.exr").is_file():
        raise FileNotFoundError(f"Phase 3 frames not found under {args.phase3_root}")
    logger.info("phase4 phase3_root=%s", args.phase3_root)
    frames = load_phase3(args.phase3_root)
    run_probe(frames, args.output_root)


if __name__ == "__main__":
    main()
