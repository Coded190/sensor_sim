"""Per-image segmentation scores.

IoU is TP / (TP + FP + FN). Precision is TP / (TP + FP). Recall is
TP / (TP + FN). A prediction is positive when the probability is above
the threshold. An image with no positive label and no positive prediction
scores 1 for all three. A zero denominator otherwise scores 0.
"""

from __future__ import annotations

import numpy as np


def counts(label: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, int]:
    """TP, FP, FN, and TN. Positive means probability > threshold."""
    truth = np.asarray(label, dtype=bool)
    prediction = np.asarray(probability, dtype=np.float64) > float(threshold)
    if truth.shape != prediction.shape:
        raise ValueError(f"label {truth.shape} != probability {prediction.shape}")
    true_positive = int(np.count_nonzero(prediction & truth))
    false_positive = int(np.count_nonzero(prediction & ~truth))
    false_negative = int(np.count_nonzero(~prediction & truth))
    true_negative = int(np.count_nonzero(~prediction & ~truth))
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
    }


def _ratio(numerator: int, denominator: int, empty: bool) -> float:
    if denominator == 0:
        return 1.0 if empty else 0.0
    return numerator / denominator


def scores(label: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """IoU, precision, and recall for one image."""
    tally = counts(label, probability, threshold)
    empty = tally["tp"] == 0 and tally["fp"] == 0 and tally["fn"] == 0
    iou = _ratio(tally["tp"], tally["tp"] + tally["fp"] + tally["fn"], empty)
    precision = _ratio(tally["tp"], tally["tp"] + tally["fp"], empty)
    recall = _ratio(tally["tp"], tally["tp"] + tally["fn"], empty)
    return {"iou": iou, "precision": precision, "recall": recall, **tally}


def mean_std(values: list[float]) -> dict[str, float]:
    """Mean and NumPy standard deviation (divide by N). One value has std 0."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mean": float(array.mean()), "std": float(array.std()), "n": int(array.size)}
