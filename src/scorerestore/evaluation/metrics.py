"""Segmentation and independent-channel consistency metrics for V1 evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

SEMANTIC_CLASSES = ("background", "staff", "notation", "text")


def segmentation_metrics(
    predicted: dict[str, Image.Image], target: dict[str, Image.Image]
) -> dict[str, Any]:
    """Report per-channel, macro, and foreground-macro independent-sigmoid metrics."""

    if set(predicted) != set(SEMANTIC_CLASSES) or set(target) != set(SEMANTIC_CLASSES):
        raise ValueError("predicted and target masks must contain the four V1 semantic classes")
    per_class = {name: _binary_metrics(predicted[name], target[name]) for name in SEMANTIC_CLASSES}
    macro = _mean_metrics(per_class.values())
    foreground = _mean_metrics(per_class[name] for name in SEMANTIC_CLASSES[1:])
    return {"per_class": per_class, "macro": macro, "foreground_macro": foreground}


def consistency_metrics(predicted: dict[str, Image.Image]) -> dict[str, float]:
    """Measure illegal background overlap and unassigned pixels after thresholding."""

    if set(predicted) != set(SEMANTIC_CLASSES):
        raise ValueError("predicted masks must contain the four V1 semantic classes")
    arrays = {
        name: np.asarray(predicted[name].convert("L"), dtype=np.uint8) >= 128
        for name in SEMANTIC_CLASSES
    }
    dimensions = {array.shape for array in arrays.values()}
    if len(dimensions) != 1:
        raise ValueError("predicted masks must have equal dimensions")
    foreground = arrays["staff"] | arrays["notation"] | arrays["text"]
    return {
        "all_false_rate": float(np.mean(~arrays["background"] & ~foreground)),
        "background_overlap_rate": float(np.mean(arrays["background"] & foreground)),
    }


def _binary_metrics(predicted: Image.Image, target: Image.Image) -> dict[str, int | float]:
    if predicted.size != target.size:
        raise ValueError("segmentation dimensions must match")
    predicted_array = np.asarray(predicted.convert("L"), dtype=np.uint8) >= 128
    target_array = np.asarray(target.convert("L"), dtype=np.uint8) >= 128
    true_positive = int(np.count_nonzero(predicted_array & target_array))
    false_positive = int(np.count_nonzero(predicted_array & ~target_array))
    false_negative = int(np.count_nonzero(~predicted_array & target_array))
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    dice = _ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": _ratio(true_positive, true_positive + false_positive + false_negative),
    }


def _mean_metrics(values: Any) -> dict[str, float]:
    rows = list(values)
    return {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in ("precision", "recall", "dice", "iou")
    }


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator
