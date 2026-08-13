"""Cleaning metrics used by the Milestone 5 baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class CleaningMetrics:
    """Foreground confusion counts, derived scores, and grayscale SSIM."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float
    dice: float
    iou: float
    ssim: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "dice": self.dice,
            "iou": self.iou,
            "ssim": self.ssim,
        }


def cleaning_metrics(
    predicted_clean: Image.Image,
    target_clean: Image.Image,
    *,
    target_ink_threshold: float = 0.5,
) -> CleaningMetrics:
    """Measure black foreground against thresholded antialiased pristine ink.

    The pristine target is converted from grayscale intensity to ink coverage as
    ``1 - intensity`` before thresholding. Overall pixel accuracy is intentionally omitted because
    sheet-music backgrounds dominate it.
    """

    if predicted_clean.size != target_clean.size:
        raise ValueError("predicted and target dimensions must match")
    if not 0.0 < target_ink_threshold <= 1.0:
        raise ValueError("target_ink_threshold must be in (0, 1]")
    predicted = np.asarray(predicted_clean.convert("L"), dtype=np.uint8) < 128
    target_intensity = np.asarray(target_clean.convert("L"), dtype=np.uint8)
    target = (255.0 - target_intensity.astype(np.float32)) / 255.0 >= target_ink_threshold
    true_positive = int(np.count_nonzero(predicted & target))
    false_positive = int(np.count_nonzero(predicted & ~target))
    false_negative = int(np.count_nonzero(~predicted & target))
    true_negative = int(predicted.size - true_positive - false_positive - false_negative)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    dice = _ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    iou = _ratio(true_positive, true_positive + false_positive + false_negative)
    return CleaningMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        precision=precision,
        recall=recall,
        f1=dice,
        dice=dice,
        iou=iou,
        ssim=restoration_ssim(predicted_clean, target_clean),
    )


def metrics_from_counts(
    *,
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
    mean_ssim: float,
) -> CleaningMetrics:
    """Create micro-averaged metrics from accumulated confusion counts."""

    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    dice = _ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    return CleaningMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        precision=precision,
        recall=recall,
        f1=dice,
        dice=dice,
        iou=_ratio(true_positive, true_positive + false_positive + false_negative),
        ssim=mean_ssim,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def restoration_ssim(predicted: Image.Image, target: Image.Image) -> float:
    """Measure grayscale structural similarity without applying a binary threshold."""

    if predicted.size != target.size:
        raise ValueError("predicted and target dimensions must match")
    return _ssim(
        np.asarray(predicted.convert("L"), dtype=np.uint8),
        np.asarray(target.convert("L"), dtype=np.uint8),
    )


def _ssim(first: np.ndarray, second: np.ndarray) -> float:
    """Compute mean local SSIM with standard scale selection and an 11px Gaussian window."""

    first_float = first.astype(np.float32)
    second_float = second.astype(np.float32)
    scale = max(1, round(min(first.shape) / 256))
    if scale > 1:
        dimensions = (max(1, first.shape[1] // scale), max(1, first.shape[0] // scale))
        first_float = cv2.resize(first_float, dimensions, interpolation=cv2.INTER_AREA)
        second_float = cv2.resize(second_float, dimensions, interpolation=cv2.INTER_AREA)
    mean_first = cv2.GaussianBlur(first_float, (11, 11), 1.5)
    mean_second = cv2.GaussianBlur(second_float, (11, 11), 1.5)
    variance_first = cv2.GaussianBlur(first_float * first_float, (11, 11), 1.5)
    variance_first -= mean_first * mean_first
    variance_second = cv2.GaussianBlur(second_float * second_float, (11, 11), 1.5)
    variance_second -= mean_second * mean_second
    covariance = cv2.GaussianBlur(first_float * second_float, (11, 11), 1.5)
    covariance -= mean_first * mean_second
    constant_one = (0.01 * 255) ** 2
    constant_two = (0.03 * 255) ** 2
    numerator = (2 * mean_first * mean_second + constant_one) * (2 * covariance + constant_two)
    denominator = (mean_first**2 + mean_second**2 + constant_one) * (
        variance_first + variance_second + constant_two
    )
    score = numerator / np.maximum(denominator, np.finfo(np.float32).eps)
    return float(np.clip(np.mean(score), -1.0, 1.0))
