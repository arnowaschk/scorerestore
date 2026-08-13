from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from scorerestore.baselines import cleaning_metrics, restoration_ssim


def test_cleaning_metrics_use_black_as_foreground_and_omit_accuracy() -> None:
    target = Image.fromarray(np.array([[0, 0], [255, 255]], dtype=np.uint8))
    predicted = Image.fromarray(np.array([[0, 255], [0, 255]], dtype=np.uint8))

    metrics = cleaning_metrics(predicted, target)

    assert (metrics.true_positive, metrics.false_positive) == (1, 1)
    assert (metrics.false_negative, metrics.true_negative) == (1, 1)
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == metrics.dice == 0.5
    assert metrics.iou == pytest.approx(1 / 3)
    assert "accuracy" not in metrics.to_dict()


def test_identical_cleaning_images_have_perfect_scores() -> None:
    image = Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8))

    metrics = cleaning_metrics(image, image)

    assert metrics.precision == metrics.recall == metrics.f1 == metrics.dice == 1.0
    assert metrics.iou == 1.0
    assert metrics.ssim == pytest.approx(1.0)


def test_cleaning_metrics_reject_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions must match"):
        cleaning_metrics(Image.new("L", (2, 2)), Image.new("L", (3, 2)))


def test_restoration_ssim_uses_continuous_grayscale_values() -> None:
    target = Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8))
    continuous = Image.fromarray(np.array([[20, 220], [230, 15]], dtype=np.uint8))

    assert restoration_ssim(continuous, target) < 1.0
