"""Readable OpenCV cleaning baseline for degraded sheet-music pages."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from scorerestore.baselines.config import BaselineConfig, BaselineVariantConfig


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """One binary cleaned page plus inspectable intermediate metadata."""

    image: Image.Image
    normalized: Image.Image
    threshold_value: float | None


def clean_classical(image: Image.Image, config: BaselineConfig) -> BaselineResult:
    """Clean one page with the first variant, retained as a small single-result API.

    Use :func:`clean_classical_variant` to select an explicit comparison variant.
    """

    return clean_classical_variant(image, config, config.variants[0])


def clean_classical_variant(
    image: Image.Image,
    config: BaselineConfig,
    variant: BaselineVariantConfig,
) -> BaselineResult:
    """Clean one page with one explicit member of the comparison suite.

    Grayscale intensity follows the project convention: ``0 = black`` and ``255 = white``.
    The returned cleaned image is binary with black ink on a white background. Morphology operates
    on the inverted ink mask so ``open`` removes small foreground speckles rather than staff ink.
    """

    normalized = normalize_illumination(image, config)
    return threshold_variant(normalized, config, variant)


def normalize_illumination(image: Image.Image, config: BaselineConfig) -> np.ndarray:
    """Estimate and divide out one smooth illumination field for all four variants."""

    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    factor = config.illumination.downsample_factor
    reduced_width = max(1, round(grayscale.shape[1] / factor))
    reduced_height = max(1, round(grayscale.shape[0] / factor))
    reduced = cv2.resize(
        grayscale,
        (reduced_width, reduced_height),
        interpolation=cv2.INTER_AREA,
    )
    reduced_background = cv2.GaussianBlur(
        reduced,
        (0, 0),
        sigmaX=config.illumination.gaussian_sigma / factor,
        sigmaY=config.illumination.gaussian_sigma / factor,
        borderType=cv2.BORDER_REPLICATE,
    )
    background = cv2.resize(
        reduced_background,
        (grayscale.shape[1], grayscale.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.divide(grayscale, np.maximum(background, 1), scale=255)


def threshold_variant(
    normalized: np.ndarray,
    config: BaselineConfig,
    variant: BaselineVariantConfig,
) -> BaselineResult:
    """Threshold a shared normalized page and optionally apply the shared morphology."""

    threshold_value: float | None
    if variant.threshold_method == "otsu":
        threshold_value, binary = cv2.threshold(
            normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        threshold_value = float(threshold_value)
    else:
        binary = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            config.threshold.adaptive_block_size,
            config.threshold.adaptive_c,
        )
        threshold_value = None

    cleaned = _apply_morphology(binary, config) if variant.apply_morphology else binary
    return BaselineResult(
        image=Image.fromarray(cleaned),
        normalized=Image.fromarray(normalized),
        threshold_value=threshold_value,
    )


def _apply_morphology(binary: np.ndarray, config: BaselineConfig) -> np.ndarray:
    operation = config.morphology.operation
    ink = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morphology.kernel_size, config.morphology.kernel_size),
    )
    operations = {
        "open": (cv2.MORPH_OPEN,),
        "close": (cv2.MORPH_CLOSE,),
        "open_close": (cv2.MORPH_OPEN, cv2.MORPH_CLOSE),
    }
    for morphology_operation in operations[operation]:
        ink = cv2.morphologyEx(
            ink,
            morphology_operation,
            kernel,
            iterations=config.morphology.iterations,
        )
    return cv2.bitwise_not(ink)
