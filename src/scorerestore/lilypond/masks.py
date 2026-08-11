"""Binary semantic-mask construction and invariant checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PIL import Image, ImageChops, ImageOps

from scorerestore.lilypond.constants import SEMANTIC_FOREGROUND_CLASSES


class MaskQAError(ValueError):
    """Raised when aligned semantic-mask invariants are violated."""


def threshold_coverage(coverage: Image.Image, threshold: float) -> Image.Image:
    """Convert grayscale foreground coverage to ``0/255`` binary target values."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be greater than 0.0 and at most 1.0")
    cutoff = round(threshold * 255)
    return coverage.convert("L").point(lambda value: 255 if value >= cutoff else 0, mode="L")


def derive_background(foreground_masks: Mapping[str, Image.Image]) -> Image.Image:
    """Derive background as the exact logical complement of foreground union."""

    missing = set(SEMANTIC_FOREGROUND_CLASSES) - set(foreground_masks)
    if missing:
        raise MaskQAError(f"missing foreground masks: {', '.join(sorted(missing))}")
    dimensions = {foreground_masks[name].size for name in SEMANTIC_FOREGROUND_CLASSES}
    if len(dimensions) != 1:
        raise MaskQAError("foreground mask dimensions differ")
    union = ImageChops.lighter(
        foreground_masks["staff"],
        ImageChops.lighter(foreground_masks["notation"], foreground_masks["text"]),
    )
    return ImageOps.invert(union)


def validate_semantic_masks(
    masks: Mapping[str, Image.Image],
    *,
    expected_nonempty: Sequence[str] = ("staff", "notation"),
) -> None:
    """Validate dimensions, binary values, nonempty expectations, and background semantics."""

    required = {"background", *SEMANTIC_FOREGROUND_CLASSES}
    missing = required - set(masks)
    if missing:
        raise MaskQAError(f"missing semantic masks: {', '.join(sorted(missing))}")
    dimensions = {masks[name].size for name in required}
    if len(dimensions) != 1:
        raise MaskQAError("semantic mask dimensions differ")
    for name in required:
        histogram = masks[name].convert("L").histogram()
        if sum(histogram[1:255]) != 0:
            raise MaskQAError(f"{name} mask is not binary")
    for name in expected_nonempty:
        if name not in SEMANTIC_FOREGROUND_CLASSES:
            raise MaskQAError(f"unknown expected foreground class: {name}")
        if masks[name].getbbox() is None:
            raise MaskQAError(f"expected nonempty {name} mask")

    foreground = {name: masks[name] for name in SEMANTIC_FOREGROUND_CLASSES}
    expected_background = derive_background(foreground)
    if ImageChops.difference(masks["background"], expected_background).getbbox() is not None:
        raise MaskQAError("background is not the foreground complement")
    foreground_union = ImageOps.invert(expected_background)
    if ImageChops.multiply(masks["background"], foreground_union).getbbox() is not None:
        raise MaskQAError("background overlaps foreground")


def foreground_pixel_count(mask: Image.Image) -> int:
    """Count foreground pixels in a validated binary mask."""

    return mask.convert("L").histogram()[255]
