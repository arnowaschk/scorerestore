from __future__ import annotations

import pytest
from PIL import Image, ImageChops, ImageOps

from scorerestore.lilypond.masks import (
    MaskQAError,
    derive_background,
    threshold_coverage,
    validate_semantic_masks,
)


def test_mask_thresholding_is_deterministic_and_binary() -> None:
    coverage = Image.new("L", (4, 1))
    coverage.putdata([0, 127, 128, 255])

    mask = threshold_coverage(coverage, 0.5)

    assert list(mask.get_flattened_data()) == [0, 0, 255, 255]


def test_background_is_exact_complement_and_foreground_overlap_is_legal() -> None:
    staff = Image.new("L", (3, 2), 0)
    notation = Image.new("L", (3, 2), 0)
    text = Image.new("L", (3, 2), 0)
    staff.putpixel((1, 0), 255)
    notation.putpixel((1, 0), 255)
    text.putpixel((2, 1), 255)

    masks = {
        "staff": staff,
        "notation": notation,
        "text": text,
    }
    masks["background"] = derive_background(masks)

    validate_semantic_masks(masks)
    assert ImageChops.multiply(staff, notation).getbbox() is not None
    foreground_union = ImageOps.invert(masks["background"])
    assert ImageChops.multiply(masks["background"], foreground_union).getbbox() is None


def test_mask_qa_rejects_background_overlap() -> None:
    masks = {
        "staff": Image.new("L", (2, 2), 255),
        "notation": Image.new("L", (2, 2), 255),
        "text": Image.new("L", (2, 2), 0),
        "background": Image.new("L", (2, 2), 255),
    }

    with pytest.raises(MaskQAError, match="background"):
        validate_semantic_masks(masks)


def test_mask_qa_rejects_nonbinary_values() -> None:
    masks = {
        "staff": Image.new("L", (2, 2), 128),
        "notation": Image.new("L", (2, 2), 255),
        "text": Image.new("L", (2, 2), 0),
        "background": Image.new("L", (2, 2), 0),
    }

    with pytest.raises(MaskQAError, match="not binary"):
        validate_semantic_masks(masks)
