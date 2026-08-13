"""Filesystem output adapter for V1 tiled inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .io import InputPage
from .tiled import CleanResult, metadata_json


def write_page_outputs(
    root: Path,
    page: InputPage,
    result: CleanResult,
    *,
    input_path: Path,
    checkpoint_metadata: dict[str, Any],
    overlay: bool,
) -> dict[str, str]:
    """Write one page under a deterministic directory and return paths relative to *root*."""

    page_root = root / f"page-{page.page_number:04d}"
    probabilities = page_root / "probabilities"
    masks = page_root / "masks"
    probabilities.mkdir(parents=True)
    masks.mkdir(parents=True)
    paths: dict[str, str] = {}
    metadata_path = page_root / "metadata.json"
    paths["metadata"] = str(metadata_path.relative_to(root))
    paths["cleaned"] = _save(result.cleaned, page_root / "cleaned.png", root)
    paths["cleaning_probability"] = _save(result.probability, probabilities / "cleaning.png", root)
    for name, image in result.mask_probabilities.items():
        paths[f"{name}_probability"] = _save(image, probabilities / f"{name}.png", root)
    for name, image in result.masks.items():
        paths[f"{name}_mask"] = _save(image, masks / f"{name}.png", root)
    if overlay:
        paths["overlay"] = _save(
            _overlay(page.image, result.masks), page_root / "overlay.png", root
        )
    metadata = {
        "input": str(input_path.resolve()),
        "page_number": page.page_number,
        "input_dpi": page.dpi,
        "outputs": paths,
        "inference": result.metadata,
        **checkpoint_metadata,
    }
    metadata_path.write_text(metadata_json(metadata), encoding="utf-8")
    return paths


def write_run_metadata(root: Path, pages: list[dict[str, str]]) -> None:
    (root / "metadata.json").write_text(
        json.dumps({"pages": pages}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _save(image: Image.Image, path: Path, root: Path) -> str:
    image.save(path, format="PNG", compress_level=9)
    return str(path.relative_to(root))


def _overlay(image: Image.Image, masks: dict[str, Image.Image]) -> Image.Image:
    base = np.asarray(image.convert("L"), dtype=np.float32)
    result = np.repeat(base[:, :, None], 3, axis=2)
    # Foreground overlaps are deliberate; successive blends make them inspectable.
    for name, color in (
        ("staff", (45, 115, 255)),
        ("notation", (235, 55, 55)),
        ("text", (45, 170, 85)),
    ):
        selected = np.asarray(masks[name], dtype=np.uint8) > 0
        result[selected] = 0.45 * result[selected] + 0.55 * np.asarray(color, dtype=np.float32)
    return Image.fromarray(result.clip(0, 255).astype(np.uint8), mode="RGB")
