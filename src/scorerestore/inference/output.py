"""Filesystem output adapter for V1 tiled inference."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .io import InputPage
from .tiled import CleanResult, metadata_json


def planned_output_paths(root: Path, pages: list[InputPage], *, overlay: bool) -> tuple[Path, ...]:
    """Return the exact paths an inference run will write, without touching the filesystem."""

    paths = [root / "metadata.json"]
    for page in pages:
        page_root = root / f"page-{page.page_number:04d}"
        probabilities = page_root / "probabilities"
        masks = page_root / "masks"
        paths.extend(
            (
                page_root / "metadata.json",
                page_root / "cleaned.png",
                probabilities / "cleaning.png",
                *(
                    probabilities / f"{name}.png"
                    for name in ("background", "staff", "notation", "text")
                ),
                *(masks / f"{name}.png" for name in ("background", "staff", "notation", "text")),
            )
        )
        if overlay:
            paths.append(page_root / "overlay.png")
    return tuple(paths)


def cleaned_pdf_path(root: Path, input_path: Path) -> Path:
    """Return the user-facing PDF path for one inference invocation."""

    return root / f"{input_path.stem}_scorerestore.pdf"


def write_cleaned_pdf(root: Path, pages: list[InputPage], *, input_path: Path) -> Path:
    """Write all cleaned page rasters as one PDF, atomically replacing the final path."""

    if not pages:
        raise ValueError("cannot create a cleaned PDF without input pages")
    destination = cleaned_pdf_path(root, input_path)
    images: list[Image.Image] = []
    temporary: Path | None = None
    try:
        for page in pages:
            with Image.open(root / f"page-{page.page_number:04d}" / "cleaned.png") as image:
                images.append(image.convert("L"))
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.stem}-", suffix=".pdf", dir=root, delete=False
        ) as handle:
            temporary = Path(handle.name)
        images[0].save(
            temporary,
            format="PDF",
            save_all=True,
            append_images=images[1:],
            resolution=pages[0].dpi or 72,
        )
        temporary.replace(destination)
        return destination
    finally:
        for image in images:
            image.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()


def remove_page_outputs(root: Path, pages: list[InputPage]) -> None:
    """Remove the intermediate outputs after the final cleaned PDF is safely written."""

    metadata = root / "metadata.json"
    if metadata.exists():
        metadata.unlink()
    for page in pages:
        page_root = root / f"page-{page.page_number:04d}"
        if page_root.exists():
            shutil.rmtree(page_root)


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
    probabilities.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
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
