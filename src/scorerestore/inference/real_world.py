"""Unannotated real-world comparison inference with transparent checkpoint selection."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from scorerestore.baselines import clean_classical_variant, load_baseline_config

from .io import read_input_pages
from .real_world_config import RealWorldComparisonConfig
from .real_world_quality import quality_row, write_quality_report
from .tiled import clean, load_checkpoint_model

_LABEL_HEIGHT = 28


class RealWorldComparisonError(ValueError):
    """Raised when real-world comparison inputs or checkpoint selection are invalid."""


@dataclass(frozen=True, slots=True)
class CheckpointSelection:
    """One explicitly requested or deterministically discovered V1 checkpoint."""

    path: Path
    backend: str
    validation_loss: float | None
    selection: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


@dataclass(frozen=True, slots=True)
class RealWorldComparisonResult:
    """Artifacts from one comparison run over every bundled real-world PDF."""

    output_directory: Path
    comparison_pdf: Path
    pdf_count: int
    page_count: int
    selections: dict[str, CheckpointSelection]


def compare_real_world(
    config: RealWorldComparisonConfig,
    output_directory: str | Path,
    *,
    checkpoint_overrides: dict[str, Path] | None = None,
    update: bool = False,
) -> RealWorldComparisonResult:
    """Clean every real-world PDF with every ordered YAML model panel.

    The generated PDF keeps every rasterized panel at its native dimensions; it never reduces a
    source page merely to fit a display page. Automatic selection ranks eligible checkpoints only by
    their saved training validation loss, which is recorded as a selection heuristic rather than a
    quality claim across incomparable runs.
    """

    output = Path(output_directory).resolve()
    if output.exists() and not update:
        raise RealWorldComparisonError(f"output directory already exists: {output}")
    if output.exists() and not output.is_dir():
        raise RealWorldComparisonError(f"comparison output is not a directory: {output}")
    sources = _source_pdfs(config.input_root)
    overrides = checkpoint_overrides or {}
    unknown_overrides = set(overrides) - {model.identifier for model in config.models}
    if unknown_overrides:
        raise RealWorldComparisonError(
            f"unknown model checkpoint overrides: {', '.join(sorted(unknown_overrides))}"
        )
    selections = {
        model.identifier: _select_checkpoint(
            overrides.get(model.identifier, model.checkpoint),
            config.runs_root,
            backend=model.backend,
        )
        for model in config.models
    }
    classical_config = load_baseline_config(config.baseline_config)
    classical_variant = next(
        (
            variant
            for variant in classical_config.variants
            if variant.name == config.baseline_variant
        ),
        None,
    )
    if classical_variant is None:
        raise RealWorldComparisonError(
            f"unknown classical baseline variant: {config.baseline_variant}"
        )
    loaded_models = {
        model.identifier: load_checkpoint_model(
            selections[model.identifier].path, device=config.device
        )
        for model in config.models
    }

    output.mkdir(parents=True, exist_ok=update)
    sheets: list[Image.Image] = []
    pages: list[dict[str, Any]] = []
    quality_rows: list[dict[str, str | int | float | None]] = []
    try:
        for source in sources:
            source_key = _source_key(source, sources)
            for page in read_input_pages(source, pdf_dpi=config.pdf_dpi):
                classical_result = clean_classical_variant(
                    page.image, classical_config, classical_variant
                )
                results = {
                    model.identifier: clean(
                        page.image,
                        model=loaded_models[model.identifier][0],
                        device=config.device,
                        tile_size=config.tile_size,
                        overlap=config.overlap,
                        cleaning_threshold=config.cleaning_threshold,
                        segmentation_threshold=config.segmentation_threshold,
                    )
                    for model in config.models
                }
                page_name = f"page-{page.page_number:04d}.png"
                original_path = output / "original" / source_key / page_name
                classical_path = output / "classical_cleaned" / source_key / page_name
                _save_png(page.image, original_path)
                _save_png(classical_result.image, classical_path)
                model_paths = {
                    model.identifier: output / model.identifier / source_key / page_name
                    for model in config.models
                }
                for model in config.models:
                    _save_png(results[model.identifier].cleaned, model_paths[model.identifier])
                quality_rows.append(
                    quality_row(
                        page.image,
                        classical_result.image,
                        source=str(source.resolve()),
                        page_number=page.page_number,
                        comparison_id="classical_cleaned",
                        comparison_label="OpenCV classical cleaned",
                    )
                )
                for model in config.models:
                    quality_rows.append(
                        quality_row(
                            page.image,
                            results[model.identifier].cleaned,
                            source=str(source.resolve()),
                            page_number=page.page_number,
                            comparison_id=model.identifier,
                            comparison_label=model.label,
                            tile_size=config.tile_size,
                            overlap=config.overlap,
                        )
                    )
                sheets.append(
                    _comparison_sheet(
                        page.image,
                        classical_result.image,
                        tuple(
                            (model.label, results[model.identifier].cleaned)
                            for model in config.models
                        ),
                    )
                )
                pages.append(
                    {
                        "source": str(source.resolve()),
                        "page_number": page.page_number,
                        "input_dpi": page.dpi,
                        "dimensions": {"width": page.image.width, "height": page.image.height},
                        "original": str(original_path.relative_to(output)),
                        "classical_cleaned": str(classical_path.relative_to(output)),
                        "models": {
                            model.identifier: {
                                "cleaned": str(model_paths[model.identifier].relative_to(output)),
                                "tile_count": results[model.identifier].metadata["tile_count"],
                            }
                            for model in config.models
                        },
                    }
                )
        comparison_pdf = output / "comparison.pdf"
        _save_pdf(sheets, comparison_pdf, config.pdf_dpi)
        quality_report = write_quality_report(output, quality_rows)
        _write_metadata(
            output,
            sources,
            pages,
            config,
            selections,
            {identifier: metadata for identifier, (_, metadata) in loaded_models.items()},
            quality_report,
        )
    finally:
        for sheet in sheets:
            sheet.close()
    return RealWorldComparisonResult(
        output,
        output / "comparison.pdf",
        len(sources),
        len(pages),
        selections,
    )


def _source_pdfs(root: Path) -> tuple[Path, ...]:
    source = root.resolve()
    if not source.is_dir():
        raise RealWorldComparisonError(f"real-world input directory does not exist: {source}")
    pdfs = tuple(
        sorted(
            path for path in source.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"
        )
    )
    if not pdfs:
        raise RealWorldComparisonError(f"real-world input directory contains no PDFs: {source}")
    return pdfs


def _select_checkpoint(
    requested: str | Path | None, runs_root: str | Path, *, backend: str
) -> CheckpointSelection:
    if requested is not None:
        path = Path(requested).resolve()
        metadata = _checkpoint_metadata(path)
        _validate_backend(metadata, backend, path)
        return CheckpointSelection(path, backend, metadata["validation_loss"], "explicit_override")

    root = Path(runs_root).resolve()
    candidates = []
    if root.is_dir():
        for path in sorted(root.glob("**/checkpoints/best.pt")):
            try:
                metadata = _checkpoint_metadata(path)
                _validate_backend(metadata, backend, path)
            except RealWorldComparisonError:
                continue
            candidates.append((metadata["validation_loss"], path.resolve()))
    if not candidates:
        raise RealWorldComparisonError(
            f"no eligible {backend} best.pt checkpoint below {root}; provide an explicit checkpoint"
        )
    loss, path = min(candidates, key=lambda item: (item[0], str(item[1])))
    return CheckpointSelection(path, backend, loss, "automatic_lowest_validation_loss")


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise RealWorldComparisonError(f"cannot read checkpoint {path}: {error}") from error
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise RealWorldComparisonError(f"checkpoint lacks V1 model configuration: {path}")
    validation_loss = checkpoint.get("validation_loss")
    if isinstance(validation_loss, bool) or not isinstance(validation_loss, (int, float)):
        raise RealWorldComparisonError(f"checkpoint has no numeric validation loss: {path}")
    if not math.isfinite(float(validation_loss)):
        raise RealWorldComparisonError(f"checkpoint validation loss is not finite: {path}")
    config = dict(checkpoint["config"])
    config.setdefault("model_backend", "unet")
    config.setdefault("pretrained", False)
    config.setdefault("freeze_batch_norm", True)
    return {"config": config, "validation_loss": float(validation_loss)}


def _validate_backend(metadata: dict[str, Any], backend: str, path: Path) -> None:
    config = metadata["config"]
    if config.get("model_backend") != backend:
        raise RealWorldComparisonError(
            f"checkpoint {path} uses {config.get('model_backend')!r}, expected {backend!r}"
        )
    if config.get("task") not in {"clean", "multitask"}:
        raise RealWorldComparisonError(
            f"checkpoint {path} has task {config.get('task')!r}; it was not trained for cleaning"
        )


def _source_key(source: Path, sources: tuple[Path, ...]) -> str:
    matching = [path for path in sources if path.stem == source.stem]
    return source.stem if len(matching) == 1 else f"{source.stem}-{sources.index(source) + 1}"


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=9)


def _comparison_sheet(
    original: Image.Image,
    classical: Image.Image,
    models: tuple[tuple[str, Image.Image], ...],
) -> Image.Image:
    """Place exact-size panels on a landscape canvas without resampling any source pixels."""

    images = (original, classical, *(image for _, image in models))
    if len(images) < 3:
        raise RealWorldComparisonError(
            "comparison needs original, classical, and at least one model"
        )
    if {image.size for image in images} != {original.size}:
        raise RealWorldComparisonError("comparison page images must have identical dimensions")
    width, height = original.size
    canvas_width = max(width * len(images), height + _LABEL_HEIGHT + 1)
    sheet = Image.new("RGB", (canvas_width, height + _LABEL_HEIGHT), "white")
    draw = ImageDraw.Draw(sheet)
    panels = (("Original", original), ("OpenCV classical cleaned", classical), *models)
    for index, (label, image) in enumerate(panels):
        offset = index * width
        draw.text((offset + 4, 6), label, fill="black")
        sheet.paste(image.convert("RGB"), (offset, _LABEL_HEIGHT))
    return sheet


def _save_pdf(sheets: list[Image.Image], path: Path, dpi: int) -> None:
    if not sheets:
        raise RealWorldComparisonError("no pages were generated for comparison PDF")
    sheets[0].save(
        path,
        format="PDF",
        save_all=True,
        append_images=sheets[1:],
        resolution=dpi,
        quality=100,
        subsampling=0,
    )


def _write_metadata(
    output: Path,
    sources: tuple[Path, ...],
    pages: list[dict[str, Any]],
    config: RealWorldComparisonConfig,
    selections: dict[str, CheckpointSelection],
    checkpoint_metadata: dict[str, dict[str, Any]],
    quality_report: dict[str, str],
) -> None:
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "input_pdfs": [str(path.resolve()) for path in sources],
        "comparison_pdf": "comparison.pdf",
        "quality_report": quality_report,
        "selection_note": (
            "Automatic selection uses saved training validation loss only and is not a "
            "cross-dataset quality comparison."
        ),
        "config": config.to_dict(),
        "models": {
            model.identifier: {
                "label": model.label,
                "selection": selections[model.identifier].to_dict(),
                "checkpoint": checkpoint_metadata[model.identifier],
            }
            for model in config.models
        },
        "inference": {
            "device": config.device,
            "tile_size": config.tile_size,
            "overlap": config.overlap,
            "pdf_dpi": config.pdf_dpi,
            "cleaning_threshold": config.cleaning_threshold,
            "segmentation_threshold": config.segmentation_threshold,
            "classical_baseline": {
                "config": str(config.baseline_config),
                "variant": config.baseline_variant,
            },
            "panel_layout": [
                "Original",
                "OpenCV classical cleaned",
                *(model.label for model in config.models),
            ],
            "resampling": "none; panels retain native raster dimensions",
        },
        "pages": pages,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
