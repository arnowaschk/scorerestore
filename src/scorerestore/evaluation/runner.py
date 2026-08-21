"""Measured V1 model evaluation, deterministic visual sheets, and runtime benchmarking."""

from __future__ import annotations

import csv
import io
import json
import platform
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageOps

from scorerestore.baselines import (
    clean_classical_variant,
    cleaning_metrics,
    load_baseline_config,
    restoration_ssim,
)
from scorerestore.dataset import MaterializedDataset
from scorerestore.dataset.manifest import sha256_file
from scorerestore.inference import clean, load_checkpoint_model, read_input_pages

from .config import EvaluationConfig
from .metrics import consistency_metrics, segmentation_metrics


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    output_directory: Path
    summary_path: Path
    metrics_csv_path: Path
    metrics_jsonl_path: Path
    sample_count: int


def evaluate(
    config: EvaluationConfig, output_directory: str | Path, *, update: bool = False
) -> EvaluationResult:
    """Evaluate named checkpoints separately by split and create deterministic report artifacts."""

    output = Path(output_directory).resolve()
    if output.exists() and not update:
        raise ValueError(f"output directory already exists: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"evaluation output is not a directory: {output}")
    completed = _completed_evaluation(output)
    if update and completed is not None:
        return completed
    manifest = config.dataset_manifest.resolve()
    dataset = MaterializedDataset(manifest)
    records = tuple(record for record in dataset._records if record["split"] in config.splits)
    if not records:
        raise ValueError("selected splits contain no materialized samples")
    baseline_config = load_baseline_config(config.baseline_config)
    variant = next(
        (item for item in baseline_config.variants if item.name == config.baseline_variant), None
    )
    if variant is None:
        raise ValueError(f"unknown baseline variant: {config.baseline_variant}")
    selected_ids = _visual_ids(records, config.report_seed, config.report_samples)
    output.mkdir(parents=True, exist_ok=update)
    baseline_rows: list[dict[str, Any]] = []
    for record, sample in _prefetched_samples(dataset._load_cleaning, records):
        baseline = clean_classical_variant(sample.image, baseline_config, variant)
        baseline_rows.append(_baseline_metric_row(record, baseline.image, sample.clean))

    rows: list[dict[str, Any]] = [*baseline_rows]
    model_summaries: dict[str, Any] = {}
    for model_spec in config.models:
        model, checkpoint_metadata = load_checkpoint_model(
            model_spec.checkpoint, device=config.device
        )
        model_rows: list[dict[str, Any]] = []
        for record, sample in _prefetched_samples(dataset._load, records):
            result = clean(
                sample.image,
                model=model,
                device=config.device,
                tile_size=config.tile_size,
                overlap=config.overlap,
                cleaning_threshold=config.cleaning_threshold,
                segmentation_threshold=config.segmentation_threshold,
            )
            row = _metric_row(model_spec.name, record, result, sample)
            rows.append(row)
            model_rows.append(row)
            if record["sample_id"] in selected_ids:
                baseline = clean_classical_variant(sample.image, baseline_config, variant)
                _comparison_sheet(
                    output / "comparisons" / model_spec.name / f"{record['sample_id']}.png",
                    sample.image,
                    baseline.image,
                    result.cleaned,
                    sample.clean,
                    _overlay(sample.image, result.masks),
                )
        model_summaries[model_spec.name] = {
            "checkpoint": checkpoint_metadata,
            "splits": {
                split: _summarize([row for row in model_rows if row["split"] == split])
                for split in config.splits
            },
        }
    summary = {
        "schema_version": 1,
        "created_at": _timestamp(),
        "dataset_manifest": str(manifest),
        "dataset_manifest_sha256": sha256_file(manifest),
        "sample_count": len(records),
        "splits": list(config.splits),
        "models": model_summaries,
        "baseline": {
            "name": f"opencv_{config.baseline_variant}",
            "splits": {
                split: _summarize([row for row in baseline_rows if row["split"] == split])
                for split in config.splits
            },
        },
        "comparisons": _controlled_comparisons(model_summaries),
        "report_selection": {"seed": config.report_seed, "sample_ids": sorted(selected_ids)},
        "metric_notes": {
            "cleaning": (
                "foreground precision/recall/F1/Dice/IoU; overall pixel accuracy is omitted"
            ),
            "segmentation": "independent sigmoid channels; foreground macro excludes background",
            "challenge": "reported separately and never merged into test",
            "comparisons": "descriptive measured results only; no improvement claim is implied",
        },
    }
    (output / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    (output / "metrics.jsonl").write_text(_jsonl(rows), encoding="utf-8")
    (output / "metrics.csv").write_text(_csv(rows), encoding="utf-8")
    (output / "summary.json").write_text(_json(summary), encoding="utf-8")
    report = output / "report"
    report.mkdir(exist_ok=update)
    (report / "summary.md").write_text(_markdown_summary(summary), encoding="utf-8")
    return EvaluationResult(
        output,
        output / "summary.json",
        output / "metrics.csv",
        output / "metrics.jsonl",
        len(records),
    )


def benchmark(
    config: EvaluationConfig,
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str,
    update: bool = False,
) -> Path:
    """Measure actual tiled inference latency; never estimate or synthesize performance values."""

    model_spec = next((item for item in config.models if item.name == model_name), None)
    if model_spec is None:
        raise ValueError(f"unknown model name: {model_name}")
    output = Path(output_path).resolve()
    if output.exists() and update and output.is_file():
        return output
    if output.exists():
        raise ValueError(f"benchmark output already exists: {output}")
    model, checkpoint = load_checkpoint_model(model_spec.checkpoint, device=config.device)
    pages = read_input_pages(input_path, pdf_dpi=config.pdf_dpi)
    device = torch.device(
        config.device if config.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    page_rows = []
    for page in pages:
        page_started = time.perf_counter()
        result = clean(
            page.image,
            model=model,
            device=config.device,
            tile_size=config.tile_size,
            overlap=config.overlap,
            cleaning_threshold=config.cleaning_threshold,
            segmentation_threshold=config.segmentation_threshold,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency = time.perf_counter() - page_started
        width, height = page.image.size
        page_rows.append(
            {
                "page_number": page.page_number,
                "dimensions": {"width": width, "height": height},
                "latency_seconds": latency,
                "megapixels_per_second": width * height / 1_000_000 / latency,
                "tile_count": result.metadata["tile_count"],
            }
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency = time.perf_counter() - started
    pixels = sum(row["dimensions"]["width"] * row["dimensions"]["height"] for row in page_rows)
    report = {
        "schema_version": 1,
        "label": "MEASURED",
        "created_at": _timestamp(),
        "input": str(Path(input_path).resolve()),
        "model_name": model_name,
        "checkpoint": checkpoint,
        "hardware": {
            "device": str(device),
            "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "tile_size": config.tile_size,
        "overlap": config.overlap,
        "pdf_dpi": config.pdf_dpi,
        "precision_mode": "float32",
        "total_latency_seconds": latency,
        "total_megapixels_per_second": pixels / 1_000_000 / latency,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "pages": page_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_json(report), encoding="utf-8")
    return output


def _prefetched_samples(
    loader: Callable[[dict[str, Any]], Any], records: tuple[dict[str, Any], ...]
) -> Iterator[tuple[dict[str, Any], Any]]:
    """Load one following sample while the caller processes the current sample.

    Evaluation is intentionally page-at-a-time: holding an entire rendered corpus in memory can
    exceed RAM even when each inference itself is tiled. One background loader overlaps PNG I/O
    and decoding with CPU/GPU work while keeping the resident working set bounded to two samples.
    """

    if not records:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="evaluation-loader") as executor:
        future = executor.submit(loader, records[0])
        for index, record in enumerate(records):
            sample = future.result()
            if index + 1 < len(records):
                future = executor.submit(loader, records[index + 1])
            yield record, sample


def _completed_evaluation(output: Path) -> EvaluationResult | None:
    summary_path = output / "summary.json"
    metrics_csv = output / "metrics.csv"
    metrics_jsonl = output / "metrics.jsonl"
    if not (summary_path.is_file() and metrics_csv.is_file() and metrics_jsonl.is_file()):
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sample_count = summary["sample_count"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        return None
    return EvaluationResult(output, summary_path, metrics_csv, metrics_jsonl, sample_count)


def _metric_row(
    model_name: str, record: dict[str, Any], result: Any, sample: Any
) -> dict[str, Any]:
    cleaning = cleaning_metrics(result.cleaned, sample.clean)
    segmentation = segmentation_metrics(result.masks, sample.masks)
    row: dict[str, Any] = {
        "evaluation_kind": "neural",
        "model": model_name,
        "sample_id": record["sample_id"],
        "source_id": record["source_id"],
        "split": record["split"],
        **{
            f"cleaning_{key}": (
                restoration_ssim(ImageOps.invert(result.probability), sample.clean)
                if key == "ssim"
                else value
            )
            for key, value in cleaning.to_dict().items()
        },
        **consistency_metrics(result.masks),
    }
    for scope, metrics in (
        ("segmentation_macro", segmentation["macro"]),
        ("segmentation_foreground_macro", segmentation["foreground_macro"]),
    ):
        row.update({f"{scope}_{key}": value for key, value in metrics.items()})
    for name, metrics in segmentation["per_class"].items():
        row.update({f"segmentation_{name}_{key}": value for key, value in metrics.items()})
    return row


def _baseline_metric_row(
    record: dict[str, Any], predicted: Image.Image, target: Image.Image
) -> dict[str, Any]:
    """Keep actual OpenCV cleaning measurements beside neural rows without fake masks."""

    cleaning = cleaning_metrics(predicted, target)
    return {
        "evaluation_kind": "classical_baseline",
        "model": "opencv_baseline",
        "sample_id": record["sample_id"],
        "source_id": record["source_id"],
        "split": record["split"],
        **{f"cleaning_{key}": value for key, value in cleaning.to_dict().items()},
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    metric_keys = [key for key, value in rows[0].items() if isinstance(value, float)]
    return {
        "sample_count": len(rows),
        **{key: float(np.mean([row[key] for row in rows])) for key in metric_keys},
    }


def _visual_ids(records: tuple[dict[str, Any], ...], seed: int, count: int) -> set[str]:
    ordered = sorted(record["sample_id"] for record in records)
    generator = np.random.default_rng(seed)
    return set(generator.choice(ordered, size=min(count, len(ordered)), replace=False).tolist())


def _comparison_sheet(path: Path, *images: Image.Image) -> None:
    labels = (
        "degraded",
        "OpenCV baseline",
        "DL cleaned",
        "pristine target",
        "segmentation overlay",
    )
    width, height = images[0].size
    panel_width = min(width, 360)
    panel_height = round(height * panel_width / width)
    sheet = Image.new("RGB", (panel_width * len(images), panel_height + 24), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(zip(labels, images, strict=True)):
        panel = image.convert("RGB").resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        sheet.paste(panel, (index * panel_width, 24))
        draw.text((index * panel_width + 4, 5), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG", compress_level=9)


def _overlay(image: Image.Image, masks: dict[str, Image.Image]) -> Image.Image:
    base = np.repeat(np.asarray(image.convert("L"), dtype=np.float32)[:, :, None], 3, axis=2)
    for name, color in (
        ("staff", (45, 115, 255)),
        ("notation", (235, 55, 55)),
        ("text", (45, 170, 85)),
    ):
        selected = np.asarray(masks[name], dtype=np.uint8) >= 128
        base[selected] = 0.45 * base[selected] + 0.55 * np.asarray(color, dtype=np.float32)
    return Image.fromarray(base.clip(0, 255).astype(np.uint8), mode="RGB")


def _controlled_comparisons(models: dict[str, Any]) -> dict[str, Any]:
    """Describe comparable candidates without claiming a winner or merging split measurements."""
    output: dict[str, Any] = {}
    names = list(models)
    for left in names:
        for right in names:
            if left >= right:
                continue
            first, second = models[left]["checkpoint"], models[right]["checkpoint"]
            first_config, second_config = (
                first.get("training_config", {}),
                second.get("training_config", {}),
            )
            comparable = _comparable_training_configs(first_config, second_config)
            tasks = {first_config.get("task"), second_config.get("task")}
            backends = {
                first.get("model", {}).get("backend"),
                second.get("model", {}).get("backend"),
            }
            comparison_type = "unqualified"
            if tasks == {"clean", "multitask"} and backends == {"unet"}:
                comparison_type = "cleaning_only_vs_multitask_custom_unet"
            elif backends == {"unet", "resnet18"}:
                comparison_type = "custom_unet_vs_resnet18_transfer"
            output[f"{left}_vs_{right}"] = {
                "models": [left, right],
                "comparison_type": comparison_type,
                "tasks": [first_config.get("task"), second_config.get("task")],
                "backends": [
                    first.get("model", {}).get("backend"),
                    second.get("model", {}).get("backend"),
                ],
                "controlled": comparable,
                "note": (
                    "Compare only matching splits, dataset manifests, budgets, and major "
                    "hyperparameters; this report makes no causal claim."
                ),
            }
    return output


def _comparable_training_configs(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Recognize a controlled clean-vs-multitask or U-Net-vs-transfer comparison conservatively."""

    ignored = {"task", "model_backend", "pretrained", "freeze_batch_norm"}
    return bool(first and second) and {
        key: value for key, value in first.items() if key not in ignored
    } == {key: value for key, value in second.items() if key not in ignored}


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# ScoreRestore measured evaluation",
        "",
        "All values below are **MEASURED**; challenge remains separate from test.",
        "",
    ]
    report_rows = [(f"OpenCV baseline ({summary['baseline']['name']})", summary["baseline"])]
    report_rows.extend(summary["models"].items())
    for name, model in report_rows:
        lines.extend(
            [
                f"## {name}",
                "",
                (
                    "| Split | Samples | Cleaning Dice | Foreground segmentation Dice "
                    "| All-false rate |"
                ),
                "|---|---:|---:|---:|---:|",
            ]
        )
        for split in summary["splits"]:
            metrics = model["splits"][split]
            if metrics["sample_count"]:
                foreground_dice = metrics.get("segmentation_foreground_macro_dice")
                all_false_rate = metrics.get("all_false_rate")
                lines.append(
                    "| "
                    f"{split} | {metrics['sample_count']} | {metrics['cleaning_dice']:.4f} | "
                    f"{foreground_dice:.4f} | {all_false_rate:.4f} |"
                    if foreground_dice is not None and all_false_rate is not None
                    else (
                        f"| {split} | {metrics['sample_count']} | "
                        f"{metrics['cleaning_dice']:.4f} | — | — |"
                    )
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def _csv(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
