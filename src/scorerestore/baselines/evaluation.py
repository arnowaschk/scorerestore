"""Materialized-dataset evaluation for the classical cleaning baseline."""

from __future__ import annotations

import csv
import io
import json
import platform
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image
from PIL import __version__ as pillow_version

from scorerestore import __version__
from scorerestore.dataset.config import SPLIT_NAMES
from scorerestore.dataset.loader import MaterializedDataset
from scorerestore.dataset.manifest import sha256_file, validate_dataset_manifest
from scorerestore.storage import FilesystemStorage

from .classical import normalize_illumination, threshold_variant
from .config import BaselineConfig, BaselineVariantConfig
from .metrics import CleaningMetrics, cleaning_metrics, metrics_from_counts

BASELINE_EVALUATION_SCHEMA_VERSION = 2


class BaselineEvaluationError(RuntimeError):
    """Raised when a baseline evaluation cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class BaselineEvaluationResult:
    """Paths and measured split summaries from one completed evaluation."""

    output_directory: Path
    summary_path: Path
    metrics_jsonl_path: Path
    metrics_csv_path: Path
    sample_count: int
    result_count: int
    split_counts: dict[str, int]
    variant_names: tuple[str, ...]


def evaluate_baseline(
    manifest_path: str | Path,
    output_directory: str | Path,
    *,
    config: BaselineConfig,
    splits: tuple[str, ...] | None = None,
) -> BaselineEvaluationResult:
    """Process selected manifest samples, save cleaned PNGs, and measure cleaning metrics."""

    manifest = Path(manifest_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise BaselineEvaluationError(f"output directory already exists: {output}")
    selected_splits = _selected_splits(splits)
    report = validate_dataset_manifest(manifest)
    records = tuple(record for record in report.records if record["split"] in selected_splits)
    if not records:
        names = ", ".join(selected_splits)
        raise BaselineEvaluationError(f"manifest has no samples in selected split(s): {names}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    temporary.chmod(0o755)
    storage = FilesystemStorage(temporary)
    dataset = MaterializedDataset(manifest, verify=False)
    rows: list[dict[str, Any]] = []
    variant_metrics: dict[str, dict[str, list[CleaningMetrics]]] = {
        variant.name: {split: [] for split in selected_splits} for variant in config.variants
    }
    try:
        for sample in dataset.iter_cleaning():
            record = sample.record
            if record["split"] not in selected_splits:
                continue
            sample_id = record["sample_id"]
            normalized = normalize_illumination(sample.image, config)
            for variant in config.variants:
                baseline = threshold_variant(normalized, config, variant)
                metrics = cleaning_metrics(
                    baseline.image,
                    sample.clean,
                    target_ink_threshold=config.target_ink_threshold,
                )
                result_path = f"results/{variant.name}/{record['split']}/{sample_id}.png"
                storage.write_bytes(result_path, _png_bytes(baseline.image))
                row = {
                    "variant": variant.name,
                    "sample_id": sample_id,
                    "dataset_id": record["dataset_id"],
                    "source_id": record["source_id"],
                    "split": record["split"],
                    "degradation_preset": record["degradation_preset"],
                    "threshold_method": variant.threshold_method,
                    "morphology_applied": variant.apply_morphology,
                    "morphology_operation": (
                        config.morphology.operation if variant.apply_morphology else "none"
                    ),
                    "threshold_value": baseline.threshold_value,
                    "result_path": result_path,
                    **metrics.to_dict(),
                }
                rows.append(row)
                variant_metrics[variant.name][record["split"]].append(metrics)

        variant_summaries = {
            variant.name: _variant_summary(
                variant_metrics[variant.name], variant, config.morphology.operation
            )
            for variant in config.variants
        }
        summary: dict[str, Any] = {
            "schema_version": BASELINE_EVALUATION_SCHEMA_VERSION,
            "baseline": "opencv_classical_cleaning",
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "dataset_id": records[0]["dataset_id"],
            "manifest_path": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "config": config.to_dict(),
            "sample_count": len(records),
            "result_count": len(rows),
            "variants": variant_summaries,
            "metric_notes": {
                "foreground": "thresholded pristine ink; black baseline pixels are foreground",
                "aggregation": "confusion metrics are micro-averaged; SSIM is a sample mean",
                "ssim": "scale-adaptive mean local SSIM with an 11px Gaussian window",
                "challenge": "challenge is reported separately and never merged into test",
                "pixel_accuracy": "intentionally omitted because page background dominates",
            },
        }
        storage.write_text("config.yaml", yaml.safe_dump(config.to_dict(), sort_keys=False))
        storage.write_text("environment.json", _json_text(_environment()))
        storage.write_text("metrics.jsonl", _jsonl_text(rows))
        storage.write_text("metrics.csv", _csv_text(rows))
        storage.write_text("summary.json", _json_text(summary))
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    split_counts = {
        split: sum(1 for record in records if record["split"] == split) for split in selected_splits
    }
    return BaselineEvaluationResult(
        output_directory=output,
        summary_path=output / "summary.json",
        metrics_jsonl_path=output / "metrics.jsonl",
        metrics_csv_path=output / "metrics.csv",
        sample_count=len(records),
        result_count=len(rows),
        split_counts=split_counts,
        variant_names=tuple(variant.name for variant in config.variants),
    )


def _selected_splits(splits: tuple[str, ...] | None) -> tuple[str, ...]:
    if splits is None:
        return SPLIT_NAMES
    unknown = set(splits) - set(SPLIT_NAMES)
    if unknown:
        raise BaselineEvaluationError(f"unknown dataset split(s): {', '.join(sorted(unknown))}")
    return tuple(split for split in SPLIT_NAMES if split in splits)


def _summarize(metrics: list[CleaningMetrics]) -> dict[str, int | float]:
    aggregate = metrics_from_counts(
        true_positive=sum(metric.true_positive for metric in metrics),
        false_positive=sum(metric.false_positive for metric in metrics),
        false_negative=sum(metric.false_negative for metric in metrics),
        true_negative=sum(metric.true_negative for metric in metrics),
        mean_ssim=sum(metric.ssim for metric in metrics) / len(metrics),
    )
    return {"sample_count": len(metrics), **aggregate.to_dict()}


def _variant_summary(
    split_metrics: dict[str, list[CleaningMetrics]],
    variant: BaselineVariantConfig,
    morphology_operation: str,
) -> dict[str, Any]:
    populated = {split: metrics for split, metrics in split_metrics.items() if metrics}
    non_challenge = [
        metric for split, metrics in populated.items() if split != "challenge" for metric in metrics
    ]
    summary: dict[str, Any] = {
        "threshold_method": variant.threshold_method,
        "morphology_applied": variant.apply_morphology,
        "morphology_operation": morphology_operation if variant.apply_morphology else "none",
        "splits": {split: _summarize(metrics) for split, metrics in populated.items()},
    }
    if non_challenge:
        summary["non_challenge"] = _summarize(non_challenge)
    return summary


def _environment() -> dict[str, object]:
    return {
        "scorerestore_version": __version__,
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "pillow_version": pillow_version,
    }


def _png_bytes(image: Image.Image) -> bytes:
    encoded = io.BytesIO()
    image.save(encoded, format="PNG", compress_level=9)
    return encoded.getvalue()


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)


def _csv_text(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
