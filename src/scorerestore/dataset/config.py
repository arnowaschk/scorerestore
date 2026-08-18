"""Strict YAML configuration for materialized V1 dataset generation."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scorerestore.config import ConfigError, apply_overrides, load_config

SPLIT_NAMES = ("train", "validation", "test", "challenge")
_DATASET_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")


class DatasetConfigError(ValueError):
    """Raised when a dataset-generation configuration is invalid."""


@dataclass(frozen=True, slots=True)
class LayoutGridConfig:
    """Deterministic Cartesian layout grid with seeded margin variation."""

    staff_sizes: tuple[float, ...]
    paper_formats: tuple[str, ...]
    orientations: tuple[str, ...]
    margin_range_mm: tuple[float, float]
    variants_per_combination: int


@dataclass(frozen=True, slots=True)
class DatasetGenerationConfig:
    """Resolved materialized-dataset configuration."""

    dataset_id: str
    seed: int
    target_samples: int
    source_manifest: Path
    source_ids: tuple[str, ...] | None
    dpi: int
    mask_threshold: float
    strict_unknown_grobs: bool
    layout: LayoutGridConfig
    split_weights: dict[str, float]
    degradation_configs: tuple[str, ...]
    challenge_degradation_config: str
    workers: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable resolved configuration."""

        return {
            "dataset_id": self.dataset_id,
            "seed": self.seed,
            "target_samples": self.target_samples,
            "source_manifest": str(self.source_manifest),
            "source_ids": list(self.source_ids) if self.source_ids is not None else None,
            "dpi": self.dpi,
            "mask_threshold": self.mask_threshold,
            "strict_unknown_grobs": self.strict_unknown_grobs,
            "layout": {
                "staff_sizes": list(self.layout.staff_sizes),
                "paper_formats": list(self.layout.paper_formats),
                "orientations": list(self.layout.orientations),
                "margin_range_mm": list(self.layout.margin_range_mm),
                "variants_per_combination": self.layout.variants_per_combination,
            },
            "splits": dict(self.split_weights),
            "degradation_configs": list(self.degradation_configs),
            "challenge_degradation_config": self.challenge_degradation_config,
            "workers": self.workers,
        }


def load_dataset_config(
    path: str | Path, *, overrides: tuple[str, ...] = ()
) -> DatasetGenerationConfig:
    """Load and strictly validate one dataset-generation YAML file."""

    config_path = Path(path)
    try:
        raw = apply_overrides(load_config(config_path), overrides)
    except ConfigError as error:
        raise DatasetConfigError(str(error)) from error
    allowed = {
        "dataset_id",
        "seed",
        "target_samples",
        "source_manifest",
        "source_ids",
        "dpi",
        "mask_threshold",
        "strict_unknown_grobs",
        "layout",
        "splits",
        "degradation_configs",
        "challenge_degradation_config",
        "workers",
    }
    _reject_unknown(raw, allowed, "dataset config")

    dataset_id = _text(raw, "dataset_id")
    if _DATASET_ID_PATTERN.fullmatch(dataset_id) is None:
        raise DatasetConfigError(
            "dataset_id must be 3-64 lowercase letters, digits, dots, underscores, or hyphens"
        )
    seed = _integer(raw, "seed")
    target_samples = _positive_integer(raw, "target_samples")
    source_manifest = Path(_text(raw, "source_manifest"))
    source_ids = _optional_text_list(raw.get("source_ids"), "source_ids")
    dpi = _positive_integer(raw, "dpi")
    mask_threshold = _number(raw, "mask_threshold")
    if not 0.0 < mask_threshold <= 1.0:
        raise DatasetConfigError("mask_threshold must be greater than 0.0 and at most 1.0")
    strict_unknown_grobs = raw.get("strict_unknown_grobs")
    if not isinstance(strict_unknown_grobs, bool):
        raise DatasetConfigError("strict_unknown_grobs must be a boolean")

    layout = _layout_config(raw.get("layout"))
    split_weights = _split_weights(raw.get("splits"))
    degradation_configs = _text_list(raw.get("degradation_configs"), "degradation_configs")
    challenge_config = _text(raw, "challenge_degradation_config")
    workers = _workers(raw.get("workers", "auto"))
    return DatasetGenerationConfig(
        dataset_id=dataset_id,
        seed=seed,
        target_samples=target_samples,
        source_manifest=source_manifest,
        source_ids=source_ids,
        dpi=dpi,
        mask_threshold=mask_threshold,
        strict_unknown_grobs=strict_unknown_grobs,
        layout=layout,
        split_weights=split_weights,
        degradation_configs=degradation_configs,
        challenge_degradation_config=challenge_config,
        workers=workers,
    )


def _layout_config(raw: Any) -> LayoutGridConfig:
    if not isinstance(raw, Mapping):
        raise DatasetConfigError("layout must be a mapping")
    allowed = {
        "staff_sizes",
        "paper_formats",
        "orientations",
        "margin_range_mm",
        "variants_per_combination",
    }
    _reject_unknown(raw, allowed, "layout")
    staff_sizes = tuple(_number_list(raw.get("staff_sizes"), "layout.staff_sizes"))
    if any(not 8.0 <= value <= 60.0 for value in staff_sizes):
        raise DatasetConfigError("layout.staff_sizes must stay within [8, 60]")
    paper_formats = _text_list(raw.get("paper_formats"), "layout.paper_formats")
    if not set(paper_formats) <= {"a4", "letter"}:
        raise DatasetConfigError("layout.paper_formats may contain only a4 and letter")
    orientations = _text_list(raw.get("orientations"), "layout.orientations")
    if not set(orientations) <= {"portrait", "landscape"}:
        raise DatasetConfigError("layout.orientations may contain only portrait and landscape")
    margin_range = _number_range(raw.get("margin_range_mm"), "layout.margin_range_mm")
    if margin_range[0] < 3.0 or margin_range[1] > 30.0:
        raise DatasetConfigError("layout.margin_range_mm must stay within [3, 30]")
    variants = raw.get("variants_per_combination")
    if isinstance(variants, bool) or not isinstance(variants, int) or variants < 1:
        raise DatasetConfigError("layout.variants_per_combination must be a positive integer")
    return LayoutGridConfig(
        staff_sizes=staff_sizes,
        paper_formats=paper_formats,
        orientations=orientations,
        margin_range_mm=margin_range,
        variants_per_combination=variants,
    )


def _split_weights(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise DatasetConfigError("splits must be a mapping")
    _reject_unknown(raw, set(SPLIT_NAMES), "splits")
    if set(raw) != set(SPLIT_NAMES):
        missing = set(SPLIT_NAMES) - set(raw)
        raise DatasetConfigError(f"splits is missing: {', '.join(sorted(missing))}")
    weights: dict[str, float] = {}
    for split in SPLIT_NAMES:
        value = raw[split]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise DatasetConfigError(f"splits.{split} must be a nonnegative number")
        weights[split] = float(value)
    if sum(weights.values()) <= 0:
        raise DatasetConfigError("at least one split weight must be positive")
    return weights


def _text(raw: Mapping[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetConfigError(f"{field} must be a nonempty string")
    return value


def _integer(raw: Mapping[str, Any], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetConfigError(f"{field} must be an integer")
    return value


def _positive_integer(raw: Mapping[str, Any], field: str) -> int:
    value = _integer(raw, field)
    if value < 1:
        raise DatasetConfigError(f"{field} must be positive")
    return value


def _workers(raw: Any) -> int:
    """Resolve ``auto`` to every CPU that the current process may schedule on."""

    if raw == "auto":
        affinity = getattr(os, "sched_getaffinity", None)
        try:
            if affinity is not None:
                return max(1, len(affinity(0)))
        except OSError:
            pass
        return max(1, os.cpu_count() or 1)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise DatasetConfigError("workers must be a positive integer or 'auto'")
    return raw


def _number(raw: Mapping[str, Any], field: str) -> float:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetConfigError(f"{field} must be a number")
    return float(value)


def _text_list(raw: Any, field: str) -> tuple[str, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise DatasetConfigError(f"{field} must be a nonempty list")
    if any(not isinstance(value, str) or not value.strip() for value in raw):
        raise DatasetConfigError(f"{field} must contain nonempty strings")
    values = tuple(raw)
    if len(values) != len(set(values)):
        raise DatasetConfigError(f"{field} must not contain duplicates")
    return values


def _optional_text_list(raw: Any, field: str) -> tuple[str, ...] | None:
    if raw is None:
        return None
    return _text_list(raw, field)


def _number_list(raw: Any, field: str) -> tuple[float, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise DatasetConfigError(f"{field} must be a nonempty list")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw):
        raise DatasetConfigError(f"{field} must contain numbers")
    values = tuple(float(value) for value in raw)
    if len(values) != len(set(values)):
        raise DatasetConfigError(f"{field} must not contain duplicates")
    return values


def _number_range(raw: Any, field: str) -> tuple[float, float]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != 2:
        raise DatasetConfigError(f"{field} must be a two-item range")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw):
        raise DatasetConfigError(f"{field} must contain numbers")
    low, high = (float(value) for value in raw)
    if low > high:
        raise DatasetConfigError(f"{field} must be ordered")
    return low, high


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise DatasetConfigError(f"unknown {context} fields: {', '.join(sorted(unknown))}")
