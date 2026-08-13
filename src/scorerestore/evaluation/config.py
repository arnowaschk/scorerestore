"""Strict YAML configuration for measured V1 evaluation and benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scorerestore.config import ConfigError, apply_overrides, load_config
from scorerestore.dataset.config import SPLIT_NAMES


class EvaluationConfigError(ValueError):
    """Raised when an evaluation would be incomparable or ambiguous."""


@dataclass(frozen=True, slots=True)
class EvaluationModel:
    name: str
    checkpoint: Path


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    dataset_manifest: Path
    models: tuple[EvaluationModel, ...]
    splits: tuple[str, ...]
    device: str
    tile_size: int
    overlap: int
    pdf_dpi: int
    cleaning_threshold: float
    segmentation_threshold: float
    baseline_config: Path
    baseline_variant: str
    report_seed: int
    report_samples: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["dataset_manifest"] = str(self.dataset_manifest)
        result["models"] = [
            {"name": model.name, "checkpoint": str(model.checkpoint)} for model in self.models
        ]
        result["splits"] = list(self.splits)
        result["baseline_config"] = str(self.baseline_config)
        return result


def load_evaluation_config(
    path: str | Path, *, overrides: tuple[str, ...] = ()
) -> EvaluationConfig:
    try:
        raw = apply_overrides(load_config(path), overrides)
    except ConfigError as error:
        raise EvaluationConfigError(str(error)) from error
    allowed = {"dataset_manifest", "models", "splits", "inference", "baseline", "report"}
    _unknown(raw, allowed, "evaluation config")
    inference = _mapping(raw, "inference")
    _unknown(
        inference,
        {
            "device",
            "tile_size",
            "overlap",
            "pdf_dpi",
            "cleaning_threshold",
            "segmentation_threshold",
        },
        "inference",
    )
    baseline = _mapping(raw, "baseline")
    _unknown(baseline, {"config", "variant"}, "baseline")
    report = _mapping(raw, "report")
    _unknown(report, {"seed", "samples"}, "report")
    tile_size, overlap = _positive(inference, "tile_size"), _nonnegative(inference, "overlap")
    if tile_size % 16 or overlap >= tile_size:
        raise EvaluationConfigError(
            "inference tile_size must be divisible by 16 and exceed overlap"
        )
    return EvaluationConfig(
        dataset_manifest=Path(_text(raw, "dataset_manifest")),
        models=_models(raw.get("models")),
        splits=_splits(raw.get("splits")),
        device=_choice(inference, "device", {"auto", "cuda", "cpu"}),
        tile_size=tile_size,
        overlap=overlap,
        pdf_dpi=_positive(inference, "pdf_dpi"),
        cleaning_threshold=_fraction(inference, "cleaning_threshold"),
        segmentation_threshold=_fraction(inference, "segmentation_threshold"),
        baseline_config=Path(_text(baseline, "config")),
        baseline_variant=_text(baseline, "variant"),
        report_seed=_integer(report, "seed"),
        report_samples=_positive(report, "samples"),
    )


def _models(raw: Any) -> tuple[EvaluationModel, ...]:
    if not isinstance(raw, list) or not raw:
        raise EvaluationConfigError("models must be a nonempty list")
    models = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise EvaluationConfigError(f"models[{index}] must be a mapping")
        _unknown(item, {"name", "checkpoint"}, f"models[{index}]")
        models.append(EvaluationModel(_text(item, "name"), Path(_text(item, "checkpoint"))))
    if len({model.name for model in models}) != len(models):
        raise EvaluationConfigError("model names must be unique")
    return tuple(models)


def _splits(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw or any(item not in SPLIT_NAMES for item in raw):
        raise EvaluationConfigError(
            f"splits must be a nonempty subset of: {', '.join(SPLIT_NAMES)}"
        )
    if len(set(raw)) != len(raw):
        raise EvaluationConfigError("splits must not contain duplicates")
    return tuple(item for item in SPLIT_NAMES if item in raw)


def _mapping(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise EvaluationConfigError(f"{name} must be a mapping")
    return value


def _unknown(raw: dict[str, Any], allowed: set[str], context: str) -> None:
    extra = set(raw) - allowed
    if extra:
        raise EvaluationConfigError(f"unknown {context} fields: {', '.join(sorted(extra))}")


def _text(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise EvaluationConfigError(f"{name} must be a nonempty string")
    return value


def _integer(raw: dict[str, Any], name: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationConfigError(f"{name} must be an integer")
    return value


def _positive(raw: dict[str, Any], name: str) -> int:
    value = _integer(raw, name)
    if value < 1:
        raise EvaluationConfigError(f"{name} must be positive")
    return value


def _nonnegative(raw: dict[str, Any], name: str) -> int:
    value = _integer(raw, name)
    if value < 0:
        raise EvaluationConfigError(f"{name} must be nonnegative")
    return value


def _fraction(raw: dict[str, Any], name: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise EvaluationConfigError(f"{name} must be within [0, 1]")
    return float(value)


def _choice(raw: dict[str, Any], name: str, choices: set[str]) -> str:
    value = _text(raw, name)
    if value not in choices:
        raise EvaluationConfigError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return value
