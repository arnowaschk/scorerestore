"""Strict YAML configuration for configurable real-world visual comparisons."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scorerestore.config import ConfigError, apply_overrides, load_config


class RealWorldComparisonConfigError(ValueError):
    """Raised when a real-world comparison configuration is ambiguous or malformed."""


@dataclass(frozen=True, slots=True)
class ComparisonModel:
    """One neural panel, using an explicit checkpoint or deterministic backend discovery."""

    identifier: str
    label: str
    backend: str
    checkpoint: Path | None


@dataclass(frozen=True, slots=True)
class RealWorldComparisonConfig:
    input_root: Path
    runs_root: Path
    device: str
    tile_size: int
    overlap: int
    pdf_dpi: int
    cleaning_threshold: float
    segmentation_threshold: float
    baseline_config: Path
    baseline_variant: str
    models: tuple[ComparisonModel, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["input_root"] = str(self.input_root)
        result["runs_root"] = str(self.runs_root)
        result["baseline_config"] = str(self.baseline_config)
        result["models"] = [
            {
                "id": model.identifier,
                "label": model.label,
                "backend": model.backend,
                "checkpoint": str(model.checkpoint) if model.checkpoint is not None else "auto",
            }
            for model in self.models
        ]
        return result


def load_real_world_comparison_config(
    path: str | Path, *, overrides: tuple[str, ...] = ()
) -> RealWorldComparisonConfig:
    try:
        raw = apply_overrides(load_config(path), overrides)
    except ConfigError as error:
        raise RealWorldComparisonConfigError(str(error)) from error
    _unknown(raw, {"input_root", "runs_root", "inference", "classical", "models"}, "config")
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
    classical = _mapping(raw, "classical")
    _unknown(classical, {"config", "variant"}, "classical")
    tile_size, overlap = _positive(inference, "tile_size"), _nonnegative(inference, "overlap")
    if tile_size % 16 or overlap >= tile_size:
        raise RealWorldComparisonConfigError(
            "inference tile_size must be divisible by 16 and exceed overlap"
        )
    return RealWorldComparisonConfig(
        input_root=Path(_text(raw, "input_root")),
        runs_root=Path(_text(raw, "runs_root")),
        device=_choice(inference, "device", {"auto", "cuda", "cpu"}),
        tile_size=tile_size,
        overlap=overlap,
        pdf_dpi=_positive(inference, "pdf_dpi"),
        cleaning_threshold=_fraction(inference, "cleaning_threshold"),
        segmentation_threshold=_fraction(inference, "segmentation_threshold"),
        baseline_config=Path(_text(classical, "config")),
        baseline_variant=_text(classical, "variant"),
        models=_models(raw.get("models")),
    )


def _models(raw: Any) -> tuple[ComparisonModel, ...]:
    if not isinstance(raw, list) or not raw:
        raise RealWorldComparisonConfigError("models must be a nonempty list")
    result = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RealWorldComparisonConfigError(f"models[{index}] must be a mapping")
        _unknown(item, {"id", "label", "backend", "checkpoint"}, f"models[{index}]")
        identifier = _text(item, "id")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", identifier):
            raise RealWorldComparisonConfigError(
                f"models[{index}].id must use lowercase letters, digits, hyphens, or underscores"
            )
        checkpoint = item.get("checkpoint")
        if checkpoint == "auto":
            checkpoint_path = None
        elif isinstance(checkpoint, str) and checkpoint:
            checkpoint_path = Path(checkpoint)
        else:
            raise RealWorldComparisonConfigError(
                f"models[{index}].checkpoint must be a path or 'auto'"
            )
        result.append(
            ComparisonModel(
                identifier,
                _text(item, "label"),
                _choice(item, "backend", {"unet", "resnet18"}),
                checkpoint_path,
            )
        )
    if len({model.identifier for model in result}) != len(result):
        raise RealWorldComparisonConfigError("models ids must be unique")
    return tuple(result)


def _mapping(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise RealWorldComparisonConfigError(f"{name} must be a mapping")
    return value


def _unknown(raw: dict[str, Any], allowed: set[str], context: str) -> None:
    extra = set(raw) - allowed
    if extra:
        raise RealWorldComparisonConfigError(
            f"unknown {context} fields: {', '.join(sorted(extra))}"
        )


def _text(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise RealWorldComparisonConfigError(f"{name} must be a nonempty string")
    return value


def _choice(raw: dict[str, Any], name: str, choices: set[str]) -> str:
    value = _text(raw, name)
    if value not in choices:
        raise RealWorldComparisonConfigError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return value


def _integer(raw: dict[str, Any], name: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RealWorldComparisonConfigError(f"{name} must be an integer")
    return value


def _positive(raw: dict[str, Any], name: str) -> int:
    value = _integer(raw, name)
    if value < 1:
        raise RealWorldComparisonConfigError(f"{name} must be positive")
    return value


def _nonnegative(raw: dict[str, Any], name: str) -> int:
    value = _integer(raw, name)
    if value < 0:
        raise RealWorldComparisonConfigError(f"{name} must be nonnegative")
    return value


def _fraction(raw: dict[str, Any], name: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise RealWorldComparisonConfigError(f"{name} must be within [0, 1]")
    return float(value)
