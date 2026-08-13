"""Strict YAML configuration for Milestone 8 inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scorerestore.config import ConfigError, apply_overrides, load_config


class InferenceConfigError(ValueError):
    """Raised when an inference configuration is invalid or ambiguous."""


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    checkpoint: Path
    device: str
    tile_size: int
    overlap: int
    cleaning_threshold: float
    segmentation_threshold: float
    pdf_dpi: int
    overlay: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["checkpoint"] = str(self.checkpoint)
        return result


def load_inference_config(path: str | Path, *, overrides: tuple[str, ...] = ()) -> InferenceConfig:
    try:
        raw = apply_overrides(load_config(path), overrides)
    except ConfigError as error:
        raise InferenceConfigError(str(error)) from error
    allowed = {
        "checkpoint",
        "device",
        "tile_size",
        "overlap",
        "cleaning_threshold",
        "segmentation_threshold",
        "pdf_dpi",
        "overlay",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise InferenceConfigError(f"unknown inference config fields: {', '.join(sorted(unknown))}")
    checkpoint = raw.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise InferenceConfigError("checkpoint must be a nonempty string")
    device = raw.get("device")
    if device not in {"auto", "cuda", "cpu"}:
        raise InferenceConfigError("device must be auto, cuda, or cpu")
    tile_size = _positive_integer(raw, "tile_size")
    overlap = _nonnegative_integer(raw, "overlap")
    if tile_size % 16:
        raise InferenceConfigError("tile_size must be divisible by 16 for V1 models")
    if overlap >= tile_size:
        raise InferenceConfigError("overlap must be smaller than tile_size")
    return InferenceConfig(
        checkpoint=Path(checkpoint),
        device=device,
        tile_size=tile_size,
        overlap=overlap,
        cleaning_threshold=_fraction(raw, "cleaning_threshold"),
        segmentation_threshold=_fraction(raw, "segmentation_threshold"),
        pdf_dpi=_positive_integer(raw, "pdf_dpi"),
        overlay=_boolean(raw, "overlay"),
    )


def _positive_integer(raw: dict[str, Any], name: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InferenceConfigError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(raw: dict[str, Any], name: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InferenceConfigError(f"{name} must be a nonnegative integer")
    return value


def _fraction(raw: dict[str, Any], name: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise InferenceConfigError(f"{name} must be within [0, 1]")
    return float(value)


def _boolean(raw: dict[str, Any], name: str) -> bool:
    value = raw.get(name)
    if not isinstance(value, bool):
        raise InferenceConfigError(f"{name} must be a boolean")
    return value
