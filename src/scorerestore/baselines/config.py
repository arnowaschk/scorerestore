"""Strict YAML configuration for the four-variant V1 classical baseline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scorerestore.config import ConfigError, apply_overrides, load_config

ThresholdMethod = Literal["otsu", "adaptive"]
MorphologyOperation = Literal["open", "close", "open_close"]


class BaselineConfigError(ValueError):
    """Raised when classical-baseline configuration is invalid."""


@dataclass(frozen=True, slots=True)
class IlluminationConfig:
    """Smooth-background estimation settings shared by every variant."""

    gaussian_sigma: float
    downsample_factor: int


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    """Adaptive-threshold parameters; Otsu has no tuneable threshold."""

    adaptive_block_size: int
    adaptive_c: float


@dataclass(frozen=True, slots=True)
class MorphologyConfig:
    """One light operation shared by both morphology-enabled variants."""

    operation: MorphologyOperation
    kernel_size: int
    iterations: int


@dataclass(frozen=True, slots=True)
class BaselineVariantConfig:
    """One fixed member of the four-variant comparison suite."""

    name: str
    threshold_method: ThresholdMethod
    apply_morphology: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "threshold_method": self.threshold_method,
            "apply_morphology": self.apply_morphology,
        }


BASELINE_VARIANTS = (
    BaselineVariantConfig("otsu", "otsu", False),
    BaselineVariantConfig("adaptive", "adaptive", False),
    BaselineVariantConfig("otsu_morphology", "otsu", True),
    BaselineVariantConfig("adaptive_morphology", "adaptive", True),
)


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Shared settings for the fixed four-variant baseline suite."""

    schema_version: int
    illumination: IlluminationConfig
    threshold: ThresholdConfig
    morphology: MorphologyConfig
    target_ink_threshold: float
    variants: tuple[BaselineVariantConfig, ...] = BASELINE_VARIANTS

    def to_dict(self) -> dict[str, object]:
        """Return a YAML/JSON-serializable resolved configuration."""

        return {
            "schema_version": self.schema_version,
            "illumination": {
                "gaussian_sigma": self.illumination.gaussian_sigma,
                "downsample_factor": self.illumination.downsample_factor,
            },
            "threshold": {
                "adaptive_block_size": self.threshold.adaptive_block_size,
                "adaptive_c": self.threshold.adaptive_c,
            },
            "morphology": {
                "operation": self.morphology.operation,
                "kernel_size": self.morphology.kernel_size,
                "iterations": self.morphology.iterations,
            },
            "target_ink_threshold": self.target_ink_threshold,
            "variants": [variant.to_dict() for variant in self.variants],
        }


def load_baseline_config(
    path: str | Path,
    *,
    overrides: Sequence[str] = (),
) -> BaselineConfig:
    """Load and strictly validate a four-variant baseline YAML file."""

    try:
        raw = apply_overrides(load_config(path), tuple(overrides))
    except ConfigError as error:
        raise BaselineConfigError(str(error)) from error
    _reject_unknown(
        raw,
        {
            "schema_version",
            "illumination",
            "threshold",
            "morphology",
            "target_ink_threshold",
            "variants",
        },
        "baseline config",
    )
    if raw.get("schema_version") != 2:
        raise BaselineConfigError("schema_version must be 2")
    illumination = _mapping(raw.get("illumination"), "illumination")
    _reject_unknown(illumination, {"gaussian_sigma", "downsample_factor"}, "illumination")
    sigma = _number(illumination.get("gaussian_sigma"), "illumination.gaussian_sigma")
    if sigma <= 0:
        raise BaselineConfigError("illumination.gaussian_sigma must be positive")
    downsample_factor = _integer(
        illumination.get("downsample_factor"), "illumination.downsample_factor"
    )
    if not 1 <= downsample_factor <= 16:
        raise BaselineConfigError("illumination.downsample_factor must be in [1, 16]")

    threshold = _mapping(raw.get("threshold"), "threshold")
    _reject_unknown(threshold, {"adaptive_block_size", "adaptive_c"}, "threshold")
    block_size = _integer(threshold.get("adaptive_block_size"), "threshold.adaptive_block_size")
    if block_size < 3 or block_size % 2 == 0:
        raise BaselineConfigError("threshold.adaptive_block_size must be an odd integer >= 3")
    adaptive_c = _number(threshold.get("adaptive_c"), "threshold.adaptive_c")

    morphology = _mapping(raw.get("morphology"), "morphology")
    _reject_unknown(morphology, {"operation", "kernel_size", "iterations"}, "morphology")
    operation = morphology.get("operation")
    if operation not in {"open", "close", "open_close"}:
        raise BaselineConfigError("morphology.operation must be open, close, or open_close")
    kernel_size = _integer(morphology.get("kernel_size"), "morphology.kernel_size")
    if kernel_size < 1 or kernel_size > 5 or kernel_size % 2 == 0:
        raise BaselineConfigError("morphology.kernel_size must be an odd integer in [1, 5]")
    iterations = _integer(morphology.get("iterations"), "morphology.iterations")
    if not 1 <= iterations <= 2:
        raise BaselineConfigError("morphology.iterations must be in [1, 2]")

    target_threshold = _number(raw.get("target_ink_threshold"), "target_ink_threshold")
    if not 0.0 < target_threshold <= 1.0:
        raise BaselineConfigError("target_ink_threshold must be in (0, 1]")
    variants = _variants(raw.get("variants"))
    return BaselineConfig(
        schema_version=2,
        illumination=IlluminationConfig(sigma, downsample_factor),
        threshold=ThresholdConfig(block_size, adaptive_c),
        morphology=MorphologyConfig(operation, kernel_size, iterations),
        target_ink_threshold=target_threshold,
        variants=variants,
    )


def _variants(raw: Any) -> tuple[BaselineVariantConfig, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise BaselineConfigError("variants must be a list of mappings")
    parsed: list[BaselineVariantConfig] = []
    for index, item in enumerate(raw):
        _reject_unknown(
            item,
            {"name", "threshold_method", "apply_morphology"},
            f"variants[{index}]",
        )
        name = item.get("name")
        method = item.get("threshold_method")
        apply_morphology = item.get("apply_morphology")
        if not isinstance(name, str) or not name:
            raise BaselineConfigError(f"variants[{index}].name must be a nonempty string")
        if method not in {"otsu", "adaptive"}:
            raise BaselineConfigError(
                f"variants[{index}].threshold_method must be otsu or adaptive"
            )
        if not isinstance(apply_morphology, bool):
            raise BaselineConfigError(f"variants[{index}].apply_morphology must be a boolean")
        parsed.append(BaselineVariantConfig(name, method, apply_morphology))
    variants = tuple(parsed)
    if variants != BASELINE_VARIANTS:
        names = ", ".join(variant.name for variant in BASELINE_VARIANTS)
        raise BaselineConfigError(f"variants must define the fixed V1 suite in order: {names}")
    return variants


def _mapping(raw: Any, field: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise BaselineConfigError(f"{field} must be a mapping")
    return raw


def _number(raw: Any, field: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise BaselineConfigError(f"{field} must be a number")
    return float(raw)


def _integer(raw: Any, field: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise BaselineConfigError(f"{field} must be an integer")
    return raw


def _reject_unknown(raw: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise BaselineConfigError(f"unknown {context} fields: {', '.join(sorted(unknown))}")
