"""Understandable non-deep-learning cleaning baseline."""

from scorerestore.baselines.classical import (
    BaselineResult,
    clean_classical,
    clean_classical_variant,
)
from scorerestore.baselines.config import (
    BASELINE_VARIANTS,
    BaselineConfig,
    BaselineConfigError,
    BaselineVariantConfig,
    load_baseline_config,
)
from scorerestore.baselines.evaluation import BaselineEvaluationResult, evaluate_baseline
from scorerestore.baselines.metrics import CleaningMetrics, cleaning_metrics

__all__ = [
    "BASELINE_VARIANTS",
    "BaselineConfig",
    "BaselineConfigError",
    "BaselineEvaluationResult",
    "BaselineResult",
    "BaselineVariantConfig",
    "CleaningMetrics",
    "clean_classical",
    "clean_classical_variant",
    "cleaning_metrics",
    "evaluate_baseline",
    "load_baseline_config",
]
