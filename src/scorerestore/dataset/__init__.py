"""Materialized V1 dataset generation, validation, loading, and reproduction."""

from scorerestore.dataset.config import DatasetGenerationConfig, load_dataset_config
from scorerestore.dataset.generation import (
    DatasetGenerationResult,
    ReproductionResult,
    generate_dataset,
    reproduce_sample,
)
from scorerestore.dataset.loader import MaterializedDataset, MaterializedSample
from scorerestore.dataset.manifest import DatasetManifestError, validate_dataset_manifest
from scorerestore.dataset.sources import CuratedLilyPondDatasetSource, assign_source_splits

__all__ = [
    "CuratedLilyPondDatasetSource",
    "DatasetGenerationConfig",
    "DatasetGenerationResult",
    "DatasetManifestError",
    "MaterializedDataset",
    "MaterializedSample",
    "ReproductionResult",
    "assign_source_splits",
    "generate_dataset",
    "load_dataset_config",
    "reproduce_sample",
    "validate_dataset_manifest",
]
