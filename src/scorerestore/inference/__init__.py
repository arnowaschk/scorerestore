"""Tiled V1 neural inference and isolated raster input adapters."""

from .config import InferenceConfig, InferenceConfigError, load_inference_config
from .io import InputPage, InputReadError, read_input_pages
from .output import (
    cleaned_pdf_path,
    planned_output_paths,
    remove_page_outputs,
    write_cleaned_pdf,
    write_page_outputs,
    write_run_metadata,
)
from .real_world import (
    CheckpointSelection,
    RealWorldComparisonError,
    RealWorldComparisonResult,
    compare_real_world,
)
from .real_world_config import (
    ComparisonModel,
    RealWorldComparisonConfig,
    RealWorldComparisonConfigError,
    load_real_world_comparison_config,
)
from .tiled import CleanResult, clean, load_checkpoint_model

__all__ = [
    "CheckpointSelection",
    "CleanResult",
    "ComparisonModel",
    "InferenceConfig",
    "InferenceConfigError",
    "InputPage",
    "InputReadError",
    "RealWorldComparisonConfig",
    "RealWorldComparisonConfigError",
    "RealWorldComparisonError",
    "RealWorldComparisonResult",
    "clean",
    "cleaned_pdf_path",
    "compare_real_world",
    "load_checkpoint_model",
    "load_inference_config",
    "load_real_world_comparison_config",
    "planned_output_paths",
    "read_input_pages",
    "remove_page_outputs",
    "write_cleaned_pdf",
    "write_page_outputs",
    "write_run_metadata",
]
