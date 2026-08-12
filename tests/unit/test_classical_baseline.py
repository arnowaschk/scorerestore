from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scorerestore.baselines import (
    clean_classical,
    clean_classical_variant,
    load_baseline_config,
)

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/baseline.yaml"


def _uneven_score_image() -> Image.Image:
    width, height = 161, 91
    illumination = np.linspace(135, 245, width, dtype=np.float32)
    page = np.repeat(illumination[None, :], height, axis=0)
    page[25:27, 15:145] = 25
    page[45:47, 15:145] = 25
    page[65:67, 15:145] = 25
    return Image.fromarray(page.astype(np.uint8))


def test_otsu_baseline_normalizes_and_returns_binary_without_geometry_change() -> None:
    source = _uneven_score_image()
    result = clean_classical(source, load_baseline_config(CONFIG_PATH))

    assert result.image.size == source.size
    assert result.normalized.size == source.size
    assert set(np.unique(np.asarray(result.image))) <= {0, 255}
    assert result.threshold_value is not None
    assert np.count_nonzero(np.asarray(result.image) == 0) > 0


def test_all_four_variants_run_and_return_binary_images() -> None:
    source = _uneven_score_image()
    config = load_baseline_config(CONFIG_PATH)
    for variant in config.variants:
        result = clean_classical_variant(source, config, variant)

        assert result.image.size == source.size
        assert set(np.unique(np.asarray(result.image))) <= {0, 255}
        assert (result.threshold_value is None) == (variant.threshold_method == "adaptive")


def test_each_supported_light_morphology_operation_runs() -> None:
    source = _uneven_score_image()
    for operation in ("open", "close", "open_close"):
        config = load_baseline_config(
            CONFIG_PATH,
            overrides=(f"morphology.operation={operation}",),
        )
        for variant in config.variants:
            if variant.apply_morphology:
                result = clean_classical_variant(source, config, variant)
                assert result.image.size == source.size
