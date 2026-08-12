from __future__ import annotations

from pathlib import Path

import pytest

from scorerestore.dataset.config import DatasetConfigError, load_dataset_config
from scorerestore.dataset.sources import CuratedLilyPondDatasetSource, assign_source_splits

PROJECT_ROOT = Path(__file__).parents[2]


def test_smoke_config_is_small_and_source_limited() -> None:
    config = load_dataset_config(PROJECT_ROOT / "configs/dataset/smoke.yaml")

    assert config.target_samples == 2
    assert config.source_ids == ("bach-bwv773-invention-02",)
    assert config.dpi == 48


def test_demo_config_defines_approximately_one_thousand_and_all_layout_axes() -> None:
    config = load_dataset_config(PROJECT_ROOT / "configs/dataset/demo.yaml")

    assert config.target_samples == 1000
    assert config.layout.staff_sizes == (12.0, 16.0, 20.0, 24.0)
    assert set(config.layout.paper_formats) == {"a4", "letter"}
    assert set(config.layout.orientations) == {"portrait", "landscape"}
    assert set(config.split_weights) == {"train", "validation", "test", "challenge"}

    assets = CuratedLilyPondDatasetSource(config.source_manifest).assets()
    assignments = assign_source_splits(
        [asset.id for asset in assets], weights=config.split_weights, seed=config.seed
    )
    assert set(assignments.values()) == {"train", "validation", "test"}


def test_challenge_config_uses_only_held_out_challenge_split() -> None:
    config = load_dataset_config(PROJECT_ROOT / "configs/dataset/challenge.yaml")

    assert config.split_weights == {
        "train": 0.0,
        "validation": 0.0,
        "test": 0.0,
        "challenge": 1.0,
    }
    assert config.challenge_degradation_config == "random"


def test_dataset_config_requires_every_split(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """dataset_id: invalid-config
seed: 1
target_samples: 1
source_manifest: manifest.yaml
dpi: 48
mask_threshold: 0.5
strict_unknown_grobs: true
layout:
  staff_sizes: [16]
  paper_formats: [a4]
  orientations: [portrait]
  margin_range_mm: [8, 12]
  variants_per_combination: 1
splits:
  train: 1
degradation_configs: [light]
challenge_degradation_config: random
""",
        encoding="utf-8",
    )

    with pytest.raises(DatasetConfigError, match="splits is missing"):
        load_dataset_config(config_path)
