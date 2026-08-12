from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scorerestore.baselines.config import BaselineConfigError, load_baseline_config

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/baseline.yaml"


def test_default_baseline_config_is_small_and_untuned() -> None:
    config = load_baseline_config(CONFIG_PATH)

    assert config.schema_version == 2
    assert config.illumination.downsample_factor == 8
    assert [variant.name for variant in config.variants] == [
        "otsu",
        "adaptive",
        "otsu_morphology",
        "adaptive_morphology",
    ]
    assert [variant.apply_morphology for variant in config.variants] == [
        False,
        False,
        True,
        True,
    ]
    assert config.morphology.operation == "open_close"
    assert config.morphology.kernel_size <= 5


def test_dotted_overrides_are_strictly_revalidated() -> None:
    config = load_baseline_config(
        CONFIG_PATH,
        overrides=("threshold.adaptive_c=9.0", "morphology.operation=open"),
    )

    assert config.threshold.adaptive_c == 9.0
    assert config.morphology.operation == "open"


def test_fixed_variant_suite_cannot_be_redefined(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["variants"][0]["name"] = "custom"
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(BaselineConfigError, match="fixed V1 suite"):
        load_baseline_config(config_path)


@pytest.mark.parametrize(
    "override, message",
    [
        ("schema_version=1", "schema_version must be 2"),
        ("illumination.downsample_factor=0", "in \\[1, 16\\]"),
        ("threshold.adaptive_block_size=4", "odd integer"),
        ("morphology.operation=none", "must be open, close, or open_close"),
        ("morphology.kernel_size=7", "in \\[1, 5\\]"),
        ("morphology.iterations=3", "in \\[1, 2\\]"),
        ("new_field=true", "unknown baseline config fields"),
    ],
)
def test_invalid_baseline_overrides_are_rejected(override: str, message: str) -> None:
    with pytest.raises(BaselineConfigError, match=message):
        load_baseline_config(CONFIG_PATH, overrides=(override,))
