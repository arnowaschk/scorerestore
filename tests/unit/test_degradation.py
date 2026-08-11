from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scorerestore import DegradationPipeline, DegradationResult, degrade
from scorerestore.degradation import (
    DEGRADATION_FAMILIES,
    PRESET_NAMES,
    DegradationConfig,
    DegradationConfigError,
    preset_config,
    recipe_json,
    resolve_degradation_config,
)

PROJECT_ROOT = Path(__file__).parents[2]


@pytest.fixture
def score_image() -> Image.Image:
    image = Image.new("RGB", (160, 96), "white")
    draw = ImageDraw.Draw(image)
    for y in range(18, 79, 12):
        draw.line((8, y, 151, y), fill="black", width=1)
    draw.ellipse((46, 39, 55, 46), fill="black")
    draw.line((54, 23, 54, 42), fill="black", width=2)
    draw.text((10, 4), "ScoreRestore", fill="black")
    return image


def test_v1_registry_has_exactly_five_required_families() -> None:
    assert DEGRADATION_FAMILIES == (
        "blur",
        "gaussian_noise",
        "uneven_illumination",
        "jpeg",
        "stains",
    )


@pytest.mark.parametrize("family", DEGRADATION_FAMILIES)
def test_each_family_changes_pixels_without_geometry(family: str, score_image: Image.Image) -> None:
    original_bytes = score_image.tobytes()
    config = DegradationConfig.from_mapping(
        {
            "preset": "custom",
            "operation_count": [1, 1],
            family: {"enabled": True},
        }
    )

    result = degrade(score_image, config=config, seed=112233)

    assert isinstance(result, DegradationResult)
    assert result.image.mode == "L"
    assert result.image.size == score_image.size
    assert result.image.tobytes() != score_image.convert("L").tobytes()
    assert score_image.tobytes() == original_bytes
    assert result.metadata["geometry_changed"] is False
    assert [operation["family"] for operation in result.recipe["operations"]] == [family]


@pytest.mark.parametrize(
    ("preset", "minimum", "maximum"),
    [("light", 1, 2), ("medium", 2, 3), ("heavy", 3, 5), ("random", 1, 5)],
)
def test_preset_composition_counts_and_dimensions(
    preset: str,
    minimum: int,
    maximum: int,
    score_image: Image.Image,
) -> None:
    result = degrade(score_image, config=preset, seed=1234)

    operations = result.recipe["operations"]
    assert minimum <= len(operations) <= maximum
    assert len({operation["family"] for operation in operations}) == len(operations)
    assert result.image.size == score_image.size


def test_same_image_config_and_seed_are_byte_identical(score_image: Image.Image) -> None:
    first = degrade(score_image, config="random", seed=987654321)
    second = degrade(score_image, config="random", seed=987654321)

    assert first.image.tobytes() == second.image.tobytes()
    assert first.recipe == second.recipe
    assert first.metadata == second.metadata


def test_reusable_pipeline_matches_convenience_api(score_image: Image.Image) -> None:
    pipeline = DegradationPipeline(preset_config("medium"))

    via_pipeline = pipeline.apply(score_image, seed=2468)
    via_function = degrade(score_image, config="medium", seed=2468)

    assert via_pipeline.image.tobytes() == via_function.image.tobytes()
    assert via_pipeline.recipe == via_function.recipe


def test_different_seeds_change_recipe_and_output(score_image: Image.Image) -> None:
    first = degrade(score_image, config="heavy", seed=1)
    second = degrade(score_image, config="heavy", seed=2)

    assert first.recipe != second.recipe
    assert first.image.tobytes() != second.image.tobytes()


def test_recipe_records_seed_order_parameters_hashes_and_versions(
    score_image: Image.Image,
) -> None:
    result = degrade(score_image, config="medium", seed=42)
    serialized = recipe_json(result)
    decoded = json.loads(serialized)

    assert decoded == result.recipe
    assert decoded["seed"] == 42
    assert decoded["schema_version"] == 1
    assert decoded["source"]["dimensions"] == {"height": 96, "width": 160}
    assert decoded["output"]["dimensions"] == {"height": 96, "width": 160}
    assert len(decoded["source"]["sha256"]) == 64
    assert len(decoded["output"]["sha256"]) == 64
    assert set(decoded["software_versions"]) == {"numpy", "pillow", "python", "scorerestore"}
    for expected_order, operation in enumerate(decoded["operations"], start=1):
        assert operation["order"] == expected_order
        assert isinstance(operation["seed"], int)
        assert operation["parameters"]


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_repository_yaml_presets_are_valid_and_identified(preset: str) -> None:
    config_path = PROJECT_ROOT / f"configs/degradation/{preset}.yaml"

    resolved = resolve_degradation_config(config_path)

    assert resolved.preset == preset
    assert resolved.enabled_families


def test_config_can_override_a_builtin_severity() -> None:
    config = DegradationConfig.from_mapping(
        {
            "preset": "medium",
            "operation_count": [1, 1],
            "blur": {"radius": [0.25, 0.25]},
            "gaussian_noise": {"enabled": False},
            "uneven_illumination": {"enabled": False},
            "jpeg": {"enabled": False},
            "stains": {"enabled": False},
        }
    )

    assert config.enabled_families == ("blur",)
    assert config.blur.radius.minimum == config.blur.radius.maximum == 0.25


def test_config_without_an_enabled_family_is_rejected() -> None:
    with pytest.raises(DegradationConfigError, match="at least one"):
        DegradationConfig.from_mapping(
            {
                "preset": "custom",
                "operation_count": [1, 1],
            }
        )


def test_operation_count_can_repeat_families_with_new_parameters(
    score_image: Image.Image,
) -> None:
    config = DegradationConfig.from_mapping(
        {
            "preset": "custom",
            "operation_count": [8, 12],
            "blur": {"enabled": True},
        }
    )

    result = degrade(score_image, config=config, seed=1234)
    operations = result.recipe["operations"]

    assert 8 <= len(operations) <= 12
    assert {operation["family"] for operation in operations} == {"blur"}
    assert len({operation["seed"] for operation in operations}) == len(operations)
    assert len({operation["parameters"]["radius"] for operation in operations}) > 1


def test_public_api_rejects_non_pillow_input() -> None:
    with pytest.raises(TypeError, match="Pillow"):
        degrade([[0, 255]], seed=1)  # type: ignore[arg-type]
