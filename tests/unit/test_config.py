from __future__ import annotations

from pathlib import Path

import pytest

from scorerestore.config import ConfigError, apply_overrides, load_config, resolve_config


def test_load_config_reads_yaml_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("training:\n  epochs: 2\n  amp: false\n", encoding="utf-8")

    assert load_config(path) == {"training": {"epochs": 2, "amp": False}}


def test_empty_config_resolves_to_empty_mapping(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    assert load_config(path) == {}


def test_config_root_must_be_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- invalid\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(path)


def test_apply_overrides_preserves_types_and_input() -> None:
    original = {"training": {"epochs": 2}, "name": "base"}

    resolved = apply_overrides(
        original,
        ["training.epochs=3", "training.amp=true", "tags=[smoke, cpu]"],
    )

    assert original == {"training": {"epochs": 2}, "name": "base"}
    assert resolved == {
        "training": {"epochs": 3, "amp": True},
        "name": "base",
        "tags": ["smoke", "cpu"],
    }


def test_apply_overrides_rejects_non_mapping_parent() -> None:
    with pytest.raises(ConfigError, match="is not a mapping"):
        apply_overrides({"training": 1}, ["training.epochs=3"])


@pytest.mark.parametrize("override", ["missing_equals", "=3", "a..b=3"])
def test_apply_overrides_rejects_invalid_syntax(override: str) -> None:
    with pytest.raises(ConfigError):
        apply_overrides({}, [override])


def test_resolve_config_supports_override_only() -> None:
    assert resolve_config(overrides=["training.epochs=3"]) == {"training": {"epochs": 3}}
