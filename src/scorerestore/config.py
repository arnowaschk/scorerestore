"""Small YAML-first configuration loader used by ScoreRestore commands."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, TypeAlias

import yaml

Config: TypeAlias = dict[str, Any]


class ConfigError(ValueError):
    """Raised when a configuration file or CLI override is invalid."""


def load_config(path: str | Path) -> Config:
    """Load a YAML mapping from *path*.

    Empty files resolve to an empty mapping. A configuration root must be a mapping because dotted
    CLI overrides address named fields.
    """

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"Cannot read configuration {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in configuration {config_path}: {error}") from error

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration root in {config_path} must be a mapping")
    if not all(isinstance(key, str) for key in raw):
        raise ConfigError(f"Configuration keys in {config_path} must be strings")
    return raw


def apply_overrides(config: Config, overrides: list[str] | tuple[str, ...]) -> Config:
    """Return a deep copy of *config* with ``key.path=value`` YAML overrides applied.

    Values use YAML scalar/list/mapping syntax, so values such as ``3``, ``true``, and ``[a, b]``
    retain their intended types. Missing mapping paths are created; an existing non-mapping path is
    rejected to surface likely configuration mistakes.
    """

    resolved = deepcopy(config)
    for override in overrides:
        key_path, value = _parse_override(override)
        cursor: Config = resolved
        for key in key_path[:-1]:
            existing = cursor.get(key)
            if existing is None:
                child: Config = {}
                cursor[key] = child
                cursor = child
            elif isinstance(existing, dict):
                cursor = existing
            else:
                joined = ".".join(key_path)
                raise ConfigError(f"Cannot apply override {joined!r}: {key!r} is not a mapping")
        cursor[key_path[-1]] = value
    return resolved


def resolve_config(
    path: str | Path | None = None,
    overrides: list[str] | tuple[str, ...] = (),
) -> Config:
    """Load an optional YAML file and apply CLI overrides."""

    return apply_overrides(load_config(path) if path is not None else {}, overrides)


def _parse_override(override: str) -> tuple[list[str], Any]:
    if "=" not in override:
        raise ConfigError(f"Override must use key.path=value syntax: {override!r}")

    raw_path, raw_value = override.split("=", maxsplit=1)
    key_path = raw_path.split(".")
    if not raw_path or any(not key for key in key_path):
        raise ConfigError(f"Override has an invalid key path: {override!r}")

    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as error:
        raise ConfigError(f"Override has invalid YAML value: {override!r}") from error
    return key_path, value
