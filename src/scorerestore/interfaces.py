"""Small public V1 interfaces reserved by the ScoreRestore specification."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from scorerestore.provenance import ScoreAsset


@runtime_checkable
class StorageBackend(Protocol):
    """Minimal storage operations used by materialized V1 datasets."""

    root: Path

    def path(self, relative_path: str | Path) -> Path: ...

    def make_directory(self, relative_path: str | Path) -> Path: ...

    def write_bytes(self, relative_path: str | Path, content: bytes) -> Path: ...

    def write_text(self, relative_path: str | Path, content: str) -> Path: ...

    def copy_file(self, source: str | Path, relative_path: str | Path) -> Path: ...


@runtime_checkable
class DatasetSource(Protocol):
    """Source of validated existing score assets for V1 dataset generation."""

    def assets(self) -> Iterable[ScoreAsset]: ...


@runtime_checkable
class ScoreGenerator(Protocol):
    """Reserved symbolic-score generator boundary; V1 has no implementation."""

    def generate(self, *, seed: int) -> str: ...
