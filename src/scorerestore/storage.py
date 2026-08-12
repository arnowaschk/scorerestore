"""Filesystem-backed materialized storage for public V1 datasets."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


class StorageError(ValueError):
    """Raised for unsafe or invalid storage keys."""


@dataclass(frozen=True, slots=True)
class FilesystemStorage:
    """Small root-confined filesystem implementation of ``StorageBackend``."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())

    def path(self, relative_path: str | Path) -> Path:
        """Resolve a storage-relative key while preventing root escape."""

        key = Path(relative_path)
        if key.is_absolute() or ".." in key.parts:
            raise StorageError(f"storage path must be relative and confined: {relative_path}")
        resolved = (self.root / key).resolve()
        if not resolved.is_relative_to(self.root):
            raise StorageError(f"storage path escapes root: {relative_path}")
        return resolved

    def make_directory(self, relative_path: str | Path) -> Path:
        path = self.path(relative_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_bytes(self, relative_path: str | Path, content: bytes) -> Path:
        destination = self.path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}-",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            temporary_path.chmod(0o644)
            temporary_path.replace(destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return destination

    def write_text(self, relative_path: str | Path, content: str) -> Path:
        return self.write_bytes(relative_path, content.encode("utf-8"))

    def copy_file(self, source: str | Path, relative_path: str | Path) -> Path:
        source_path = Path(source)
        if not source_path.is_file():
            raise StorageError(f"source file does not exist: {source_path}")
        destination = self.path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        destination.chmod(0o644)
        return destination
