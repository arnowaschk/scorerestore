from __future__ import annotations

from pathlib import Path

import pytest

from scorerestore.interfaces import StorageBackend
from scorerestore.storage import FilesystemStorage, StorageError


def test_filesystem_storage_writes_and_copies_inside_root(tmp_path: Path) -> None:
    storage = FilesystemStorage(tmp_path / "root")
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")

    written = storage.write_text("manifests/test.json", "{}\n")
    copied = storage.copy_file(source, "inputs/source.bin")

    assert isinstance(storage, StorageBackend)
    assert written.read_text(encoding="utf-8") == "{}\n"
    assert copied.read_bytes() == b"source"
    assert written.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute/path"])
def test_filesystem_storage_rejects_root_escape(tmp_path: Path, unsafe: str) -> None:
    storage = FilesystemStorage(tmp_path / "root")

    with pytest.raises(StorageError, match=r"confined|relative"):
        storage.path(unsafe)
