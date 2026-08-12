"""Lightweight materialized target loader independent of PyTorch training code."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .manifest import read_manifest_records, validate_dataset_manifest


@dataclass(frozen=True, slots=True)
class MaterializedSample:
    """One degraded input with pristine and four semantic target images."""

    image: Image.Image
    clean: Image.Image
    masks: dict[str, Image.Image]
    record: dict[str, Any]


class MaterializedDataset(Sequence[MaterializedSample]):
    """Sequence-style V1 loader; a later milestone may adapt it to PyTorch Dataset."""

    def __init__(self, manifest_path: str | Path, *, verify: bool = True) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        if verify:
            report = validate_dataset_manifest(self.manifest_path)
            self._records = report.records
        else:
            self._records = read_manifest_records(self.manifest_path)
        self.dataset_root = self.manifest_path.parent.parent

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int | slice) -> MaterializedSample | list[MaterializedSample]:
        if isinstance(index, slice):
            return [self._load(record) for record in self._records[index]]
        return self._load(self._records[index])

    def __iter__(self) -> Iterator[MaterializedSample]:
        for record in self._records:
            yield self._load(record)

    def _load(self, record: dict[str, Any]) -> MaterializedSample:
        image = _open_copy(self.dataset_root / record["input_path"])
        clean = _open_copy(self.dataset_root / record["clean_target_path"])
        masks = {
            name: _open_copy(self.dataset_root / path)
            for name, path in record["mask_paths"].items()
        }
        return MaterializedSample(image=image, clean=clean, masks=masks, record=record)


def _open_copy(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.copy()
