"""Materialized-dataset tensors and deterministic foreground-aware crop sampling."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from scorerestore.dataset.loader import MaterializedDataset
from scorerestore.models.unet import SEMANTIC_CLASSES


def image_intensity_tensor(image: Image.Image) -> Tensor:
    """Convert an image to N/A grayscale intensity where black=0 and white=1."""

    pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0)


def ink_coverage_tensor(image: Image.Image) -> Tensor:
    """Convert pristine grayscale intensity to continuous desired ink coverage (ink=1)."""

    return 1.0 - image_intensity_tensor(image)


def mask_tensor(image: Image.Image) -> Tensor:
    """Convert a binary 0/255 semantic mask to 0/1 foreground membership."""

    return (image_intensity_tensor(image) >= 0.5).to(torch.float32)


class ForegroundCropDataset(Dataset[dict[str, Tensor]]):
    """Deterministic crop Dataset with an 80/20 foreground-aware/uniform default split."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        crop_size: int,
        crops_per_epoch: int,
        foreground_fraction: float,
        minimum_foreground_occupancy: float,
        seed: int,
    ) -> None:
        dataset = MaterializedDataset(manifest_path)
        self.records = tuple(record for record in dataset._records if record["split"] == split)
        if not self.records:
            raise ValueError(f"dataset has no {split!r} samples")
        self.dataset = dataset
        self.crop_size = crop_size
        self.crops_per_epoch = crops_per_epoch
        self.foreground_fraction = foreground_fraction
        self.minimum_foreground_occupancy = minimum_foreground_occupancy
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.crops_per_epoch

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        generator = np.random.default_rng(_sample_seed(self.seed, self.epoch, index))
        record = self.records[int(generator.integers(len(self.records)))]
        sample = self.dataset._load(record)
        image = image_intensity_tensor(sample.image)
        clean = ink_coverage_tensor(sample.clean)
        masks = torch.cat([mask_tensor(sample.masks[name]) for name in SEMANTIC_CLASSES], dim=0)
        top, left = self._crop_origin(masks, generator)
        return {
            "image": _crop_or_pad(image, top, left, self.crop_size, padding_value=1.0),
            "clean": _crop_or_pad(clean, top, left, self.crop_size, padding_value=0.0),
            "segmentation": _crop_or_pad(masks, top, left, self.crop_size, padding_value=0.0),
        }

    def _crop_origin(self, masks: Tensor, generator: np.random.Generator) -> tuple[int, int]:
        height, width = masks.shape[-2:]
        foreground = torch.any(masks[1:] > 0.5, dim=0).numpy()
        foreground_aware = generator.random() < self.foreground_fraction and foreground.any()
        attempts = 20 if foreground_aware else 1
        fallback = _random_origin(height, width, self.crop_size, generator)
        for _ in range(attempts):
            top, left = _random_origin(height, width, self.crop_size, generator)
            if not foreground_aware:
                return top, left
            crop = _crop_or_pad(
                torch.from_numpy(foreground).unsqueeze(0),
                top,
                left,
                self.crop_size,
                padding_value=0,
            )
            if float(crop.float().mean()) >= self.minimum_foreground_occupancy:
                return top, left
        return fallback


def _sample_seed(seed: int, epoch: int, index: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{epoch}:{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _random_origin(
    height: int, width: int, size: int, generator: np.random.Generator
) -> tuple[int, int]:
    return (
        int(generator.integers(max(1, height - size + 1))),
        int(generator.integers(max(1, width - size + 1))),
    )


def _crop_or_pad(value: Tensor, top: int, left: int, size: int, *, padding_value: float) -> Tensor:
    """Crop tensor content and deterministically pad images white, targets/masks empty."""

    channels, height, width = value.shape
    result = torch.full((channels, size, size), padding_value, dtype=value.dtype)
    source_bottom, source_right = min(height, top + size), min(width, left + size)
    copied_height, copied_width = max(0, source_bottom - top), max(0, source_right - left)
    if copied_height and copied_width:
        result[:, :copied_height, :copied_width] = value[:, top:source_bottom, left:source_right]
    return result
