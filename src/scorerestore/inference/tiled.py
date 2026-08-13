"""Bounded-memory, overlap-blended tiled inference for ScoreRestore V1."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

from scorerestore import __version__
from scorerestore.models import build_model, model_provenance


@dataclass(frozen=True, slots=True)
class CleanResult:
    """Outputs from one page, all exactly matching the original raster dimensions."""

    cleaned: Image.Image
    probability: Image.Image
    masks: dict[str, Image.Image]
    mask_probabilities: dict[str, Image.Image]
    metadata: dict[str, Any]


def clean(
    image: Image.Image | np.ndarray,
    *,
    model: nn.Module,
    device: str = "auto",
    tile_size: int = 1024,
    overlap: int = 128,
    cleaning_threshold: float = 0.5,
    segmentation_threshold: float = 0.5,
) -> CleanResult:
    """Restore an arbitrary-size grayscale page using bounded-memory overlap blending."""

    if tile_size < 16 or tile_size % 16:
        raise ValueError("tile_size must be at least 16 and divisible by 16")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must be within [0, tile_size)")
    if not 0 <= cleaning_threshold <= 1 or not 0 <= segmentation_threshold <= 1:
        raise ValueError("thresholds must be within [0, 1]")
    tensor = _image_tensor(image)
    height, width = tensor.shape[-2:]
    selected_device = _device(device)
    model = model.to(selected_device)
    model.eval()
    cleaning_logits, segmentation_logits, tile_count = _predict_tiled(
        model, tensor, selected_device, tile_size, overlap
    )
    cleaning_probability = torch.sigmoid(cleaning_logits)[0, 0]
    segmentation_probability = torch.sigmoid(segmentation_logits)[0]
    classes = ("background", "staff", "notation", "text")
    masks = {
        name: _probability_image(segmentation_probability[index] >= segmentation_threshold)
        for index, name in enumerate(classes)
    }
    probabilities = {
        name: _probability_image(segmentation_probability[index])
        for index, name in enumerate(classes)
    }
    return CleanResult(
        cleaned=_binary_cleaned_image(cleaning_probability, cleaning_threshold),
        probability=_probability_image(cleaning_probability),
        masks=masks,
        mask_probabilities=probabilities,
        metadata={
            "input_dimensions": {"width": width, "height": height},
            "device": str(selected_device),
            "tile_size": tile_size,
            "overlap": overlap,
            "tile_count": tile_count,
            "cleaning_threshold": cleaning_threshold,
            "segmentation_threshold": segmentation_threshold,
            "semantics": "four independent sigmoid channels: background, staff, notation, text",
        },
    )


def load_checkpoint_model(
    path: str | Path, *, device: str = "auto"
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the architecture embedded in a V1 training checkpoint and return its provenance."""

    selected_device = _device(device)
    try:
        checkpoint = torch.load(Path(path), map_location=selected_device, weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"cannot load checkpoint {path}: {error}") from error
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint must contain config and model_state_dict")
    try:
        model = build_model(
            config["model_backend"],
            base_channels=config["base_channels"],
            pretrained=config["pretrained"],
            freeze_batch_norm=config["freeze_batch_norm"],
        )
        model.load_state_dict(state)
    except (KeyError, RuntimeError, ValueError) as error:
        raise ValueError(f"checkpoint model metadata is incompatible: {error}") from error
    model.to(selected_device).eval()
    return model, {
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_loss": checkpoint.get("validation_loss"),
        "model": model_provenance(model),
        "scorerestore_version": __version__,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "python_version": platform.python_version(),
    }


def _predict_tiled(
    model: nn.Module, image: Tensor, device: torch.device, tile_size: int, overlap: int
) -> tuple[Tensor, Tensor, int]:
    _, _, height, width = image.shape
    rows, columns = (
        _tile_starts(height, tile_size, overlap),
        _tile_starts(width, tile_size, overlap),
    )
    clean_sum = torch.zeros((1, 1, height, width), device=device)
    segment_sum = torch.zeros((1, 4, height, width), device=device)
    weights = torch.zeros((1, 1, height, width), device=device)
    window = _blend_window(tile_size, device)
    with torch.no_grad():
        for top in rows:
            for left in columns:
                tile = _padded_tile(image, top, left, tile_size).to(device)
                output = model(tile)
                actual_height, actual_width = (
                    min(tile_size, height - top),
                    min(tile_size, width - left),
                )
                tile_window = window[:, :, :actual_height, :actual_width]
                destination = (
                    slice(None),
                    slice(None),
                    slice(top, top + actual_height),
                    slice(left, left + actual_width),
                )
                clean_sum[destination] += (
                    output.cleaning[:, :, :actual_height, :actual_width] * tile_window
                )
                segment_sum[destination] += (
                    output.segmentation[:, :, :actual_height, :actual_width] * tile_window
                )
                weights[destination] += tile_window
    return clean_sum / weights, segment_sum / weights, len(rows) * len(columns)


def _tile_starts(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    if length <= tile_size:
        return (0,)
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return tuple(starts)


def _padded_tile(image: Tensor, top: int, left: int, tile_size: int) -> Tensor:
    tile = image[:, :, top : top + tile_size, left : left + tile_size]
    pad_height, pad_width = tile_size - tile.shape[-2], tile_size - tile.shape[-1]
    # ReflectionPad only permits a single pad smaller than its input. Repeated reflection grows
    # tiny pages safely while preserving the required boundary semantics.
    while pad_height or pad_width:
        height, width = tile.shape[-2:]
        if height == 1 or width == 1:
            return F.pad(tile, (0, pad_width, 0, pad_height), mode="replicate")
        next_height, next_width = min(pad_height, height - 1), min(pad_width, width - 1)
        tile = F.pad(tile, (0, next_width, 0, next_height), mode="reflect")
        pad_height, pad_width = pad_height - next_height, pad_width - next_width
    return tile


def _blend_window(tile_size: int, device: torch.device) -> Tensor:
    # A raised-cosine window blends logits smoothly while its epsilon preserves border coverage.
    axis = torch.hann_window(tile_size, periodic=False, device=device).clamp_min(1e-3)
    return (axis[:, None] * axis[None, :]).unsqueeze(0).unsqueeze(0)


def _image_tensor(image: Image.Image | np.ndarray) -> Tensor:
    if isinstance(image, Image.Image):
        pixels = np.asarray(image.convert("L"), dtype=np.float32)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            pixels = image
        elif image.ndim == 3 and image.shape[2] in {3, 4}:
            pixels = np.asarray(Image.fromarray(image).convert("L"))
        else:
            raise ValueError("NumPy image must be HxW grayscale or HxWx3/4 color")
        if not np.issubdtype(pixels.dtype, np.number):
            raise ValueError("NumPy image must contain numeric values")
        pixels = pixels.astype(np.float32)
        if pixels.max(initial=0) <= 1:
            pixels *= 255
    else:
        raise TypeError("image must be a Pillow image or NumPy array")
    return torch.from_numpy(np.ascontiguousarray(pixels / 255.0)).unsqueeze(0).unsqueeze(0)


def _device(selection: str) -> torch.device:
    if selection == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(
        "cuda"
        if selection == "auto" and torch.cuda.is_available()
        else selection
        if selection != "auto"
        else "cpu"
    )


def _probability_image(value: Tensor) -> Image.Image:
    pixels = value.to(torch.float32).clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(pixels, mode="L")


def _binary_cleaned_image(probability: Tensor, threshold: float) -> Image.Image:
    # Final V1 cleaning is binary black ink (0), white background (255).
    pixels = torch.where(probability >= threshold, 0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(pixels, mode="L")


def metadata_json(metadata: dict[str, Any]) -> str:
    """Serialize run metadata consistently for the CLI adapter."""

    return json.dumps(metadata, indent=2, sort_keys=True) + "\n"
