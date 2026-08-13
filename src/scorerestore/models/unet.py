"""The intentionally readable custom U-Net used by ScoreRestore V1.

The network consumes grayscale intensity (black=0, white=1).  Its cleaning head predicts ink
coverage logits and its segmentation head always predicts the four independent semantic logits in
the fixed ``background, staff, notation, text`` order.  Heads are retained for every task mode so
one shared architecture can serve all V1 experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

import torch
from torch import Tensor, nn
from torch.nn import functional as F

SEMANTIC_CLASSES = ("background", "staff", "notation", "text")


class ModelBackend(StrEnum):
    """Small V1 model registry boundary."""

    UNET = "unet"
    RESNET18 = "resnet18"


@dataclass(frozen=True, slots=True)
class UNetOutput:
    """Raw logits from the two independent task heads."""

    cleaning: Tensor
    segmentation: Tensor


def _group_count(channels: int) -> int:
    """Choose a GroupNorm group count that divides narrow smoke-test channel widths."""

    return next(groups for groups in (8, 4, 2, 1) if channels % groups == 0)


class DoubleConv(nn.Module):
    """Two 3x3 convolution / GroupNorm / ReLU operations."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


class Down(nn.Module):
    """One encoder level: downsample then extract features."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.MaxPool2d(2), DoubleConv(input_channels, output_channels))

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


class Up(nn.Module):
    """One decoder level: bilinear upsample, concatenate its skip connection, then convolve."""

    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.convolution = DoubleConv(input_channels + skip_channels, output_channels)

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.convolution(torch.cat((skip, value), dim=1))


class UNet(nn.Module):
    """Four-level U-Net with one shared decoder and separate cleaning/segmentation heads."""

    def __init__(self, *, base_channels: int = 32) -> None:
        super().__init__()
        if base_channels < 2:
            raise ValueError("base_channels must be at least 2")
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.encoder = nn.ModuleList(
            (DoubleConv(1, channels[0]), *(Down(a, b) for a, b in pairwise(channels)))
        )
        self.bottleneck = Down(channels[-1], channels[-1] * 2)
        self.decoder = nn.ModuleList(
            (
                Up(channels[-1] * 2, channels[-1], channels[-1]),
                Up(channels[-1], channels[-2], channels[-2]),
                Up(channels[-2], channels[-3], channels[-3]),
                Up(channels[-3], channels[-4], channels[-4]),
            )
        )
        self.cleaning_head = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.segmentation_head = nn.Conv2d(channels[0], len(SEMANTIC_CLASSES), kernel_size=1)

    def forward(self, image: Tensor) -> UNetOutput:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("UNet expects an N x 1 x H x W grayscale tensor")
        skips: list[Tensor] = []
        value = image
        for level in self.encoder:
            value = level(value)
            skips.append(value)
        value = self.bottleneck(value)
        for level, skip in zip(self.decoder, reversed(skips), strict=True):
            value = level(value, skip)
        return UNetOutput(
            cleaning=self.cleaning_head(value), segmentation=self.segmentation_head(value)
        )


def build_model(
    backend: str = ModelBackend.UNET,
    *,
    base_channels: int = 32,
    pretrained: bool = False,
    freeze_batch_norm: bool = True,
) -> nn.Module:
    """Build a V1 model through the intentionally small registry interface."""

    if backend == ModelBackend.UNET:
        if pretrained:
            raise ValueError("the custom U-Net has no pretrained V1 weights")
        return UNet(base_channels=base_channels)
    if backend == ModelBackend.RESNET18:
        from .resnet18 import ResNet18UNet

        return ResNet18UNet(
            decoder_channels=base_channels,
            pretrained=pretrained,
            freeze_batch_norm=freeze_batch_norm,
        )
    raise ValueError(f"unsupported V1 model backend: {backend}")


def count_parameters(model: nn.Module) -> int:
    """Return the trainable parameter count recorded in run provenance."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_provenance(model: nn.Module) -> dict[str, object]:
    """Return architecture and weight origin without inventing unavailable provenance."""

    provenance = getattr(model, "provenance", None)
    if callable(provenance):
        return provenance()
    return {
        "backend": ModelBackend.UNET,
        "architecture": "custom_unet",
        "pretrained": False,
        "weights": None,
        "weights_url": None,
        "grayscale_adaptation": None,
        "batch_norm": "GroupNorm; no BatchNorm layers",
    }
