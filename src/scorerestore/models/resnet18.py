"""TorchVision ResNet-18 transfer encoder with a small readable U-Net-like decoder.

The ImageNet encoder is adapted for ScoreRestore's single grayscale channel by replacing the
first convolution with the arithmetic mean of its RGB kernels. This is the standard direct
three-to-one channel projection specified for V1. BatchNorm affine parameters remain trainable,
but its running statistics are deliberately frozen because V1 page-tile batches can be small.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .unet import SEMANTIC_CLASSES, DoubleConv, UNetOutput

RESNET18_WEIGHTS_NAME = "ResNet18_Weights.IMAGENET1K_V1"


class ResNetUp(nn.Module):
    """Bilinear decoder level that consumes one ResNet skip feature."""

    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.convolution = DoubleConv(input_channels + skip_channels, output_channels)

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.convolution(torch.cat((skip, value), dim=1))


class ResNet18UNet(nn.Module):
    """Shared ResNet-18 encoder / U-Net-like decoder with the V1 two-head output contract."""

    def __init__(
        self, *, decoder_channels: int = 64, pretrained: bool, freeze_batch_norm: bool
    ) -> None:
        super().__init__()
        if decoder_channels < 2:
            raise ValueError("decoder_channels must be at least 2")
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        original = backbone.conv1
        grayscale = nn.Conv2d(
            1,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )
        with torch.no_grad():
            # Averaging RGB kernels is the explicit V1 grayscale adaptation.
            grayscale.weight.copy_(original.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = grayscale
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.up4 = ResNetUp(512, 256, 256)
        self.up3 = ResNetUp(256, 128, 128)
        self.up2 = ResNetUp(128, 64, 64)
        self.up1 = ResNetUp(64, 64, decoder_channels)
        self.final = DoubleConv(decoder_channels, decoder_channels)
        self.cleaning_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.segmentation_head = nn.Conv2d(decoder_channels, len(SEMANTIC_CLASSES), kernel_size=1)
        self.pretrained = pretrained
        self.freeze_batch_norm = freeze_batch_norm
        if freeze_batch_norm:
            self._freeze_encoder_batch_norm()

    def forward(self, image: Tensor) -> UNetOutput:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("ResNet18UNet expects an N x 1 x H x W grayscale tensor")
        stem = self.stem(image)
        layer1 = self.layer1(self.pool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        value = self.layer4(layer3)
        value = self.up4(value, layer3)
        value = self.up3(value, layer2)
        value = self.up2(value, layer1)
        value = self.up1(value, stem)
        value = F.interpolate(value, size=image.shape[-2:], mode="bilinear", align_corners=False)
        value = self.final(value)
        return UNetOutput(
            cleaning=self.cleaning_head(value), segmentation=self.segmentation_head(value)
        )

    def train(self, mode: bool = True) -> ResNet18UNet:
        """Keep pretrained BatchNorm running estimates fixed during small-batch fine-tuning."""

        super().train(mode)
        if mode and self.freeze_batch_norm:
            self._freeze_encoder_batch_norm()
        return self

    def provenance(self) -> dict[str, Any]:
        """The exact transfer-weight and adaptation decisions recorded in every run."""

        weights = ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None
        return {
            "backend": "resnet18",
            "architecture": "torchvision_resnet18_unet_decoder",
            "pretrained": self.pretrained,
            "weights": RESNET18_WEIGHTS_NAME if weights is not None else None,
            "weights_url": weights.url if weights is not None else None,
            "weights_dataset": "ImageNet-1K",
            "grayscale_adaptation": "conv1 RGB kernels averaged across input channels",
            "batch_norm": (
                "encoder BatchNorm running statistics frozen; affine parameters trainable"
                if self.freeze_batch_norm
                else "encoder BatchNorm running statistics trainable"
            ),
        }

    def _freeze_encoder_batch_norm(self) -> None:
        for module in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            for child in module.modules():
                if isinstance(child, nn.BatchNorm2d):
                    child.eval()
