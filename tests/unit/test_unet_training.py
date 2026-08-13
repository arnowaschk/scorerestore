from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from scorerestore.models import build_model, model_provenance
from scorerestore.training import TrainingConfigError, load_training_config
from scorerestore.training.losses import LossWeights, task_loss
from scorerestore.training.runner import _format_duration


def test_custom_unet_preserves_spatial_shape_and_has_fixed_independent_heads() -> None:
    output = build_model(base_channels=4)(torch.rand(2, 1, 64, 80))

    assert output.cleaning.shape == (2, 1, 64, 80)
    assert output.segmentation.shape == (2, 4, 64, 80)


def test_task_modes_only_compute_their_enabled_loss_terms() -> None:
    cleaning = torch.zeros(1, 1, 16, 16, requires_grad=True)
    segmentation = torch.zeros(1, 4, 16, 16, requires_grad=True)
    clean_target = torch.zeros_like(cleaning)
    segmentation_target = torch.zeros_like(segmentation)

    _, clean_terms = task_loss(
        "clean", cleaning, segmentation, clean_target, segmentation_target, LossWeights()
    )
    _, segment_terms = task_loss(
        "segment", cleaning, segmentation, clean_target, segmentation_target, LossWeights()
    )
    total, multitask_terms = task_loss(
        "multitask", cleaning, segmentation, clean_target, segmentation_target, LossWeights()
    )

    assert set(clean_terms) == {"clean_loss"}
    assert set(segment_terms) == {"segment_loss"}
    assert set(multitask_terms) == {"clean_loss", "segment_loss"}
    total.backward()
    assert cleaning.grad is not None
    assert segmentation.grad is not None


def test_tiny_static_batch_can_be_intentionally_overfit() -> None:
    """A deliberately repetitive tiny problem verifies forward/loss/backward optimizer wiring."""

    torch.manual_seed(7)
    model = build_model(base_channels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0)
    image = torch.zeros(1, 1, 32, 32)
    clean_target = torch.zeros_like(image)
    clean_target[:, :, 8:24, 8:24] = 1
    segmentation_target = torch.cat(
        (1 - clean_target, clean_target, clean_target, torch.zeros_like(clean_target)), dim=1
    )
    weights = LossWeights()

    losses = []
    for _ in range(20):
        output = model(image)
        loss, _ = task_loss(
            "multitask",
            output.cleaning,
            output.segmentation,
            clean_target,
            segmentation_target,
            weights,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0]


def test_progress_duration_is_human_readable() -> None:
    assert _format_duration(0.4) == "0:00:00"
    assert _format_duration(65.2) == "0:01:05"
    assert _format_duration(3_661) == "1:01:01"


def test_resnet18_backend_forward_backward_and_frozen_batch_norm() -> None:
    model = build_model("resnet18", base_channels=4, pretrained=False, freeze_batch_norm=True)
    model.train()
    image = torch.rand(1, 1, 64, 64)
    output = model(image)
    loss = output.cleaning.mean() + output.segmentation.mean()
    loss.backward()

    assert output.cleaning.shape == (1, 1, 64, 64)
    assert output.segmentation.shape == (1, 4, 64, 64)
    assert model.stem[1].training is False
    assert model.stem[1].weight.grad is not None


def test_resnet18_pretrained_weights_load_and_rgb_kernels_are_averaged() -> None:
    """Official ImageNet weights are a small integration check for the transfer-learning path."""

    if os.environ.get("SCORERESTORE_TEST_PRETRAINED") != "1":
        pytest.skip("set SCORERESTORE_TEST_PRETRAINED=1 to fetch/check pretrained weights")
    from torchvision.models import ResNet18_Weights

    model = build_model("resnet18", base_channels=4, pretrained=True, freeze_batch_norm=True)
    expected = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False)["conv1.weight"]

    assert torch.equal(model.stem[0].weight, expected.mean(dim=1, keepdim=True))
    assert model_provenance(model)["weights"] == "ResNet18_Weights.IMAGENET1K_V1"


def test_model_configuration_rejects_pretrained_custom_unet(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text(
        """dataset_manifest: example.jsonl
task: clean
model:
  backend: unet
  base_channels: 4
  pretrained: true
  freeze_batch_norm: true
crop: {size: 64}
sampling:
  train_crops_per_epoch: 1
  validation_crops: 1
  foreground_fraction: 0.8
  minimum_foreground_occupancy: 0.01
training:
  batch_size: 1
  epochs: 1
  learning_rate: 0.001
  weight_decay: 0.0
  gradient_accumulation: 1
  early_stopping_patience: null
  device: cpu
  seed: 1
  num_workers: 0
loss:
  cleaning_bce: 1.0
  cleaning_dice: 1.0
  segmentation_bce: 1.0
  segmentation_dice: 1.0
  clean_task: 1.0
  segment_task: 1.0
  segmentation_classes: [1.0, 1.0, 1.0, 1.0]
""",
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigError, match="only supported"):
        load_training_config(config)
