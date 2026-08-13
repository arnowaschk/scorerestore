"""Task-aware V1 loss functions with independent sigmoid segmentation channels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class LossWeights:
    """Configurable loss weights recorded in every training run."""

    cleaning_bce: float = 1.0
    cleaning_dice: float = 1.0
    segmentation_bce: float = 1.0
    segmentation_dice: float = 1.0
    clean_task: float = 1.0
    segment_task: float = 1.0
    segmentation_classes: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


def soft_dice_loss(
    logits: Tensor, target: Tensor, *, class_weights: Tensor | None = None
) -> Tensor:
    """Mean independent sigmoid Dice loss, allowing continuous cleaning targets."""

    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(2, logits.ndim))
    intersection = (probabilities * target).sum(dimensions)
    denominator = probabilities.sum(dimensions) + target.sum(dimensions)
    loss = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    if class_weights is not None:
        loss = loss * class_weights.view(1, -1)
        return loss.sum() / (class_weights.sum() * loss.shape[0])
    return loss.mean()


def cleaning_loss(logits: Tensor, target: Tensor, weights: LossWeights) -> Tensor:
    return weights.cleaning_bce * F.binary_cross_entropy_with_logits(
        logits, target
    ) + weights.cleaning_dice * soft_dice_loss(logits, target)


def segmentation_loss(logits: Tensor, target: Tensor, weights: LossWeights) -> Tensor:
    class_weights = torch.tensor(
        weights.segmentation_classes, device=logits.device, dtype=logits.dtype
    )
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * class_weights.view(1, -1, 1, 1)).mean()
    return weights.segmentation_bce * bce + weights.segmentation_dice * soft_dice_loss(
        logits, target, class_weights=class_weights
    )


def task_loss(
    task: str,
    cleaning_logits: Tensor,
    segmentation_logits: Tensor,
    clean_target: Tensor,
    segmentation_target: Tensor,
    weights: LossWeights,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return total loss and only the active task terms for clean/segment/multitask."""

    terms: dict[str, Tensor] = {}
    if task in {"clean", "multitask"}:
        terms["clean_loss"] = cleaning_loss(cleaning_logits, clean_target, weights)
    if task in {"segment", "multitask"}:
        terms["segment_loss"] = segmentation_loss(segmentation_logits, segmentation_target, weights)
    if not terms:
        raise ValueError(f"unsupported task mode: {task}")
    total = sum(
        (weights.clean_task if name == "clean_loss" else weights.segment_task) * value
        for name, value in terms.items()
    )
    return total, terms
