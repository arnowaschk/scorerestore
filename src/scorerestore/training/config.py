"""Strict YAML configuration for the Milestone 6 plain-PyTorch trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scorerestore.config import ConfigError, apply_overrides, load_config

from .losses import LossWeights


class TrainingConfigError(ValueError):
    """Raised when a training configuration would make the V1 pipeline ambiguous."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    dataset_manifest: Path
    task: str
    model_backend: str
    base_channels: int
    pretrained: bool
    freeze_batch_norm: bool
    crop_size: int
    train_crops_per_epoch: int
    validation_crops: int
    foreground_fraction: float
    minimum_foreground_occupancy: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    gradient_accumulation: int
    early_stopping_patience: int | None
    device: str
    seed: int
    num_workers: int
    loss: LossWeights

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["dataset_manifest"] = str(self.dataset_manifest)
        return output


def load_training_config(path: str | Path, *, overrides: tuple[str, ...] = ()) -> TrainingConfig:
    try:
        raw = apply_overrides(load_config(path), overrides)
    except ConfigError as error:
        raise TrainingConfigError(str(error)) from error
    allowed = {
        "dataset_manifest",
        "task",
        "model",
        "crop",
        "sampling",
        "training",
        "loss",
    }
    _keys(raw, allowed, "training config")
    task = _choice(raw, "task", {"clean", "segment", "multitask"})
    model = _mapping(raw, "model")
    _keys(model, {"backend", "base_channels", "pretrained", "freeze_batch_norm"}, "model")
    model_backend = _choice(model, "backend", {"unet", "resnet18"})
    pretrained = _boolean(model, "pretrained")
    freeze_batch_norm = _boolean(model, "freeze_batch_norm")
    if model_backend == "unet" and pretrained:
        raise TrainingConfigError("model.pretrained is only supported for model.backend=resnet18")
    crop = _mapping(raw, "crop")
    _keys(crop, {"size"}, "crop")
    sampling = _mapping(raw, "sampling")
    _keys(
        sampling,
        {
            "train_crops_per_epoch",
            "validation_crops",
            "foreground_fraction",
            "minimum_foreground_occupancy",
        },
        "sampling",
    )
    training = _mapping(raw, "training")
    _keys(
        training,
        {
            "batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "gradient_accumulation",
            "early_stopping_patience",
            "device",
            "seed",
            "num_workers",
        },
        "training",
    )
    loss = _mapping(raw, "loss")
    _keys(
        loss,
        {
            "cleaning_bce",
            "cleaning_dice",
            "segmentation_bce",
            "segmentation_dice",
            "clean_task",
            "segment_task",
            "segmentation_classes",
        },
        "loss",
    )
    class_weights = _number_list(loss, "segmentation_classes", length=4)
    config = TrainingConfig(
        dataset_manifest=Path(_text(raw, "dataset_manifest")),
        task=task,
        model_backend=model_backend,
        base_channels=_positive(model, "base_channels"),
        pretrained=pretrained,
        freeze_batch_norm=freeze_batch_norm,
        crop_size=_positive(crop, "size"),
        train_crops_per_epoch=_positive(sampling, "train_crops_per_epoch"),
        validation_crops=_positive(sampling, "validation_crops"),
        foreground_fraction=_fraction(sampling, "foreground_fraction"),
        minimum_foreground_occupancy=_fraction(sampling, "minimum_foreground_occupancy"),
        batch_size=_positive(training, "batch_size"),
        epochs=_positive(training, "epochs"),
        learning_rate=_positive_number(training, "learning_rate"),
        weight_decay=_nonnegative_number(training, "weight_decay"),
        gradient_accumulation=_positive(training, "gradient_accumulation"),
        early_stopping_patience=_optional_nonnegative(training, "early_stopping_patience"),
        device=_choice(training, "device", {"auto", "cuda", "cpu"}),
        seed=_integer(training, "seed"),
        num_workers=_nonnegative(training, "num_workers"),
        loss=LossWeights(
            **{
                name: _positive_number(loss, name)
                for name in (
                    "cleaning_bce",
                    "cleaning_dice",
                    "segmentation_bce",
                    "segmentation_dice",
                    "clean_task",
                    "segment_task",
                )
            },
            segmentation_classes=class_weights,
        ),
    )
    if config.crop_size % 16:
        raise TrainingConfigError("crop.size must be divisible by 16 for the four-level U-Net")
    return config


def _mapping(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise TrainingConfigError(f"{name} must be a mapping")
    return value


def _keys(raw: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise TrainingConfigError(f"unknown {context} fields: {', '.join(sorted(unknown))}")


def _text(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise TrainingConfigError(f"{name} must be a nonempty string")
    return value


def _choice(raw: dict[str, Any], name: str, choices: set[str]) -> str:
    value = _text(raw, name)
    if value not in choices:
        raise TrainingConfigError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return value


def _integer(raw: dict[str, Any], name: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingConfigError(f"{name} must be an integer")
    return value


def _boolean(raw: dict[str, Any], name: str) -> bool:
    value = raw.get(name)
    if not isinstance(value, bool):
        raise TrainingConfigError(f"{name} must be a boolean")
    return value


def _positive(raw: dict[str, Any], name: str) -> int:
    value = _integer(raw, name)
    if value < 1:
        raise TrainingConfigError(f"{name} must be positive")
    return value


def _nonnegative(raw: dict[str, Any], name: str) -> int:
    value = _integer(raw, name)
    if value < 0:
        raise TrainingConfigError(f"{name} must be nonnegative")
    return value


def _optional_nonnegative(raw: dict[str, Any], name: str) -> int | None:
    if raw.get(name) is None:
        return None
    return _nonnegative(raw, name)


def _number(raw: dict[str, Any], name: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigError(f"{name} must be a number")
    return float(value)


def _positive_number(raw: dict[str, Any], name: str) -> float:
    value = _number(raw, name)
    if value <= 0:
        raise TrainingConfigError(f"{name} must be positive")
    return value


def _nonnegative_number(raw: dict[str, Any], name: str) -> float:
    value = _number(raw, name)
    if value < 0:
        raise TrainingConfigError(f"{name} must be nonnegative")
    return value


def _fraction(raw: dict[str, Any], name: str) -> float:
    value = _number(raw, name)
    if not 0 <= value <= 1:
        raise TrainingConfigError(f"{name} must be within [0, 1]")
    return value


def _number_list(raw: dict[str, Any], name: str, *, length: int) -> tuple[float, ...]:
    value = raw.get(name)
    if not isinstance(value, list) or len(value) != length:
        raise TrainingConfigError(f"{name} must be a {length}-item list")
    result = tuple(float(item) for item in value)
    if any(item <= 0 for item in result):
        raise TrainingConfigError(f"{name} values must be positive")
    return result
