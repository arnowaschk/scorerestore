"""A compact, explicit plain-PyTorch V1 training loop."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import yaml
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from scorerestore import __version__
from scorerestore.dataset.manifest import sha256_file
from scorerestore.lilypond.constants import LILYPOND_VERSION
from scorerestore.models import build_model, count_parameters, model_provenance

from .config import TrainingConfig
from .data import ForegroundCropDataset
from .losses import task_loss


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Locations and final summary produced by a completed training run."""

    output_directory: Path
    checkpoint_path: Path
    best_validation_loss: float
    epochs_completed: int


def train(
    config: TrainingConfig, output_directory: str | Path, *, update: bool = False
) -> TrainingResult:
    """Train a model, optionally resuming an interrupted compatible run in place."""

    output = Path(output_directory).resolve()
    if output.exists() and not update:
        raise ValueError(f"output directory already exists: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"training output is not a directory: {output}")
    if update and output.exists():
        completed = _completed_result(output, config)
        if completed is not None:
            return completed
        _validate_resume_config(output, config)
    device = _device(config.device)
    _seed_everything(config.seed)
    manifest = config.dataset_manifest.resolve()
    train_data = ForegroundCropDataset(
        manifest,
        split="train",
        crop_size=config.crop_size,
        crops_per_epoch=config.train_crops_per_epoch,
        foreground_fraction=config.foreground_fraction,
        minimum_foreground_occupancy=config.minimum_foreground_occupancy,
        seed=config.seed,
    )
    # Datasets without a validation split remain useful for an intentional tiny overfit check.
    try:
        validation_data = ForegroundCropDataset(
            manifest,
            split="validation",
            crop_size=config.crop_size,
            crops_per_epoch=config.validation_crops,
            foreground_fraction=config.foreground_fraction,
            minimum_foreground_occupancy=config.minimum_foreground_occupancy,
            seed=config.seed + 1,
        )
    except ValueError:
        validation_data = ForegroundCropDataset(
            manifest,
            split="train",
            crop_size=config.crop_size,
            crops_per_epoch=config.validation_crops,
            foreground_fraction=config.foreground_fraction,
            minimum_foreground_occupancy=config.minimum_foreground_occupancy,
            seed=config.seed + 1,
        )
    train_loader = _loader(train_data, config, shuffle=False)
    validation_loader = _loader(validation_data, config, shuffle=False)
    model = build_model(
        config.model_backend,
        base_channels=config.base_channels,
        pretrained=config.pretrained,
        freeze_batch_norm=config.freeze_batch_norm,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    amp_dtype = _cuda_amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    output.mkdir(parents=True, exist_ok=update)
    logger = ExperimentLogger(output)
    best_loss, stale_epochs, completed = float("inf"), 0, 0
    best_checkpoint = output / "checkpoints/best.pt"
    last_checkpoint = output / "checkpoints/last.pt"
    if update and last_checkpoint.is_file():
        state = _load_resume_state(last_checkpoint, model, optimizer, scaler, config)
        best_loss, stale_epochs, completed = state
        logger.retain_through_epoch(completed)
        logger.update_environment({"resumed_at": _timestamp()})
    elif update and output.exists() and (output / "config.yaml").is_file():
        raise ValueError(
            f"cannot resume {output}: checkpoints/last.pt is missing; start a new output directory"
        )
    else:
        logger.write_static(config, _environment(config, device, model, manifest))
    print(
        f"Starting {config.task} training on {device}: {config.epochs} epoch(s), "
        f"{len(train_loader)} train and {len(validation_loader)} validation batch(es)/epoch.",
        flush=True,
    )
    try:
        for epoch in range(completed + 1, config.epochs + 1):
            train_data.set_epoch(epoch)
            validation_data.set_epoch(epoch)
            train_metrics = _epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                config,
                device,
                amp_dtype=amp_dtype,
                epoch=epoch,
                total_epochs=config.epochs,
            )
            validation_metrics = _epoch(
                model,
                validation_loader,
                None,
                scaler,
                config,
                device,
                amp_dtype=amp_dtype,
                epoch=epoch,
                total_epochs=config.epochs,
            )
            completed = epoch
            row = {
                "epoch": epoch,
                **train_metrics,
                **validation_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            logger.append(row)
            print(
                f"Epoch {epoch}/{config.epochs} complete | "
                f"train loss {train_metrics['train_loss']:.5f} | "
                f"validation loss {validation_metrics['validation_loss']:.5f}",
                flush=True,
            )
            current = validation_metrics["validation_loss"]
            if current < best_loss:
                best_loss, stale_epochs = current, 0
                _atomic_torch_save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config.to_dict(),
                        "epoch": epoch,
                        "validation_loss": current,
                    },
                    best_checkpoint,
                )
            else:
                stale_epochs += 1
            _atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "config": config.to_dict(),
                    "epoch": completed,
                    "best_validation_loss": best_loss,
                    "stale_epochs": stale_epochs,
                },
                last_checkpoint,
            )
            if (
                config.early_stopping_patience is not None
                and stale_epochs > config.early_stopping_patience
            ):
                break
        logger.write_plots()
        _write_prediction(model, validation_loader, output / "comparisons", device)
        logger.update_environment({"ended_at": _timestamp()})
        logger.write_summary(
            {
                "task": config.task,
                "epochs_completed": completed,
                "best_validation_loss": best_loss,
                "checkpoint": str(best_checkpoint.relative_to(output)),
            }
        )
    except Exception:
        # Keep partial metrics/provenance for diagnosis rather than silently deleting evidence.
        raise
    return TrainingResult(output, best_checkpoint, best_loss, completed)


def _completed_result(output: Path, config: TrainingConfig) -> TrainingResult | None:
    """Return a completed compatible run so ``--update`` is safely idempotent."""

    summary_path = output / "report" / "summary.json"
    best_checkpoint = output / "checkpoints" / "best.pt"
    if not summary_path.is_file() or not best_checkpoint.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        epochs = summary["epochs_completed"]
        best_loss = summary["best_validation_loss"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        return None
    if isinstance(best_loss, bool) or not isinstance(best_loss, (int, float)):
        return None
    _validate_resume_config(output, config)
    return TrainingResult(output, best_checkpoint, float(best_loss), epochs)


def _validate_resume_config(output: Path, config: TrainingConfig) -> None:
    config_path = output / "config.yaml"
    if not config_path.is_file():
        return
    try:
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read existing training config: {error}") from error
    if _config_signature(saved) != _config_signature(config.to_dict()):
        raise ValueError(
            "existing training output uses a different configuration; start a new output directory"
        )


def _load_resume_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    config: TrainingConfig,
) -> tuple[float, int, int]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("config") != config.to_dict():
            raise ValueError("saved checkpoint configuration differs")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        completed = state["epoch"]
        best_loss = state["best_validation_loss"]
        stale_epochs = state["stale_epochs"]
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"cannot resume training from {path}: {error}") from error
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or isinstance(stale_epochs, bool)
        or not isinstance(stale_epochs, int)
        or isinstance(best_loss, bool)
        or not isinstance(best_loss, (int, float))
    ):
        raise ValueError(f"cannot resume training from {path}: invalid checkpoint state")
    return float(best_loss), stale_epochs, completed


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Persist a resumable checkpoint without exposing a partial file after interruption."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _config_signature(config: object) -> str:
    """Compare YAML and checkpoint configurations despite YAML tuple-to-list conversion."""

    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _loader(
    dataset: ForegroundCropDataset, config: TrainingConfig, *, shuffle: bool
) -> DataLoader[dict[str, Tensor]]:
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers
    )


def _epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler,
    config: TrainingConfig,
    device: torch.device,
    *,
    amp_dtype: torch.dtype | None,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    prefix = "train" if training else "validation"
    totals: dict[str, float] = {"loss": 0.0}
    batches = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    phase = "train" if training else "validation"
    started = time.monotonic()
    update_interval = max(1, len(loader) // 20)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader, start=1):
            image = batch["image"].to(device)
            clean = batch["clean"].to(device)
            segmentation = batch["segmentation"].to(device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                output = model(image)
                loss, terms = task_loss(
                    config.task,
                    output.cleaning,
                    output.segmentation,
                    clean,
                    segmentation,
                    config.loss,
                )
                if not torch.isfinite(loss):
                    raise ValueError(
                        f"non-finite {phase} loss at epoch {epoch}, batch {batch_index}; "
                        "reduce the learning rate or inspect the data"
                    )
                scaled_loss = loss / config.gradient_accumulation
            if training:
                scaler.scale(scaled_loss).backward()
                if batch_index % config.gradient_accumulation == 0 or batch_index == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            totals["loss"] += float(loss.detach().cpu())
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            batches += 1
            if batch_index % update_interval == 0 or batch_index == len(loader):
                elapsed = time.monotonic() - started
                remaining = elapsed / batch_index * (len(loader) - batch_index)
                print(
                    f"Epoch {epoch}/{total_epochs} {phase} {batch_index}/{len(loader)} "
                    f"({batch_index / len(loader):.0%}) | loss {totals['loss'] / batches:.5f} | "
                    f"elapsed {_format_duration(elapsed)} | ETA {_format_duration(remaining)}",
                    flush=True,
                )
    return {f"{prefix}_{name}": total / batches for name, total in totals.items()}


class ExperimentLogger:
    """The intentionally simple V1 file experiment logger (no service or database required)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "checkpoints").mkdir(exist_ok=True)
        for name in ("plots", "comparisons", "report"):
            (root / name).mkdir(exist_ok=True)
        self.rows = _read_metric_rows(root / "metrics.jsonl")

    def write_static(self, config: TrainingConfig, environment: dict[str, Any]) -> None:
        (self.root / "config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
        )
        (self.root / "environment.json").write_text(_json(environment), encoding="utf-8")

    def update_environment(self, values: dict[str, Any]) -> None:
        path = self.root / "environment.json"
        environment = json.loads(path.read_text(encoding="utf-8"))
        environment.update(values)
        path.write_text(_json(environment), encoding="utf-8")

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        self._write_rows()

    def retain_through_epoch(self, epoch: int) -> None:
        """Discard a metrics row that was logged after the last durable training checkpoint."""

        retained = [
            row
            for row in self.rows
            if (isinstance(row.get("epoch"), int) and row["epoch"] <= epoch)
        ]
        if len(retained) != len(self.rows):
            self.rows = retained
            self._write_rows()

    def _write_rows(self) -> None:
        with (self.root / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        if not self.rows:
            return
        with (self.root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def write_summary(self, summary: dict[str, Any]) -> None:
        (self.root / "report" / "summary.json").write_text(_json(summary), encoding="utf-8")

    def write_plots(self) -> None:
        """Write dependency-free SVG loss/task/LR curves for quick inspection."""

        for fields, name in (
            (("train_loss", "validation_loss"), "loss.svg"),
            (
                (
                    "train_clean_loss",
                    "validation_clean_loss",
                    "train_segment_loss",
                    "validation_segment_loss",
                ),
                "task-loss.svg",
            ),
            (("learning_rate",), "learning-rate.svg"),
        ):
            available = tuple(field for field in fields if field in self.rows[0])
            if available:
                (self.root / "plots" / name).write_text(
                    _line_plot(self.rows, available), encoding="utf-8"
                )


def _read_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("metric row is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read existing training metrics: {error}") from error
    return rows


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


def _cuda_amp_dtype(device: torch.device) -> torch.dtype | None:
    """Prefer BF16 on supported CUDA cards to avoid FP16 overflow in transfer fine-tuning."""

    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _git_value(*args: str) -> str | None:
    try:
        return (
            subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
            or None
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment(
    config: TrainingConfig, device: torch.device, model: torch.nn.Module, manifest: Path
) -> dict[str, Any]:
    provenance = model_provenance(model)
    return {
        "scorerestore_version": __version__,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "amp_dtype": str(_cuda_amp_dtype(device)).replace("torch.", "")
        if device.type == "cuda"
        else None,
        "device": str(device),
        "lilypond_version": LILYPOND_VERSION,
        "dataset_id": _dataset_id(manifest),
        "dataset_manifest": str(manifest),
        "dataset_manifest_sha256": sha256_file(manifest),
        "random_seed": config.seed,
        "task": config.task,
        "model": provenance,
        "model_architecture": provenance["architecture"],
        "parameter_count": count_parameters(model),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": _git_value("status", "--porcelain") not in (None, ""),
        "docker_build_identity": os.environ.get("SCORERESTORE_BUILD_ID"),
        "started_at": _timestamp(),
    }


def _dataset_id(manifest: Path) -> str | None:
    with manifest.open(encoding="utf-8") as stream:
        first = json.loads(next(stream))
        return first.get("dataset_id")


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _format_duration(seconds: float) -> str:
    """Format elapsed and estimated remaining time without an external progress dependency."""

    total_seconds = max(0, round(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _write_prediction(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    directory: Path,
    device: torch.device,
) -> None:
    """Save one honest validation prediction as probability maps, not a later inference API."""

    batch = next(iter(loader))
    model.eval()
    with torch.no_grad():
        output = model(batch["image"].to(device))
    _save_tensor_image(batch["image"][0], directory / "input.png", invert=False)
    _save_tensor_image(batch["clean"][0], directory / "clean-target.png", invert=True)
    _save_tensor_image(
        torch.sigmoid(output.cleaning[0]).cpu(),
        directory / "cleaning-probability.png",
        invert=True,
    )
    for index, name in enumerate(("background", "staff", "notation", "text")):
        _save_tensor_image(
            torch.sigmoid(output.segmentation[0, index]).cpu(),
            directory / f"{name}-probability.png",
            invert=True,
        )


def _save_tensor_image(value: Tensor, path: Path, *, invert: bool) -> None:
    from PIL import Image

    pixels = value.squeeze().clamp(0, 1).mul(255).to(torch.uint8).numpy()
    if invert:
        pixels = 255 - pixels
    Image.fromarray(pixels, mode="L").save(path)


def _line_plot(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    width, height, padding = 640, 300, 36
    values = [float(row[field]) for row in rows for field in fields]
    low, high = min(values), max(values)
    span = high - low or 1.0
    colors = ("#1769aa", "#d32f2f", "#388e3c", "#7b1fa2")
    paths = []
    for field, color in zip(fields, colors, strict=False):
        points = []
        for index, row in enumerate(rows):
            x = padding + (width - 2 * padding) * index / max(1, len(rows) - 1)
            y = height - padding - (height - 2 * padding) * (float(row[field]) - low) / span
            points.append(f"{x:.1f},{y:.1f}")
        paths.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" '
            f'points="{" ".join(points)}"/><text x="{padding}" y="{18 + 16 * len(paths)}" '
            f'fill="{color}">{field}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
        f'<path d="M {padding} {padding} V {height - padding} H {width - padding}" '
        f'stroke="#444" fill="none"/>{"".join(paths)}</svg>\n'
    )
