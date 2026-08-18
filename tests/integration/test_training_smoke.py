from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scorerestore.training import load_training_config, train

PROJECT_ROOT = Path(__file__).parents[2]


def test_cpu_training_smoke_writes_checkpoint_metrics_and_provenance(tmp_path: Path) -> None:
    manifest = PROJECT_ROOT / "data/generated/scorerestore-smoke-v1/manifests/samples.jsonl"
    if not manifest.is_file():
        pytest.skip("materialized smoke dataset is not available")
    config = load_training_config(
        PROJECT_ROOT / "configs/training/smoke.yaml",
        overrides=(f"dataset_manifest={manifest}",),
    )
    result = train(config, tmp_path / "run")

    assert result.checkpoint_path.is_file()
    rows = [
        json.loads(line)
        for line in (result.output_directory / "metrics.jsonl").read_text().splitlines()
    ]
    environment = json.loads((result.output_directory / "environment.json").read_text())
    assert len(rows) == config.epochs
    assert {"train_loss", "validation_loss", "train_clean_loss", "train_segment_loss"} <= set(
        rows[0]
    )
    assert environment["task"] == "multitask"
    assert environment["dataset_manifest_sha256"]
    assert (result.output_directory / "plots/loss.svg").is_file()
    assert (result.output_directory / "comparisons/cleaning-probability.png").is_file()

    # A completed output is idempotent under --update. If interruption happens after the final
    # epoch checkpoint but before the summary, it rebuilds only the final report.
    (result.output_directory / "report/summary.json").unlink()
    resumed = train(config, result.output_directory, update=True)
    assert resumed.epochs_completed == config.epochs
    resumed_rows = [
        json.loads(line)
        for line in (result.output_directory / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(resumed_rows) == config.epochs
    assert (result.output_directory / "report/summary.json").is_file()


def test_resnet18_training_smoke_records_transfer_weight_provenance(tmp_path: Path) -> None:
    if os.environ.get("SCORERESTORE_TEST_PRETRAINED") != "1":
        pytest.skip("set SCORERESTORE_TEST_PRETRAINED=1 to fetch/check pretrained weights")
    manifest = PROJECT_ROOT / "data/generated/scorerestore-smoke-v1/manifests/samples.jsonl"
    if not manifest.is_file():
        pytest.skip("materialized smoke dataset is not available")
    config = load_training_config(
        PROJECT_ROOT / "configs/training/smoke.yaml",
        overrides=(
            f"dataset_manifest={manifest}",
            "model.backend=resnet18",
            "model.base_channels=4",
            "model.pretrained=true",
            "training.epochs=1",
        ),
    )
    result = train(config, tmp_path / "run")

    environment = json.loads((result.output_directory / "environment.json").read_text())
    assert result.checkpoint_path.is_file()
    assert environment["model"]["weights"] == "ResNet18_Weights.IMAGENET1K_V1"
    assert environment["model"]["weights_url"].startswith("https://download.pytorch.org/")
    assert environment["model"]["grayscale_adaptation"].startswith("conv1 RGB kernels averaged")
    assert "statistics frozen" in environment["model"]["batch_norm"]
