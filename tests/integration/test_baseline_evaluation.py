from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import pytest
from PIL import Image

from scorerestore.cli import main
from scorerestore.lilypond.constants import LILYPOND_VERSION
from scorerestore.lilypond.renderer import LilyPondRenderError, detect_lilypond_version

PROJECT_ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="session")
def baseline_lilypond_binary() -> Path:
    configured = os.environ.get("SCORERESTORE_TEST_LILYPOND")
    candidate = Path(configured) if configured else Path(shutil.which("lilypond") or "")
    if not candidate.is_file():
        pytest.skip("LilyPond is not installed")
    try:
        version = detect_lilypond_version(candidate)
    except LilyPondRenderError as error:
        pytest.skip(str(error))
    if version != LILYPOND_VERSION:
        pytest.skip(f"integration tests require LilyPond {LILYPOND_VERSION}, found {version}")
    return candidate


def test_baseline_cli_processes_dataset_writes_metrics_and_images(
    tmp_path: Path,
    baseline_lilypond_binary: Path,
) -> None:
    generated = tmp_path / "generated"
    assert (
        main(
            [
                "generate",
                "-c",
                str(PROJECT_ROOT / "configs/dataset/smoke.yaml"),
                "--output-root",
                str(generated),
                "--lilypond",
                str(baseline_lilypond_binary),
            ]
        )
        == 0
    )
    manifest = generated / "scorerestore-smoke-v1/manifests/samples.jsonl"
    output = tmp_path / "baseline"
    assert (
        main(
            [
                "baseline",
                str(manifest),
                "-c",
                str(PROJECT_ROOT / "configs/baseline.yaml"),
                "-o",
                str(output),
                "--split",
                "train",
            ]
        )
        == 0
    )

    metric_rows = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as source:
        csv_rows = list(csv.DictReader(source))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest_records = {
        record["sample_id"]: record
        for record in (
            json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
        )
    }

    expected_variants = {
        "otsu",
        "adaptive",
        "otsu_morphology",
        "adaptive_morphology",
    }
    assert len(metric_rows) == len(csv_rows) == 8
    assert {row["variant"] for row in metric_rows} == expected_variants
    assert summary["sample_count"] == 2
    assert summary["result_count"] == 8
    assert set(summary["variants"]) == expected_variants
    for variant in expected_variants:
        assert summary["variants"][variant]["splits"]["train"]["sample_count"] == 2
        assert "accuracy" not in summary["variants"][variant]["splits"]["train"]
        assert {row["sample_id"] for row in metric_rows if row["variant"] == variant} == set(
            manifest_records
        )
    assert summary["metric_notes"]["challenge"].startswith("challenge is reported separately")
    for row in metric_rows:
        result_path = output / row["result_path"]
        assert result_path.is_file()
        with Image.open(result_path) as result_image:
            dimensions = manifest_records[row["sample_id"]]["dimensions"]
            assert result_image.size == (dimensions["width"], dimensions["height"])
    assert (output / "config.yaml").is_file()
    assert (output / "environment.json").is_file()
