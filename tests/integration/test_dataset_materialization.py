from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from scorerestore.cli import main
from scorerestore.dataset import MaterializedDataset, validate_dataset_manifest
from scorerestore.dataset.manifest import DatasetManifestError, sha256_file
from scorerestore.lilypond.constants import LILYPOND_VERSION
from scorerestore.lilypond.renderer import LilyPondRenderError, detect_lilypond_version

PROJECT_ROOT = Path(__file__).parents[2]
SMOKE_CONFIG = PROJECT_ROOT / "configs/dataset/smoke.yaml"


@pytest.fixture(scope="session")
def dataset_lilypond_binary() -> Path:
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


def test_materialize_validate_load_and_exactly_reproduce_smoke_dataset(
    tmp_path: Path,
    dataset_lilypond_binary: Path,
) -> None:
    output_root = tmp_path / "generated"
    assert (
        main(
            [
                "generate",
                "-c",
                str(SMOKE_CONFIG),
                "--output-root",
                str(output_root),
                "--lilypond",
                str(dataset_lilypond_binary),
            ]
        )
        == 0
    )
    dataset_root = output_root / "scorerestore-smoke-v1"
    manifest_path = dataset_root / "manifests/samples.jsonl"
    report = validate_dataset_manifest(manifest_path)

    assert dataset_root.stat().st_mode & 0o777 == 0o755
    assert manifest_path.stat().st_mode & 0o777 == 0o644
    assert len(report.records) == 2
    assert set(report.source_splits.values()) == {"train"}
    assert not (
        {source for source, split in report.source_splits.items() if split == "train"}
        & {source for source, split in report.source_splits.items() if split == "test"}
    )
    sample_ids = {record["sample_id"] for record in report.records}
    assert len(sample_ids) == 2
    assert all(record["source_license_status"] == "public_domain" for record in report.records)

    dataset = MaterializedDataset(manifest_path)
    sample = dataset[0]
    assert sample.image.size == sample.clean.size
    assert set(sample.masks) == {"background", "staff", "notation", "text"}
    assert all(mask.size == sample.clean.size for mask in sample.masks.values())
    assert sample.image.tobytes() != sample.clean.tobytes()
    cleaning_sample = next(dataset.iter_cleaning())
    assert cleaning_sample.image.size == cleaning_sample.clean.size
    assert cleaning_sample.record["sample_id"] == sample.record["sample_id"]

    reproduced_path = tmp_path / "reproduced.png"
    sample_id = report.records[0]["sample_id"]
    assert (
        main(
            [
                "dataset",
                "reproduce",
                sample_id,
                "--data-root",
                str(output_root),
                "--dataset-id",
                "scorerestore-smoke-v1",
                "--source-manifest",
                str(PROJECT_ROOT / "assets/scores/manifest.yaml"),
                "--lilypond",
                str(dataset_lilypond_binary),
                "-o",
                str(reproduced_path),
            ]
        )
        == 0
    )
    assert sha256_file(reproduced_path) == report.records[0]["hashes"]["input"]

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    records[1]["split"] = "test"
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    with pytest.raises(DatasetManifestError, match="spans splits"):
        validate_dataset_manifest(manifest_path)
