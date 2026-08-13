from __future__ import annotations

from pathlib import Path

import pytest

from scorerestore.cli import main


def test_root_help_lists_public_v1_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "ScoreRestore V1 proof of concept" in output
    assert "ScoreRestore is not an OMR system" in output
    for command in (
        "generate",
        "train",
        "evaluate",
        "infer",
        "baseline",
        "dataset",
        "inspect",
        "degrade",
    ):
        assert command in output


def test_version_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == "scorerestore 0.1.0\n"


def test_train_exposes_neural_backend_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["train", "--help"])

    assert exit_info.value.code == 0
    assert "custom U-Net or transfer-learning backend" in capsys.readouterr().out


def test_infer_exposes_tiled_inference_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["infer", "--help"])

    assert exit_info.value.code == 0
    assert "tiled inference" in capsys.readouterr().out


def test_later_milestone_command_fails_explicitly(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["evaluate"]) == 2
    assert "reserved for a later" in capsys.readouterr().err


def test_dataset_reproduce_requires_sample_id() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["dataset", "reproduce"])

    assert exit_info.value.code == 2


def test_dataset_reproduce_reports_missing_sample(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["dataset", "reproduce", "sample-1", "--data-root", str(tmp_path)]) == 1
    assert "was not found" in capsys.readouterr().err


def test_inspect_help_lists_provenance(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["inspect", "--help"])

    assert exit_info.value.code == 0
    assert "provenance" in capsys.readouterr().out
