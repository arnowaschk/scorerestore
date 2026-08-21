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
        "compare-real-world",
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
    assert capsys.readouterr().out == "scorerestore 0.9.0\n"


def test_train_exposes_neural_backend_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["train", "--help"])

    assert exit_info.value.code == 0
    assert "custom U-Net or transfer-learning backend" in capsys.readouterr().out


def test_generate_exposes_update_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["generate", "--help"])

    assert exit_info.value.code == 0
    assert "--update" in capsys.readouterr().out


def test_infer_exposes_tiled_inference_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["infer", "--help"])

    assert exit_info.value.code == 0
    assert "tiled inference" in capsys.readouterr().out


def test_evaluation_commands_expose_their_public_interfaces(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["evaluate", "--help"])
    assert exit_info.value.code == 0
    assert "visual comparison reports" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exit_info:
        main(["benchmark", "--help"])
    assert exit_info.value.code == 0
    assert "tiled inference runtime" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exit_info:
        main(["compare-real-world", "--help"])
    assert exit_info.value.code == 0
    assert "full-resolution" in capsys.readouterr().out


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
