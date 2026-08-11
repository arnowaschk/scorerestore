from __future__ import annotations

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


def test_later_milestone_command_fails_explicitly(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["train"]) == 2
    assert "reserved for a later" in capsys.readouterr().err


def test_dataset_reproduce_requires_sample_id() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["dataset", "reproduce"])

    assert exit_info.value.code == 2


def test_dataset_reproduce_is_an_explicit_placeholder(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["dataset", "reproduce", "sample-1"]) == 2
    assert "dataset reproduce" in capsys.readouterr().err


def test_inspect_help_lists_provenance(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["inspect", "--help"])

    assert exit_info.value.code == 0
    assert "provenance" in capsys.readouterr().out
