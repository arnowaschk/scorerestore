from __future__ import annotations

from pathlib import Path

import pytest

from scorerestore.lilypond.renderer import (
    LilyPondRenderConfig,
    LilyPondRenderError,
    detect_lilypond_version,
    render_score,
)


def test_detect_lilypond_version_parses_exact_semver(tmp_path: Path) -> None:
    binary = _fake_lilypond(tmp_path, "9.8.7")

    assert detect_lilypond_version(binary) == "9.8.7"


def test_renderer_refuses_unpinned_lilypond_before_writing_output(tmp_path: Path) -> None:
    binary = _fake_lilypond(tmp_path, "9.8.7")
    source = tmp_path / "source.ly"
    source.write_text('\\version "2.26.0"\n{ c\'4 }\n', encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(LilyPondRenderError, match="version mismatch"):
        render_score(source, output, config=LilyPondRenderConfig(lilypond_binary=binary))

    assert not output.exists()


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1])
def test_render_config_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="mask_threshold"):
        LilyPondRenderConfig(mask_threshold=threshold)


def _fake_lilypond(tmp_path: Path, version: str) -> Path:
    binary = tmp_path / "lilypond"
    binary.write_text(f'#!/bin/sh\nprintf "GNU LilyPond {version} (fake)\\n"\n', encoding="utf-8")
    binary.chmod(0o755)
    return binary
