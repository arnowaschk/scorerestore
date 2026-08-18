from __future__ import annotations

from pathlib import Path

import pytest

from scorerestore.lilypond.renderer import (
    LilyPondRenderConfig,
    LilyPondRenderError,
    _convert_source,
    _repair_legacy_unbraced_new_contexts,
    _repair_lilypond_compatibility,
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


def test_renderer_repairs_legacy_unbraced_named_context_in_temporary_copy() -> None:
    source = r"""voice = \relative c' {
  \new Voice = "legacy" % old LilyPond shorthand
  \override Fingering.staff-padding = #'()
  c4
}
unchanged = { "literal { brace }" % comment { brace }
  \new Voice = "modern" { d4 }
}
"""

    repaired = _repair_legacy_unbraced_new_contexts(source)

    assert '\\new Voice = "legacy" { % old LilyPond shorthand' in repaired
    assert "  c4\n  }\n}" in repaired
    assert '\\new Voice = "modern" { d4 }' in repaired
    assert '"literal { brace }"' in repaired


def test_renderer_repairs_legacy_layout_context_in_temporary_copy() -> None:
    source = r"""\score { c'4 }
\layout {
  \context Staff
  \override TextScript.font-shape = #'italic
}
"""

    repaired = _repair_lilypond_compatibility(source)

    assert "\\layout {\n  \\context {\n    \\Staff" in repaired
    assert "  \\override TextScript.font-shape = #'italic\n  }\n}" in repaired


def test_renderer_converts_local_include_tree_without_editing_sources(tmp_path: Path) -> None:
    source_directory = tmp_path / "score"
    included_directory = source_directory / "parts"
    included_directory.mkdir(parents=True)
    source = source_directory / "score.ly"
    included = included_directory / "voice.ly"
    source.write_text('\\version "2.16.0"\n\\include "parts/voice.ly"\n', encoding="utf-8")
    included.write_text(
        "\\layout {\n  \\context Staff\n  \\override TextScript.font-shape = #'italic\n}\n",
        encoding="utf-8",
    )
    original_included = included.read_bytes()
    lilypond = _fake_lilypond(tmp_path, "2.26.0")
    converter = tmp_path / "convert-ly"
    converter.write_text('#!/bin/sh\nshift\ncat "$1"\n', encoding="utf-8")
    converter.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()

    converted = _convert_source(source, work, LilyPondRenderConfig(lilypond_binary=lilypond))

    assert converted == work / "converted/score.ly"
    converted_include = work / "converted/parts/voice.ly"
    assert converted_include.is_file()
    assert "\\context {\n    \\Staff" in converted_include.read_text(encoding="utf-8")
    assert included.read_bytes() == original_included


def _fake_lilypond(tmp_path: Path, version: str) -> Path:
    binary = tmp_path / "lilypond"
    binary.write_text(f'#!/bin/sh\nprintf "GNU LilyPond {version} (fake)\\n"\n', encoding="utf-8")
    binary.chmod(0o755)
    return binary
