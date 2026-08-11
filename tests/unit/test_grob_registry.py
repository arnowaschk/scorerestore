from __future__ import annotations

import pytest

from scorerestore.lilypond.grobs import GROB_CLASS_REGISTRY, classify_grob


@pytest.mark.parametrize("grob", ["StaffSymbol", "LedgerLineSpanner"])
def test_staff_registry_contains_only_staff_geometry(grob: str) -> None:
    assert classify_grob(grob) == ("staff", True)


@pytest.mark.parametrize(
    "grob",
    ["NoteHead", "Stem", "Beam", "BarLine", "Clef", "DynamicText", "TupletNumber"],
)
def test_conventional_music_grobs_are_notation(grob: str) -> None:
    assert classify_grob(grob) == ("notation", True)


@pytest.mark.parametrize(
    "grob",
    [
        "LyricText",
        "InstrumentName",
        "MetronomeMark",
        "Fingering",
        "TextScript",
        "ChordName",
        "BassFigure",
        "BarNumber",
    ],
)
def test_ordinary_text_grobs_are_text(grob: str) -> None:
    assert classify_grob(grob) == ("text", True)


def test_unknown_grob_defaults_to_notation() -> None:
    assert classify_grob("CustomProjectGrob") == ("notation", False)


def test_registry_has_no_cross_class_duplicates() -> None:
    assert len(GROB_CLASS_REGISTRY) == len(set(GROB_CLASS_REGISTRY))
