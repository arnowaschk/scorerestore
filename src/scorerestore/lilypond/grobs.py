"""Centralized V1 LilyPond grob-to-semantic-class registry.

Staff lines and ledger lines are the only ``staff`` grobs. Ordinary textual content is ``text``.
All other known musical symbols are ``notation``. Unknown printable grobs deliberately default to
``notation`` and are reported by the renderer so conventional external files remain usable while a
strict diagnostic mode can flag registry gaps.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal, TypeAlias

SemanticForegroundClass: TypeAlias = Literal["staff", "notation", "text"]

_STAFF_GROBS = {
    "LedgerLineSpanner",
    "StaffSymbol",
}

_TEXT_GROBS = {
    "BalloonTextItem",
    "BarNumber",
    "BassFigure",
    "BassFigureAlignment",
    "BassFigureAlignmentPositioning",
    "BassFigureBracket",
    "ChordName",
    "CombineTextScript",
    "Fingering",
    "FootnoteItem",
    "HorizontalBracketText",
    "InstrumentName",
    "InstrumentSwitch",
    "JumpScript",
    "LyricExtender",
    "LyricHyphen",
    "LyricSpace",
    "LyricText",
    "MetronomeMark",
    "MeasureCounter",
    "NoteName",
    "RehearsalMark",
    "SectionLabel",
    "StringNumber",
    "StrokeFinger",
    "TextScript",
    "TextSpanner",
}

_NOTATION_GROBS = {
    "Accidental",
    "AccidentalCautionary",
    "AccidentalPlacement",
    "Ambitus",
    "AmbitusAccidental",
    "AmbitusLine",
    "AmbitusNoteHead",
    "Arpeggio",
    "BarLine",
    "Beam",
    "BendAfter",
    "BreakAlignGroup",
    "BreakAlignment",
    "BreathingSign",
    "Clef",
    "ClefModifier",
    "ClusterSpanner",
    "CueClef",
    "CueEndClef",
    "DotColumn",
    "Dots",
    "DoublePercentRepeat",
    "DoublePercentRepeatCounter",
    "DynamicLineSpanner",
    "DynamicText",
    "DynamicTextSpanner",
    "Episema",
    "Flag",
    "Glissando",
    "GraceSpacing",
    "GridLine",
    "GridPoint",
    "Hairpin",
    "HorizontalBracket",
    "KeyCancellation",
    "KeySignature",
    "LaissezVibrerTie",
    "LaissezVibrerTieColumn",
    "LeftEdge",
    "MeasureGrouping",
    "MeasureSpanner",
    "MultiMeasureRest",
    "MultiMeasureRestNumber",
    "MultiMeasureRestScript",
    "NonMusicalPaperColumn",
    "NoteCollision",
    "NoteColumn",
    "NoteHead",
    "NoteSpacing",
    "OttavaBracket",
    "ParenthesesItem",
    "PaperColumn",
    "PercentRepeat",
    "PercentRepeatCounter",
    "RepeatSlash",
    "Rest",
    "RestCollision",
    "Script",
    "ScriptColumn",
    "Slur",
    "SostenutoPedal",
    "SostenutoPedalLineSpanner",
    "SpacingSpanner",
    "SpanBar",
    "SpanBarStub",
    "StaffGrouper",
    "StaffSpacing",
    "Stem",
    "StemStub",
    "StemTremolo",
    "SustainPedal",
    "SustainPedalLineSpanner",
    "SystemStartBar",
    "SystemStartBrace",
    "SystemStartBracket",
    "SystemStartSquare",
    "TabNoteHead",
    "Tie",
    "TieColumn",
    "TimeSignature",
    "TrillPitchAccidental",
    "TrillPitchGroup",
    "TrillPitchHead",
    "TrillSpanner",
    "TupletBracket",
    "TupletNumber",
    "UnaCordaPedal",
    "UnaCordaPedalLineSpanner",
    "VaticanaLigature",
    "VerticalAlignment",
    "VerticalAxisGroup",
    "VoltaBracket",
    "VoltaBracketSpanner",
}

GROB_CLASS_REGISTRY = MappingProxyType(
    {
        **{name: "staff" for name in sorted(_STAFF_GROBS)},
        **{name: "notation" for name in sorted(_NOTATION_GROBS)},
        **{name: "text" for name in sorted(_TEXT_GROBS)},
    }
)


def classify_grob(name: str) -> tuple[SemanticForegroundClass, bool]:
    """Return a grob's class and whether it was explicitly registered."""

    classification = GROB_CLASS_REGISTRY.get(name)
    if classification is None:
        return "notation", False
    return classification, True
