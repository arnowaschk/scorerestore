"""Pinned V1 LilyPond and mask-rendering constants."""

from typing import Final

LILYPOND_VERSION: Final = "2.26.0"
LILYPOND_LINUX_X86_64_SHA256: Final = (
    "cd8a097a9f52cb2b9f4e7914774786f203f4fc61fcd299afcbb63c23fa5c6b24"
)
DEFAULT_DPI: Final = 300
DEFAULT_MASK_THRESHOLD: Final = 0.5
SEMANTIC_FOREGROUND_CLASSES: Final = ("staff", "notation", "text")
