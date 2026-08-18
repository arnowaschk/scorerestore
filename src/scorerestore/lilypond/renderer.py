"""Aligned multi-pass LilyPond rendering and semantic-mask QA."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageChops, ImageDraw, ImageOps

from scorerestore import __version__
from scorerestore.lilypond.constants import (
    DEFAULT_DPI,
    DEFAULT_MASK_THRESHOLD,
    LILYPOND_VERSION,
    SEMANTIC_FOREGROUND_CLASSES,
)
from scorerestore.lilypond.grobs import GROB_CLASS_REGISTRY
from scorerestore.lilypond.masks import (
    MaskQAError,
    derive_background,
    foreground_pixel_count,
    threshold_coverage,
    validate_semantic_masks,
)

_VERSION_PATTERN = re.compile(r"GNU LilyPond ([0-9]+\.[0-9]+\.[0-9]+)")
_UNKNOWN_GROB_PATTERN = re.compile(r"SCORERESTORE_UNKNOWN_GROB:([A-Za-z0-9_-]+)")
_PAGE_NUMBER_PATTERN = re.compile(r"-page([0-9]+)\.png$")
_LEGACY_UNBRACED_NEW_CONTEXT_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)(?P<context_command>\\new[ \t]+"
    r'[A-Za-z][A-Za-z0-9_-]*[ \t]*=[ \t]*"(?:[^"\\]|\\.)*")'
    r"(?P<trailing>[ \t]*(?:%[^\n]*)?)$",
    re.MULTILINE,
)
_LEGACY_LAYOUT_CONTEXT_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)\\context[ \t]+(?P<name>[A-Za-z][A-Za-z0-9_-]*)"
    r"(?P<trailing>[ \t]*(?:%[^\n]*)?)$",
    re.MULTILINE,
)
_LAYOUT_OPEN_PATTERN = re.compile(r"\\layout[ \t\r\n]*\{")
_INCLUDE_PATTERN = re.compile(r'\\include\s+"([^"\\]+)"')


class LilyPondRenderError(RuntimeError):
    """Raised when rendering, version validation, or mask QA fails."""


@dataclass(frozen=True, slots=True)
class LilyPondLayoutConfig:
    """Modest deterministic V1 engraving variation applied to every render pass."""

    staff_size: float
    paper_format: Literal["a4", "letter"]
    orientation: Literal["portrait", "landscape"]
    top_margin_mm: float
    bottom_margin_mm: float
    left_margin_mm: float
    right_margin_mm: float

    def __post_init__(self) -> None:
        if not 8.0 <= self.staff_size <= 60.0:
            raise ValueError("staff_size must be within the modest V1 range [8, 60]")
        if self.paper_format not in {"a4", "letter"}:
            raise ValueError("paper_format must be 'a4' or 'letter'")
        if self.orientation not in {"portrait", "landscape"}:
            raise ValueError("orientation must be 'portrait' or 'landscape'")
        margins = (
            self.top_margin_mm,
            self.bottom_margin_mm,
            self.left_margin_mm,
            self.right_margin_mm,
        )
        if any(not 3.0 <= margin <= 30.0 for margin in margins):
            raise ValueError("layout margins must be within the realistic V1 range [3, 30] mm")

    def to_dict(self) -> dict[str, object]:
        """Return manifest-ready layout parameters."""

        return {
            "staff_size": self.staff_size,
            "paper_format": self.paper_format,
            "orientation": self.orientation,
            "margins_mm": {
                "top": self.top_margin_mm,
                "bottom": self.bottom_margin_mm,
                "left": self.left_margin_mm,
                "right": self.right_margin_mm,
            },
        }


@dataclass(frozen=True, slots=True)
class LilyPondRenderConfig:
    """Deterministic V1 render settings."""

    lilypond_binary: str | Path = "lilypond"
    dpi: int = DEFAULT_DPI
    mask_threshold: float = DEFAULT_MASK_THRESHOLD
    strict_unknown_grobs: bool = False
    expected_nonempty: tuple[str, ...] = ("staff", "notation")
    timeout_seconds: int = 180
    layout: LilyPondLayoutConfig | None = None

    def __post_init__(self) -> None:
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        if not 0.0 < self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be greater than 0.0 and at most 1.0")
        invalid_classes = set(self.expected_nonempty) - set(SEMANTIC_FOREGROUND_CLASSES)
        if invalid_classes:
            names = ", ".join(sorted(invalid_classes))
            raise ValueError(f"unknown expected_nonempty classes: {names}")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class RenderedMaskPage:
    """Materialized output paths and dimensions for one score page."""

    page: int
    width: int
    height: int
    directory: Path
    pristine_path: Path
    mask_paths: dict[str, Path]
    qa_panel_path: Path


@dataclass(frozen=True, slots=True)
class LilyPondRenderResult:
    """Result of a successful, QA-validated score render."""

    source_path: Path
    source_sha256: str
    source_hash_verified: bool
    output_directory: Path
    metadata_path: Path
    lilypond_version: str
    unknown_grobs: tuple[str, ...]
    pages: tuple[RenderedMaskPage, ...]


@dataclass(frozen=True, slots=True)
class LilyPondPreflightResult:
    """Successful compatibility preflight for one source, without page materialization."""

    source_path: Path
    source_sha256: str
    lilypond_version: str
    unknown_grobs: tuple[str, ...]


def detect_lilypond_version(binary: str | Path = "lilypond") -> str:
    """Return the exact LilyPond semantic version reported by *binary*."""

    try:
        completed = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LilyPondRenderError(f"cannot execute LilyPond binary {binary!s}: {error}") from error
    output = f"{completed.stdout}\n{completed.stderr}"
    match = _VERSION_PATTERN.search(output)
    if completed.returncode != 0 or match is None:
        raise LilyPondRenderError(
            f"cannot determine LilyPond version from {binary!s}: {output.strip()}"
        )
    return match.group(1)


def render_score(
    source: str | Path,
    output_directory: str | Path,
    *,
    config: LilyPondRenderConfig | None = None,
    expected_source_sha256: str | None = None,
) -> LilyPondRenderResult:
    """Render pristine and independent semantic masks for an ordinary LilyPond file.

    The source is never edited. A converted temporary copy and a generated Scheme/layout wrapper
    apply the same geometry to every pass. Semantic PNG masks encode ``255 = foreground`` and
    ``0 = background``; pristine grayscale retains ``0 = black`` and ``255 = white``.
    """

    render_config = config or LilyPondRenderConfig()
    source_path = Path(source).resolve()
    output_path = Path(output_directory).resolve()
    if not source_path.is_file():
        raise LilyPondRenderError(f"LilyPond source does not exist: {source_path}")
    if source_path.suffix != ".ly":
        raise LilyPondRenderError(f"LilyPond source must use the .ly suffix: {source_path}")
    if output_path.exists():
        raise LilyPondRenderError(f"output directory already exists: {output_path}")

    lilypond_version = detect_lilypond_version(render_config.lilypond_binary)
    if lilypond_version != LILYPOND_VERSION:
        raise LilyPondRenderError(
            f"LilyPond version mismatch: expected {LILYPOND_VERSION}, got {lilypond_version}"
        )

    source_sha256 = _sha256_file(source_path)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise LilyPondRenderError(
            f"source SHA-256 mismatch: expected {expected_source_sha256}, got {source_sha256}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_path.name}-", dir=output_path.parent))
    try:
        work_directory = temporary_root / "work"
        result_directory = temporary_root / "result"
        work_directory.mkdir()
        result_directory.mkdir()
        converted_source = _convert_source(source_path, work_directory, render_config)
        pass_pages, unknown_grobs = _render_passes(
            converted_source,
            source_path.parent,
            work_directory,
            render_config,
        )
        if render_config.strict_unknown_grobs and unknown_grobs:
            names = ", ".join(sorted(unknown_grobs))
            raise LilyPondRenderError(f"strict unknown-grob validation failed: {names}")

        rendered_pages, page_metadata = _materialize_pages(
            pass_pages,
            result_directory,
            render_config,
        )
        metadata = {
            "scorerestore_version": __version__,
            "generator": "scorerestore.lilypond",
            "lilypond_version": lilypond_version,
            "required_lilypond_version": LILYPOND_VERSION,
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "source_hash_verified": expected_source_sha256 is not None,
            "dpi": render_config.dpi,
            "mask_threshold": render_config.mask_threshold,
            "mask_encoding": {"background": 0, "foreground": 255},
            "unknown_grobs": sorted(unknown_grobs),
            "strict_unknown_grobs": render_config.strict_unknown_grobs,
            "grob_registry_entries": len(GROB_CLASS_REGISTRY),
            "render_parameters": (
                render_config.layout.to_dict() if render_config.layout is not None else None
            ),
            "pages": page_metadata,
        }
        metadata_path = result_directory / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result_directory.replace(output_path)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    shutil.rmtree(temporary_root, ignore_errors=True)

    final_pages = tuple(_relocate_page(page, output_path) for page in rendered_pages)
    return LilyPondRenderResult(
        source_path=source_path,
        source_sha256=source_sha256,
        source_hash_verified=expected_source_sha256 is not None,
        output_directory=output_path,
        metadata_path=output_path / "metadata.json",
        lilypond_version=lilypond_version,
        unknown_grobs=tuple(sorted(unknown_grobs)),
        pages=final_pages,
    )


def preflight_score(
    source: str | Path,
    *,
    config: LilyPondRenderConfig | None = None,
    expected_source_sha256: str | None = None,
) -> LilyPondPreflightResult:
    """Check that one source converts and engraves with the configured LilyPond.

    This executes the same conversion and semantic-grob wrapper as ``render_score``
    but suppresses page output.  It is deliberately intended for a fast, complete
    source-corpus compatibility check before expensive dataset generation.
    """

    render_config = config or LilyPondRenderConfig()
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise LilyPondRenderError(f"LilyPond source does not exist: {source_path}")
    if source_path.suffix != ".ly":
        raise LilyPondRenderError(f"LilyPond source must use the .ly suffix: {source_path}")

    lilypond_version = detect_lilypond_version(render_config.lilypond_binary)
    if lilypond_version != LILYPOND_VERSION:
        raise LilyPondRenderError(
            f"LilyPond version mismatch: expected {LILYPOND_VERSION}, got {lilypond_version}"
        )
    source_sha256 = _sha256_file(source_path)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise LilyPondRenderError(
            f"source SHA-256 mismatch: expected {expected_source_sha256}, got {source_sha256}"
        )

    with tempfile.TemporaryDirectory(prefix="scorerestore-lilypond-preflight-") as temporary:
        work_directory = Path(temporary) / "work"
        work_directory.mkdir()
        converted_source = _convert_source(source_path, work_directory, render_config)
        wrapper = work_directory / "preflight.ly"
        wrapper.write_text(
            _wrapper_source("pristine", converted_source, render_config.layout), encoding="utf-8"
        )
        output = _run_lilypond(
            wrapper,
            work_directory / "preflight",
            source_path.parent,
            render_config,
            no_print_pages=True,
        )
    unknown_grobs = set(_UNKNOWN_GROB_PATTERN.findall(output))
    if render_config.strict_unknown_grobs and unknown_grobs:
        names = ", ".join(sorted(unknown_grobs))
        raise LilyPondRenderError(f"strict unknown-grob validation failed: {names}")
    return LilyPondPreflightResult(
        source_path=source_path,
        source_sha256=source_sha256,
        lilypond_version=lilypond_version,
        unknown_grobs=tuple(sorted(unknown_grobs)),
    )


def _convert_source(
    source_path: Path,
    work_directory: Path,
    config: LilyPondRenderConfig,
) -> Path:
    converted_root = work_directory / "converted"
    converted_root.mkdir()
    convert_binary = _convert_binary(config)
    source_root = source_path.parent.resolve()
    pending = [source_path]
    converted: set[Path] = set()
    while pending:
        current_source = pending.pop()
        if current_source in converted:
            continue
        converted.add(current_source)
        source_text = current_source.read_text(encoding="utf-8")
        for include in _INCLUDE_PATTERN.findall(source_text):
            included = (current_source.parent / include).resolve()
            if included.is_file() and _is_within(included, source_root):
                pending.append(included)

        converted_text = _convert_file(current_source, convert_binary, config)
        destination = converted_root / current_source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_repair_lilypond_compatibility(converted_text), encoding="utf-8")
    return converted_root / source_path.name


def _convert_binary(config: LilyPondRenderConfig) -> Path:
    lilypond_binary = Path(config.lilypond_binary)
    convert_binary = lilypond_binary.with_name("convert-ly")
    if convert_binary.exists():
        return convert_binary
    located = shutil.which("convert-ly")
    if located is None:
        raise LilyPondRenderError("convert-ly was not found beside LilyPond or on PATH")
    return Path(located)


def _convert_file(
    source_path: Path,
    convert_binary: Path,
    config: LilyPondRenderConfig,
) -> str:
    try:
        completed = subprocess.run(
            [str(convert_binary), "--current-version", str(source_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            cwd=source_path.parent,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LilyPondRenderError(f"convert-ly failed for {source_path}: {error}") from error
    if completed.returncode != 0 or not completed.stdout.strip():
        detail = completed.stderr.strip() or "convert-ly produced no source"
        raise LilyPondRenderError(f"convert-ly failed for {source_path}: {detail}")
    return completed.stdout


def _is_within(path: Path, directory: Path) -> bool:
    """Return whether a resolved included source is contained in its score tree."""

    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _repair_lilypond_compatibility(source: str) -> str:
    """Apply narrow structural repairs absent from ``convert-ly`` output."""

    return _repair_legacy_layout_contexts(_repair_legacy_unbraced_new_contexts(source))


def _repair_legacy_unbraced_new_contexts(source: str) -> str:
    """Give legacy ``\\new Context = \"id\"`` declarations an explicit music body.

    LilyPond 2.26 requires a music expression immediately after a named ``\\new``
    context.  Some older Mutopia sources rely on the pre-2.19 shorthand where the
    rest of the enclosing music block is implicitly the context body.  ``convert-ly``
    updates many older syntaxes but deliberately leaves this structural shorthand
    unchanged.  Repair it only in our temporary converted copy, preserving source
    files and their provenance hashes byte-for-byte.
    """

    matches = tuple(_LEGACY_UNBRACED_NEW_CONTEXT_PATTERN.finditer(source))
    if not matches:
        return source

    brace_pairs = _lilypond_brace_pairs(source)
    insertions: list[tuple[int, str]] = []
    for match in matches:
        enclosing = _enclosing_brace(match.start(), brace_pairs)
        if enclosing is None:
            raise LilyPondRenderError(
                "cannot repair legacy unbraced \\new context outside a music block"
            )
        _, closing_brace = enclosing
        close_line_start = source.rfind("\n", 0, closing_brace) + 1
        if source[close_line_start:closing_brace].strip():
            raise LilyPondRenderError(
                "cannot repair legacy unbraced \\new context with an inline closing brace"
            )
        insertions.append((match.start("trailing"), " {"))
        insertions.append((close_line_start, f"{match.group('indent')}}}\n"))

    repaired = source
    for position, text in sorted(insertions, key=lambda item: item[0], reverse=True):
        repaired = f"{repaired[:position]}{text}{repaired[position:]}"
    return repaired


def _repair_legacy_layout_contexts(source: str) -> str:
    """Turn legacy ``\\context Staff`` layout declarations into modern blocks.

    In a ``\\layout`` block, LilyPond 2.26 requires ``\\context { \\Staff ... }``.
    Older sources used ``\\context Staff`` followed by overrides until the end of
    the layout block.  This function adapts that exact shorthand in the temporary
    converted source only.
    """

    brace_pairs = _lilypond_brace_pairs(source)
    layout_blocks: dict[int, int] = {}
    for match in _LAYOUT_OPEN_PATTERN.finditer(source):
        opening = match.end() - 1
        closing = brace_pairs.get(opening)
        if closing is not None:
            layout_blocks[opening] = closing
    if not layout_blocks:
        return source

    replacements: list[tuple[int, int, str]] = []
    for match in _LEGACY_LAYOUT_CONTEXT_PATTERN.finditer(source):
        enclosing = _enclosing_brace(match.start(), layout_blocks)
        if enclosing is None:
            continue
        _, closing_brace = enclosing
        close_line_start = source.rfind("\n", 0, closing_brace) + 1
        if source[close_line_start:closing_brace].strip():
            raise LilyPondRenderError(
                "cannot repair legacy layout context with an inline closing brace"
            )
        replacement = (
            f"{match.group('indent')}\\context {{{match.group('trailing')}\n"
            f"{match.group('indent')}  \\{match.group('name')}"
        )
        replacements.append((match.start(), match.end(), replacement))
        replacements.append((close_line_start, close_line_start, f"{match.group('indent')}}}\n"))

    repaired = source
    for start, end, text in sorted(replacements, key=lambda item: item[0], reverse=True):
        repaired = f"{repaired[:start]}{text}{repaired[end:]}"
    return repaired


def _lilypond_brace_pairs(source: str) -> dict[int, int]:
    """Return matching unescaped brace positions, ignoring strings and comments."""

    pairs: dict[int, int] = {}
    openings: list[int] = []
    in_string = False
    escaped = False
    in_comment = False
    for position, character in enumerate(source):
        if in_comment:
            if character == "\n":
                in_comment = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "%":
            in_comment = True
        elif character == '"':
            in_string = True
        elif character == "{":
            openings.append(position)
        elif character == "}" and openings:
            pairs[openings.pop()] = position
    return pairs


def _enclosing_brace(position: int, brace_pairs: dict[int, int]) -> tuple[int, int] | None:
    """Find the innermost matched brace pair enclosing *position*."""

    candidates = [
        (opening, closing)
        for opening, closing in brace_pairs.items()
        if opening < position < closing
    ]
    return max(candidates, default=None, key=lambda pair: pair[0])


def _render_passes(
    converted_source: Path,
    source_include_directory: Path,
    work_directory: Path,
    config: LilyPondRenderConfig,
) -> tuple[dict[str, list[Path]], set[str]]:
    wrappers = work_directory / "wrappers"
    raw = work_directory / "raw"
    wrappers.mkdir()
    raw.mkdir()
    pass_pages: dict[str, list[Path]] = {}
    unknown_grobs: set[str] = set()
    for target in ("pristine", "staff", "notation", "text", "none"):
        wrapper = wrappers / f"{target}.ly"
        wrapper.write_text(
            _wrapper_source(target, converted_source, config.layout), encoding="utf-8"
        )
        output_prefix = raw / target
        stderr = _run_lilypond(
            wrapper,
            output_prefix,
            source_include_directory,
            config,
        )
        unknown_grobs.update(_UNKNOWN_GROB_PATTERN.findall(stderr))
        pass_pages[target] = _find_rendered_pages(output_prefix)

    page_counts = {target: len(paths) for target, paths in pass_pages.items()}
    if len(set(page_counts.values())) != 1:
        raise LilyPondRenderError(f"render pass page counts differ: {page_counts}")
    return pass_pages, unknown_grobs


def _wrapper_source(
    target: str,
    converted_source: Path,
    layout: LilyPondLayoutConfig | None,
) -> str:
    registry = "\n".join(
        f"    ({name} . {classification})"
        for name, classification in sorted(GROB_CLASS_REGISTRY.items())
    )
    include_path = str(converted_source).replace("\\", "/").replace('"', '\\"')
    staff_size = f"#(set-global-staff-size {layout.staff_size})\n" if layout else ""
    paper_override = _paper_override(layout) if layout else ""
    return f'''\\version "{LILYPOND_VERSION}"

{staff_size}#(define scorerestore-target '{target})
#(define scorerestore-grob-registry
  '(
{registry}
  ))
#(define scorerestore-seen-unknown '())

#(define (scorerestore-grob-name grob)
   (assq-ref (ly:grob-property grob 'meta '()) 'name))

#(define (scorerestore-printable? grob)
   (not (eq? (ly:grob-property-data grob 'stencil) #f)))

#(define (scorerestore-classify grob)
   (let* ((name (scorerestore-grob-name grob))
          (entry (and name (assq name scorerestore-grob-registry))))
     (if entry
         (cdr entry)
         (begin
           (when (and name
                      (scorerestore-printable? grob)
                      (not (memq name scorerestore-seen-unknown)))
             (set! scorerestore-seen-unknown (cons name scorerestore-seen-unknown))
             (ly:message (format #f "SCORERESTORE_UNKNOWN_GROB:~a" name)))
           'notation))))

#(define (scorerestore-mask-engraver context)
   (make-engraver
    (acknowledgers
     ((grob-interface engraver grob source-engraver)
      (let ((classification (scorerestore-classify grob)))
        (when (or (eq? scorerestore-target 'none)
                  (and (not (eq? scorerestore-target 'pristine))
                       (not (eq? scorerestore-target classification))))
          (ly:grob-set-property! grob 'transparent #t)))))))

\\layout {{
  \\context {{
    \\Score
    \\consists #scorerestore-mask-engraver
  }}
}}

\\include "{include_path}"
{paper_override}
'''


def _paper_override(layout: LilyPondLayoutConfig) -> str:
    paper_name = (
        f"{layout.paper_format}landscape"
        if layout.orientation == "landscape"
        else layout.paper_format
    )
    return f'''\n\\paper {{
  #(set-paper-size "{paper_name}")
  top-margin = {layout.top_margin_mm}\\mm
  bottom-margin = {layout.bottom_margin_mm}\\mm
  left-margin = {layout.left_margin_mm}\\mm
  right-margin = {layout.right_margin_mm}\\mm
}}
'''


def _run_lilypond(
    wrapper: Path,
    output_prefix: Path,
    source_include_directory: Path,
    config: LilyPondRenderConfig,
    *,
    no_print_pages: bool = False,
) -> str:
    command = [str(config.lilypond_binary), "--loglevel=INFO"]
    if no_print_pages:
        command.append("-dno-print-pages")
    else:
        command.extend(
            [
                "--png",
                f"-dresolution={config.dpi}",
                "-danti-alias-factor=1",
            ]
        )
    command.extend(
        [
            "-I",
            str(source_include_directory),
            "-o",
            str(output_prefix),
            str(wrapper),
        ]
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            cwd=wrapper.parent,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LilyPondRenderError(f"LilyPond render failed for {wrapper.name}: {error}") from error
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        raise LilyPondRenderError(f"LilyPond render failed for {wrapper.name}:\n{output.strip()}")
    return output


def _find_rendered_pages(output_prefix: Path) -> list[Path]:
    single_page = output_prefix.with_suffix(".png")
    if single_page.is_file():
        return [single_page]
    pages = list(output_prefix.parent.glob(f"{output_prefix.name}-page*.png"))
    pages.sort(key=_page_number)
    if not pages:
        raise LilyPondRenderError(f"LilyPond produced no PNG pages for {output_prefix.name}")
    return pages


def _page_number(path: Path) -> int:
    match = _PAGE_NUMBER_PATTERN.search(path.name)
    if match is None:
        raise LilyPondRenderError(f"cannot determine page number from {path.name}")
    return int(match.group(1))


def _materialize_pages(
    pass_pages: dict[str, list[Path]],
    result_directory: Path,
    config: LilyPondRenderConfig,
) -> tuple[tuple[RenderedMaskPage, ...], list[dict[str, object]]]:
    rendered_pages: list[RenderedMaskPage] = []
    metadata_pages: list[dict[str, object]] = []
    page_count = len(pass_pages["pristine"])
    for page_index in range(page_count):
        raw_images = {
            target: _load_grayscale(paths[page_index]) for target, paths in pass_pages.items()
        }
        dimensions = {target: image.size for target, image in raw_images.items()}
        if len(set(dimensions.values())) != 1:
            raise LilyPondRenderError(
                f"page {page_index + 1} render dimensions differ: {dimensions}"
            )

        paper_coverage = ImageOps.invert(raw_images["none"])
        coverages = {
            "staff": ImageChops.subtract(ImageOps.invert(raw_images["staff"]), paper_coverage),
            "notation": ImageChops.subtract(
                ImageOps.invert(raw_images["notation"]), paper_coverage
            ),
            "text": ImageOps.invert(raw_images["text"]),
        }
        masks = {
            name: threshold_coverage(coverage, config.mask_threshold)
            for name, coverage in coverages.items()
        }
        masks["background"] = derive_background(masks)
        foreground_pixels = {
            name: foreground_pixel_count(mask)
            for name, mask in masks.items()
            if name != "background"
        }
        try:
            validate_semantic_masks(masks, expected_nonempty=config.expected_nonempty)
        except MaskQAError as error:
            raise LilyPondRenderError(f"page {page_index + 1} mask QA failed: {error}") from error

        page_directory = result_directory / f"page-{page_index + 1:03d}"
        page_directory.mkdir()
        pristine_path = page_directory / "pristine.png"
        raw_images["pristine"].save(pristine_path, format="PNG", compress_level=9)
        mask_paths: dict[str, Path] = {}
        for name in ("background", "staff", "notation", "text"):
            mask_path = page_directory / f"mask_{name}.png"
            masks[name].save(mask_path, format="PNG", compress_level=9)
            mask_paths[name] = mask_path
        qa_panel_path = page_directory / "qa.png"
        _create_qa_panel(raw_images["pristine"], masks).save(
            qa_panel_path, format="PNG", compress_level=9
        )

        width, height = raw_images["pristine"].size
        rendered_pages.append(
            RenderedMaskPage(
                page=page_index + 1,
                width=width,
                height=height,
                directory=page_directory,
                pristine_path=pristine_path,
                mask_paths=mask_paths,
                qa_panel_path=qa_panel_path,
            )
        )
        metadata_pages.append(
            {
                "page": page_index + 1,
                "dimensions": {"width": width, "height": height},
                "foreground_pixels": foreground_pixels,
                "outputs": {
                    "pristine": str(pristine_path.relative_to(result_directory)),
                    "masks": {
                        name: str(path.relative_to(result_directory))
                        for name, path in mask_paths.items()
                    },
                    "qa_panel": str(qa_panel_path.relative_to(result_directory)),
                },
                "qa": {
                    "dimensions_match": True,
                    "background_is_foreground_complement": True,
                    "background_foreground_overlap_pixels": 0,
                },
            }
        )
    return tuple(rendered_pages), metadata_pages


def _load_grayscale(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        if "A" in opened.getbands():
            rgba = opened.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(white, rgba).convert("L")
        return opened.convert("L")


def _create_qa_panel(pristine: Image.Image, masks: dict[str, Image.Image]) -> Image.Image:
    overlay = pristine.convert("RGB")
    for name, color in (
        ("staff", (220, 45, 45)),
        ("notation", (40, 90, 220)),
        ("text", (30, 160, 80)),
    ):
        color_layer = Image.new("RGB", overlay.size, color)
        tinted = Image.blend(overlay, color_layer, 0.65)
        overlay = Image.composite(tinted, overlay, masks[name])

    panels = [
        ("pristine", pristine.convert("RGB")),
        ("staff", ImageOps.invert(masks["staff"]).convert("RGB")),
        ("notation", ImageOps.invert(masks["notation"]).convert("RGB")),
        ("text", ImageOps.invert(masks["text"]).convert("RGB")),
        ("combined overlay", overlay),
    ]
    maximum_panel_size = (420, 420)
    resized: list[tuple[str, Image.Image]] = []
    for label, panel in panels:
        panel.thumbnail(maximum_panel_size, Image.Resampling.LANCZOS)
        resized.append((label, panel))

    label_height = 24
    canvas_width = sum(panel.width for _, panel in resized)
    canvas_height = max(panel.height for _, panel in resized) + label_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    x_offset = 0
    for label, panel in resized:
        draw.text((x_offset + 6, 6), label, fill="black")
        canvas.paste(panel, (x_offset, label_height))
        x_offset += panel.width
    return canvas


def _relocate_page(page: RenderedMaskPage, output_directory: Path) -> RenderedMaskPage:
    page_directory = output_directory / page.directory.name
    return RenderedMaskPage(
        page=page.page,
        width=page.width,
        height=page.height,
        directory=page_directory,
        pristine_path=page_directory / page.pristine_path.name,
        mask_paths={name: page_directory / path.name for name, path in page.mask_paths.items()},
        qa_panel_path=page_directory / page.qa_panel_path.name,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
