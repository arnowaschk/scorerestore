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


class LilyPondRenderError(RuntimeError):
    """Raised when rendering, version validation, or mask QA fails."""


@dataclass(frozen=True, slots=True)
class LilyPondRenderConfig:
    """Deterministic V1 render settings."""

    lilypond_binary: str | Path = "lilypond"
    dpi: int = DEFAULT_DPI
    mask_threshold: float = DEFAULT_MASK_THRESHOLD
    strict_unknown_grobs: bool = False
    expected_nonempty: tuple[str, ...] = ("staff", "notation")
    timeout_seconds: int = 180

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


def _convert_source(
    source_path: Path,
    work_directory: Path,
    config: LilyPondRenderConfig,
) -> Path:
    lilypond_binary = Path(config.lilypond_binary)
    convert_binary = lilypond_binary.with_name("convert-ly")
    if not convert_binary.exists():
        located = shutil.which("convert-ly")
        if located is None:
            raise LilyPondRenderError("convert-ly was not found beside LilyPond or on PATH")
        convert_binary = Path(located)

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

    converted_source = work_directory / "converted-source.ly"
    converted_source.write_text(completed.stdout, encoding="utf-8")
    return converted_source


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
        wrapper.write_text(_wrapper_source(target, converted_source), encoding="utf-8")
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


def _wrapper_source(target: str, converted_source: Path) -> str:
    registry = "\n".join(
        f"    ({name} . {classification})"
        for name, classification in sorted(GROB_CLASS_REGISTRY.items())
    )
    include_path = str(converted_source).replace("\\", "/").replace('"', '\\"')
    return f'''\\version "{LILYPOND_VERSION}"

#(define scorerestore-target '{target})
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
'''


def _run_lilypond(
    wrapper: Path,
    output_prefix: Path,
    source_include_directory: Path,
    config: LilyPondRenderConfig,
) -> str:
    command = [
        str(config.lilypond_binary),
        "--png",
        f"-dresolution={config.dpi}",
        "-danti-alias-factor=1",
        "--loglevel=INFO",
        "-I",
        str(source_include_directory),
        "-o",
        str(output_prefix),
        str(wrapper),
    ]
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
