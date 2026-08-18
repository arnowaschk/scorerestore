"""Reference-free, diagnostic quality proxies for real-world cleaned score pages.

The real-world inputs intentionally have no pristine annotations.  These measurements therefore
do not claim to be ground-truth restoration or readability scores.  They make observable failure
modes comparable across every configured cleaning panel: lost source ink, added speckles, broken
long horizontal structure, and tiled-inference seams.
"""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

_METRIC_COLUMNS = (
    "cleaned_ink_fraction",
    "source_ink_retention",
    "cleaned_ink_from_source_fraction",
    "tiny_ink_components_per_megapixel",
    "staff_like_line_retention",
    "tile_seam_discontinuity",
    "tile_seam_ink_density_ratio",
)


def quality_row(
    original: Image.Image,
    cleaned: Image.Image,
    *,
    source: str,
    page_number: int,
    comparison_id: str,
    comparison_label: str,
    tile_size: int | None = None,
    overlap: int = 0,
) -> dict[str, str | int | float | None]:
    """Measure one binary cleaned page against its unannotated scan.

    Source ink is an Otsu-derived proxy from the input scan.  It supplies a consistent reference
    for measuring preservation, but is deliberately not presented as ground truth.
    """

    source_pixels = np.asarray(original.convert("L"), dtype=np.uint8)
    cleaned_pixels = np.asarray(cleaned.convert("L"), dtype=np.uint8)
    if source_pixels.shape != cleaned_pixels.shape:
        raise ValueError("original and cleaned pages must have matching dimensions")
    _, source_mask = cv2.threshold(source_pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    source_ink = source_mask.astype(bool)
    cleaned_ink = cleaned_pixels < 128
    source_count = int(np.count_nonzero(source_ink))
    cleaned_count = int(np.count_nonzero(cleaned_ink))
    overlap_count = int(np.count_nonzero(source_ink & cleaned_ink))
    megapixels = cleaned_ink.size / 1_000_000
    rows, columns = cleaned_ink.shape
    seam_metrics = _tile_seam_metrics(
        cleaned_ink,
        height=rows,
        width=columns,
        tile_size=tile_size,
        overlap=overlap,
    )
    return {
        "source": source,
        "page_number": page_number,
        "comparison_id": comparison_id,
        "comparison_label": comparison_label,
        "width": columns,
        "height": rows,
        "cleaned_ink_fraction": float(cleaned_count / cleaned_ink.size),
        "source_ink_retention": _ratio(overlap_count, source_count),
        "cleaned_ink_from_source_fraction": _ratio(overlap_count, cleaned_count),
        "tiny_ink_components_per_megapixel": _tiny_components(cleaned_ink) / megapixels,
        "staff_like_line_retention": _staff_like_line_retention(source_ink, cleaned_ink),
        **seam_metrics,
    }


def write_quality_report(
    output: Path, rows: list[dict[str, str | int | float | None]]
) -> dict[str, str]:
    """Write one-row-per-page metrics plus per-PDF and corpus-level summary tables."""

    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["source"]),
            int(row["page_number"]),
            str(row["comparison_id"]),
        ),
    )
    summary = {
        "schema_version": 1,
        "metric_note": (
            "Reference-free diagnostic proxies only; they do not measure ground-truth restoration "
            "or musical readability without annotated references."
        ),
        "metric_definitions": {
            "cleaned_ink_fraction": (
                "Fraction of output pixels classified as black ink; descriptive, not an "
                "optimisation target."
            ),
            "source_ink_retention": (
                "Recall of Otsu-derived input ink retained by the cleaned output; higher "
                "usually preserves more source structure."
            ),
            "cleaned_ink_from_source_fraction": (
                "Fraction of output ink overlapping Otsu-derived input ink; higher usually "
                "indicates less introduced ink."
            ),
            "tiny_ink_components_per_megapixel": (
                "Connected black components of at most four pixels per megapixel; lower "
                "usually means fewer isolated speckles."
            ),
            "staff_like_line_retention": (
                "Retention of long horizontal structures extracted from input and output; "
                "unavailable where no source structure is found."
            ),
            "tile_seam_discontinuity": (
                "Mean binary-ink discontinuity at internal tiled-inference starts; lower is "
                "better and unavailable for the classical panel."
            ),
            "tile_seam_ink_density_ratio": (
                "Ink density on tiled-inference starts divided by page ink density; values "
                "far above one can indicate seam artefacts and are unavailable for the "
                "classical panel."
            ),
        },
        "aggregation": (
            "All summaries are macro means over finite per-page values; every page has equal "
            "weight."
        ),
        "per_source": _summaries(ordered, key="source"),
        "overall": _summaries(ordered, key=None),
    }
    csv_path = output / "quality_metrics.csv"
    json_path = output / "quality_summary.json"
    markdown_path = output / "quality_report.md"
    csv_path.write_text(_csv(ordered), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(ordered, summary), encoding="utf-8")
    return {
        "metrics_csv": csv_path.name,
        "summary_json": json_path.name,
        "report_markdown": markdown_path.name,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _tiny_components(ink: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    areas = stats[1:count, cv2.CC_STAT_AREA]
    return int(np.count_nonzero(areas <= 4))


def _staff_like_line_retention(source: np.ndarray, cleaned: np.ndarray) -> float | None:
    # Long horizontal opening gives a cheap score-specific proxy without requiring an OMR engine.
    length = max(16, min(source.shape[1] // 12, 128))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    source_lines = cv2.morphologyEx(source.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    cleaned_lines = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    source_count = int(np.count_nonzero(source_lines))
    return _ratio(int(np.count_nonzero(source_lines & cleaned_lines)), source_count)


def _tile_seam_metrics(
    ink: np.ndarray, *, height: int, width: int, tile_size: int | None, overlap: int
) -> dict[str, float | None]:
    if tile_size is None or (height <= tile_size and width <= tile_size):
        return {"tile_seam_discontinuity": None, "tile_seam_ink_density_ratio": None}
    seam = np.zeros_like(ink, dtype=bool)
    for start in _tile_starts(height, tile_size, overlap)[1:]:
        seam[max(0, start - 1) : min(height, start + 2), :] = True
    for start in _tile_starts(width, tile_size, overlap)[1:]:
        seam[:, max(0, start - 1) : min(width, start + 2)] = True
    page_density = float(np.mean(ink))
    seam_density = float(np.mean(ink[seam])) if np.any(seam) else 0.0
    context = (
        cv2.dilate(seam.astype(np.uint8), np.ones((13, 13), dtype=np.uint8)).astype(bool) & ~seam
    )
    context_density = float(np.mean(ink[context])) if np.any(context) else 0.0
    return {
        "tile_seam_discontinuity": abs(seam_density - context_density),
        "tile_seam_ink_density_ratio": None if page_density == 0 else seam_density / page_density,
    }


def _tile_starts(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    if length <= tile_size:
        return (0,)
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return tuple(starts)


def _summaries(
    rows: list[dict[str, str | int | float | None]], key: str | None
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str] | str, list[dict[str, str | int | float | None]]] = defaultdict(
        list
    )
    for row in rows:
        group = (str(row[key]), str(row["comparison_id"])) if key else str(row["comparison_id"])
        groups[group].append(row)
    summaries = []
    for _group, members in sorted(groups.items(), key=lambda item: str(item[0])):
        first = members[0]
        entry: dict[str, Any] = {
            "comparison_id": first["comparison_id"],
            "comparison_label": first["comparison_label"],
            "page_count": len(members),
        }
        if key:
            entry[key] = first[key]
        for metric in _METRIC_COLUMNS:
            values = [float(row[metric]) for row in members if row[metric] is not None]
            entry[metric] = float(np.mean(values)) if values else None
        summaries.append(entry)
    return summaries


def _csv(rows: list[dict[str, str | int | float | None]]) -> str:
    columns = (
        "source",
        "page_number",
        "comparison_id",
        "comparison_label",
        "width",
        "height",
        *_METRIC_COLUMNS,
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _markdown(rows: list[dict[str, str | int | float | None]], summary: dict[str, Any]) -> str:
    lines = [
        "# Real-world diagnostic quality report",
        "",
        summary["metric_note"],
        "",
        (
            "Higher is usually favourable for source-ink, output-ink-from-source, and staff-line "
            "retention. Lower is usually favourable for tiny components and seam discontinuity. "
            "Ink fraction is descriptive only."
        ),
        "",
        "## Per-page comparison panels",
        "",
        _detail_table(rows),
        "",
        "## Per-PDF summary",
        "",
        _summary_table(summary["per_source"], include_source=True),
        "",
        "## Whole input-directory summary",
        "",
        _summary_table(summary["overall"], include_source=False),
        "",
    ]
    return "\n".join(lines)


def _detail_table(rows: list[dict[str, Any]]) -> str:
    return _render_table(
        rows,
        [
            "source",
            "page_number",
            "comparison_id",
            "comparison_label",
            *_METRIC_COLUMNS,
        ],
    )


def _summary_table(rows: list[dict[str, Any]], *, include_source: bool) -> str:
    columns = (["source"] if include_source else []) + [
        "comparison_id",
        "comparison_label",
        "page_count",
        *_METRIC_COLUMNS,
    ]
    return _render_table(rows, columns)


def _render_table(rows: list[dict[str, Any]], columns: list[str] | tuple[str, ...]) -> str:
    headings = [column.replace("_", " ") for column in columns]
    rendered = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            if value is None:
                cells.append("—")
            elif isinstance(value, float):
                cells.append(f"{value:.5f}")
            else:
                cells.append(str(value))
        rendered.append("| " + " | ".join(cells) + " |")
    return "\n".join(rendered)
