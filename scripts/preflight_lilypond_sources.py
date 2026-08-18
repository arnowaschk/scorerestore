#!/usr/bin/env python3
"""Validate every manifest source with the ScoreRestore LilyPond compatibility pipeline.

The preflight converts each source, applies ScoreRestore's temporary legacy-syntax repairs, and
engraves it without writing page images.  It is a fast fail-fast guard before full dataset
generation: every source must pass the pinned LilyPond version and strict Grob classification.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scorerestore.lilypond import LilyPondRenderConfig, LilyPondRenderError, preflight_score
from scorerestore.provenance import ProvenanceValidationError, validate_score_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="strict score manifest")
    parser.add_argument("--lilypond", default="lilypond", help="LilyPond executable")
    parser.add_argument(
        "--workers",
        default="auto",
        help="parallel workers: positive integer or auto (default: auto)",
    )
    args = parser.parse_args()
    try:
        report = validate_score_manifest(args.manifest)
        workers = _workers(args.workers)
    except (ProvenanceValidationError, ValueError) as error:
        print(f"LilyPond source preflight failed: {error}", file=sys.stderr)
        return 1

    assets = report.assets
    active_workers = min(workers, len(assets))
    print(f"Preflighting {len(assets)} source(s) with {active_workers} worker(s).", flush=True)
    failures: list[tuple[str, str]] = []
    completed = 0
    next_report = max(1, (len(assets) + 19) // 20)
    with ThreadPoolExecutor(
        max_workers=active_workers, thread_name_prefix="lilypond-preflight"
    ) as pool:
        futures = {
            pool.submit(
                preflight_score,
                asset.source_path,
                config=LilyPondRenderConfig(
                    lilypond_binary=args.lilypond,
                    strict_unknown_grobs=True,
                ),
                expected_source_sha256=asset.source_sha256,
            ): asset.id
            for asset in assets
        }
        for future in as_completed(futures):
            source_id = futures[future]
            try:
                future.result()
            except LilyPondRenderError as error:
                failures.append((source_id, str(error)))
            completed += 1
            if completed == len(assets) or completed >= next_report:
                percent = 100 * completed / len(assets)
                print(
                    f"LilyPond preflight: {completed}/{len(assets)} ({percent:.0f}%)",
                    flush=True,
                )
                next_report += max(1, (len(assets) + 19) // 20)

    if failures:
        print(f"LilyPond source preflight failed for {len(failures)} source(s):", file=sys.stderr)
        for source_id, error in sorted(failures):
            print(f"- {source_id}: {error}", file=sys.stderr)
        return 1
    print(f"LilyPond preflight passed for all {len(assets)} source(s).")
    return 0


def _workers(raw: str) -> int:
    if raw == "auto":
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 1
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("--workers must be a positive integer or auto") from error
    if value <= 0:
        raise ValueError("--workers must be a positive integer or auto")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
