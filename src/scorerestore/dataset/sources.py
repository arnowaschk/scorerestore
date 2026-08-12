"""Validated curated score source and deterministic source-level splitting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from scorerestore.provenance import ScoreAsset, validate_score_manifest

from .config import SPLIT_NAMES


class DatasetSourceError(ValueError):
    """Raised when configured source selection cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class CuratedLilyPondDatasetSource:
    """V1 ``DatasetSource`` backed by the strict score provenance manifest."""

    manifest_path: Path
    source_ids: tuple[str, ...] | None = None

    def assets(self) -> tuple[ScoreAsset, ...]:
        report = validate_score_manifest(self.manifest_path)
        if self.source_ids is None:
            return report.assets
        by_id = {asset.id: asset for asset in report.assets}
        missing = set(self.source_ids) - set(by_id)
        if missing:
            raise DatasetSourceError(
                f"source_ids not found in provenance manifest: {', '.join(sorted(missing))}"
            )
        return tuple(by_id[source_id] for source_id in self.source_ids)


def assign_source_splits(
    source_ids: tuple[str, ...] | list[str],
    *,
    weights: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign each underlying source to exactly one deterministic split."""

    if set(weights) != set(SPLIT_NAMES):
        raise ValueError("split weights must define train, validation, test, and challenge")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("at least one split weight must be positive")
    cumulative: list[tuple[str, float]] = []
    running = 0.0
    for split in SPLIT_NAMES:
        running += weights[split] / total
        cumulative.append((split, running))

    assignments: dict[str, str] = {}
    for source_id in source_ids:
        digest = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64
        selected = SPLIT_NAMES[-1]
        for split, threshold in cumulative:
            if fraction < threshold:
                selected = split
                break
        assignments[source_id] = selected
    return assignments
