from __future__ import annotations

from scorerestore.dataset.config import SPLIT_NAMES
from scorerestore.dataset.sources import assign_source_splits


def test_source_split_assignment_is_deterministic_and_exclusive() -> None:
    source_ids = [f"source-{index}" for index in range(100)]
    weights = {"train": 0.7, "validation": 0.1, "test": 0.1, "challenge": 0.1}

    first = assign_source_splits(source_ids, weights=weights, seed=1234)
    second = assign_source_splits(source_ids, weights=weights, seed=1234)

    assert first == second
    assert set(first) == set(source_ids)
    assert set(first.values()) == set(SPLIT_NAMES)
    train_sources = {source_id for source_id, split in first.items() if split == "train"}
    test_sources = {source_id for source_id, split in first.items() if split == "test"}
    assert train_sources.isdisjoint(test_sources)


def test_different_split_seed_changes_assignments() -> None:
    source_ids = [f"source-{index}" for index in range(20)]
    weights = {"train": 0.7, "validation": 0.1, "test": 0.1, "challenge": 0.1}

    assert assign_source_splits(source_ids, weights=weights, seed=1) != assign_source_splits(
        source_ids, weights=weights, seed=2
    )
