from __future__ import annotations

from pathlib import Path

import yaml


def test_reviewed_mutopia_expansion_catalog_has_40_unique_public_domain_candidates() -> None:
    catalog = yaml.safe_load(
        (Path(__file__).parents[2] / "assets/scores/corpus-40.yaml").read_text(encoding="utf-8")
    )

    assert catalog["schema_version"] == 1
    assert len(catalog["scores"]) == 40
    assert len({item["id"] for item in catalog["scores"]}) == 40
    assert {item["coverage"] for item in catalog["scores"]} >= {
        "chamber_ensemble",
        "dense_polyphony",
        "multiple_voices",
        "piano",
        "sparse_notation",
        "vocal_lyrics",
    }
    assert all(item["death_year"] < 1929 for item in catalog["scores"])
