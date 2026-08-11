from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from scorerestore.cli import main
from scorerestore.provenance import ProvenanceValidationError, validate_score_manifest

PROJECT_ROOT = Path(__file__).parents[2]
BUNDLED_MANIFEST = PROJECT_ROOT / "assets/scores/manifest.yaml"


def test_bundled_manifest_validates_rights_and_hashes() -> None:
    report = validate_score_manifest(BUNDLED_MANIFEST)

    assert report.verified_hashes == 3
    assert {asset.id for asset in report.assets} == {
        "bach-bwv773-invention-02",
        "beethoven-woo59-fur-elise",
        "foster-hard-times",
    }
    assert all(asset.composition_rights.status == "public_domain" for asset in report.assets)
    assert all(asset.source_file_rights.status == "public_domain" for asset in report.assets)


@pytest.mark.parametrize("rights_field", ["composition_rights", "source_file_rights"])
def test_ci_policy_rejects_missing_rights_field(tmp_path: Path, rights_field: str) -> None:
    manifest, raw = _copy_manifest(tmp_path)
    del raw["scores"][0][rights_field]
    _write_manifest(manifest, raw)

    with pytest.raises(ProvenanceValidationError, match=rights_field):
        validate_score_manifest(manifest)


@pytest.mark.parametrize("status", ["unresolved", "incompatible"])
def test_ci_policy_rejects_non_distributable_rights(tmp_path: Path, status: str) -> None:
    manifest, raw = _copy_manifest(tmp_path)
    raw["scores"][0]["source_file_rights"]["status"] = status
    _write_manifest(manifest, raw)

    with pytest.raises(ProvenanceValidationError, match="is not distributable"):
        validate_score_manifest(manifest)


def test_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest, raw = _copy_manifest(tmp_path)
    source = tmp_path / raw["scores"][0]["source_file"]
    source.write_text(source.read_text(encoding="utf-8") + "% tampered\n", encoding="utf-8")

    with pytest.raises(ProvenanceValidationError, match="source_sha256 mismatch"):
        validate_score_manifest(manifest)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    manifest, raw = _copy_manifest(tmp_path)
    raw["scores"][1]["id"] = raw["scores"][0]["id"]
    _write_manifest(manifest, raw)

    with pytest.raises(ProvenanceValidationError, match="duplicates"):
        validate_score_manifest(manifest)


def test_source_path_cannot_escape_manifest_directory(tmp_path: Path) -> None:
    manifest, raw = _copy_manifest(tmp_path)
    raw["scores"][0]["source_file"] = "../outside.ly"
    _write_manifest(manifest, raw)

    with pytest.raises(ProvenanceValidationError, match="must stay within"):
        validate_score_manifest(manifest)


def test_compatible_license_requires_license_metadata(tmp_path: Path) -> None:
    manifest, raw = _copy_manifest(tmp_path)
    raw["scores"][0]["source_file_rights"]["status"] = "compatible_license"
    _write_manifest(manifest, raw)

    with pytest.raises(ProvenanceValidationError, match="license is required"):
        validate_score_manifest(manifest)


def test_provenance_cli_reports_verified_hashes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", "provenance", "--manifest", str(BUNDLED_MANIFEST)]) == 0

    output = capsys.readouterr().out
    assert "Validated 3 score source(s)" in output
    assert "3 SHA-256 hash(es)" in output


def test_provenance_cli_returns_failure_for_invalid_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, raw = _copy_manifest(tmp_path)
    del raw["scores"][0]["composition_rights"]
    _write_manifest(manifest, raw)

    assert main(["inspect", "provenance", "--manifest", str(manifest)]) == 1
    assert "Provenance validation failed" in capsys.readouterr().err


def _copy_manifest(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    raw = yaml.safe_load(BUNDLED_MANIFEST.read_text(encoding="utf-8"))
    shutil.copytree(BUNDLED_MANIFEST.parent / "sources", tmp_path / "sources")
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, raw)
    return manifest, raw


def _write_manifest(path: Path, raw: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
