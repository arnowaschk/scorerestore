"""Strict provenance and integrity validation for bundled V1 score sources."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

MANIFEST_SCHEMA_VERSION = 1
ALLOWED_RIGHTS_STATUSES = frozenset({"public_domain", "compatible_license"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RightsRecord:
    """Explicit distribution status for a composition or source file."""

    status: str
    basis: str
    license: str | None = None
    license_url: str | None = None


@dataclass(frozen=True, slots=True)
class ScoreAsset:
    """Validated manifest data for one bundled LilyPond source."""

    id: str
    title: str
    composer: str
    work_identifier: str
    source_url: str
    source_file: str
    source_path: Path
    source_sha256: str
    composition_rights: RightsRecord
    source_file_rights: RightsRecord
    verified_date: date
    notes: str


@dataclass(frozen=True, slots=True)
class ProvenanceValidationReport:
    """Successful strict validation result."""

    manifest_path: Path
    assets: tuple[ScoreAsset, ...]

    @property
    def verified_hashes(self) -> int:
        """Number of source-file hashes verified against local bytes."""

        return len(self.assets)


class ProvenanceValidationError(ValueError):
    """Raised with every detected manifest, rights, or integrity problem."""

    def __init__(self, errors: list[str] | tuple[str, ...]) -> None:
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


def validate_score_manifest(path: str | Path) -> ProvenanceValidationReport:
    """Validate all metadata, rights declarations, paths, and hashes in a score manifest.

    Validation is deliberately strict for bundled public V1 assets. Missing or unresolved rights,
    unsafe paths, absent files, and hash mismatches all fail the complete validation.
    """

    manifest_path = Path(path).resolve()
    manifest = _load_manifest(manifest_path)
    errors: list[str] = []

    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != MANIFEST_SCHEMA_VERSION:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA_VERSION}")

    raw_scores = manifest.get("scores")
    if not isinstance(raw_scores, list) or not raw_scores:
        errors.append("scores must be a nonempty list")
        raise ProvenanceValidationError(errors)

    assets: list[ScoreAsset] = []
    seen_ids: set[str] = set()
    seen_source_files: set[str] = set()
    for index, raw_score in enumerate(raw_scores):
        prefix = f"scores[{index}]"
        if not isinstance(raw_score, dict):
            errors.append(f"{prefix} must be a mapping")
            continue
        if not all(isinstance(key, str) for key in raw_score):
            errors.append(f"{prefix} keys must be strings")
            continue

        asset = _validate_score(
            raw_score,
            prefix=prefix,
            manifest_dir=manifest_path.parent,
            errors=errors,
        )
        if asset is None:
            continue
        if asset.id in seen_ids:
            errors.append(f"{prefix}.id duplicates {asset.id!r}")
        else:
            seen_ids.add(asset.id)
        if asset.source_file in seen_source_files:
            errors.append(f"{prefix}.source_file duplicates {asset.source_file!r}")
        else:
            seen_source_files.add(asset.source_file)
        assets.append(asset)

    if errors:
        raise ProvenanceValidationError(errors)
    return ProvenanceValidationReport(manifest_path=manifest_path, assets=tuple(assets))


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ProvenanceValidationError(
            [f"cannot read manifest {manifest_path}: {error}"]
        ) from error
    except yaml.YAMLError as error:
        raise ProvenanceValidationError([f"invalid YAML in {manifest_path}: {error}"]) from error

    if not isinstance(raw, dict):
        raise ProvenanceValidationError(["manifest root must be a mapping"])
    if not all(isinstance(key, str) for key in raw):
        raise ProvenanceValidationError(["manifest root keys must be strings"])
    return raw


def _validate_score(
    raw: dict[str, Any],
    *,
    prefix: str,
    manifest_dir: Path,
    errors: list[str],
) -> ScoreAsset | None:
    errors_before = len(errors)
    score_id = _required_text(raw, "id", prefix, errors)
    title = _required_text(raw, "title", prefix, errors)
    composer = _required_text(raw, "composer", prefix, errors)
    work_identifier = _required_text(raw, "work_identifier", prefix, errors)
    source_url = _required_text(raw, "source_url", prefix, errors)
    source_file = _required_text(raw, "source_file", prefix, errors)
    source_sha256 = _required_text(raw, "source_sha256", prefix, errors)
    notes = _required_text(raw, "notes", prefix, errors)
    composition_rights = _validate_rights(raw, "composition_rights", prefix, errors)
    source_file_rights = _validate_rights(raw, "source_file_rights", prefix, errors)
    verified_date = _validate_date(raw.get("verified_date"), f"{prefix}.verified_date", errors)

    if source_url is not None and not _is_http_url(source_url):
        errors.append(f"{prefix}.source_url must be an absolute HTTP(S) URL")
    if source_sha256 is not None and _SHA256_PATTERN.fullmatch(source_sha256) is None:
        errors.append(f"{prefix}.source_sha256 must be 64 lowercase hexadecimal characters")

    source_path: Path | None = None
    if source_file is not None:
        source_path = _resolve_source_path(source_file, manifest_dir, prefix, errors)
    if source_path is not None and source_sha256 is not None and source_path.is_file():
        actual_sha256 = _sha256_file(source_path)
        if actual_sha256 != source_sha256:
            errors.append(
                f"{prefix}.source_sha256 mismatch for {source_file!r}: "
                f"expected {source_sha256}, got {actual_sha256}"
            )

    if len(errors) != errors_before:
        return None
    assert score_id is not None
    assert title is not None
    assert composer is not None
    assert work_identifier is not None
    assert source_url is not None
    assert source_file is not None
    assert source_path is not None
    assert source_sha256 is not None
    assert composition_rights is not None
    assert source_file_rights is not None
    assert verified_date is not None
    assert notes is not None
    return ScoreAsset(
        id=score_id,
        title=title,
        composer=composer,
        work_identifier=work_identifier,
        source_url=source_url,
        source_file=source_file,
        source_path=source_path,
        source_sha256=source_sha256,
        composition_rights=composition_rights,
        source_file_rights=source_file_rights,
        verified_date=verified_date,
        notes=notes,
    )


def _required_text(raw: dict[str, Any], field: str, prefix: str, errors: list[str]) -> str | None:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{prefix}.{field} must be a nonempty string")
        return None
    return value


def _validate_rights(
    raw: dict[str, Any], field: str, prefix: str, errors: list[str]
) -> RightsRecord | None:
    value = raw.get(field)
    rights_prefix = f"{prefix}.{field}"
    if not isinstance(value, dict):
        errors.append(f"{rights_prefix} must be a mapping with explicit status and basis")
        return None

    status = _required_text(value, "status", rights_prefix, errors)
    basis = _required_text(value, "basis", rights_prefix, errors)
    license_name = value.get("license")
    license_url = value.get("license_url")
    if status is not None and status not in ALLOWED_RIGHTS_STATUSES:
        allowed = ", ".join(sorted(ALLOWED_RIGHTS_STATUSES))
        errors.append(f"{rights_prefix}.status {status!r} is not distributable; allowed: {allowed}")
    if license_name is not None and (not isinstance(license_name, str) or not license_name.strip()):
        errors.append(f"{rights_prefix}.license must be a nonempty string when present")
    if license_url is not None and (
        not isinstance(license_url, str) or not _is_http_url(license_url)
    ):
        errors.append(f"{rights_prefix}.license_url must be an absolute HTTP(S) URL when present")
    if status == "compatible_license":
        if not isinstance(license_name, str) or not license_name.strip():
            errors.append(f"{rights_prefix}.license is required for compatible_license")
        if not isinstance(license_url, str) or not _is_http_url(license_url):
            errors.append(f"{rights_prefix}.license_url is required for compatible_license")

    if status not in ALLOWED_RIGHTS_STATUSES or basis is None:
        return None
    return RightsRecord(
        status=status,
        basis=basis,
        license=license_name if isinstance(license_name, str) else None,
        license_url=license_url if isinstance(license_url, str) else None,
    )


def _validate_date(value: Any, field: str, errors: list[str]) -> date | None:
    if isinstance(value, datetime):
        errors.append(f"{field} must use YYYY-MM-DD without a time")
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.isoformat() == value:
            return parsed
    errors.append(f"{field} must use a valid YYYY-MM-DD date")
    return None


def _resolve_source_path(
    source_file: str, manifest_dir: Path, prefix: str, errors: list[str]
) -> Path | None:
    relative_path = Path(source_file)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        errors.append(f"{prefix}.source_file must stay within the manifest directory")
        return None
    if relative_path.suffix != ".ly":
        errors.append(f"{prefix}.source_file must reference a .ly file")
        return None

    source_path = (manifest_dir / relative_path).resolve()
    if not source_path.is_relative_to(manifest_dir):
        errors.append(f"{prefix}.source_file resolves outside the manifest directory")
        return None
    if not source_path.is_file():
        errors.append(f"{prefix}.source_file does not exist: {source_file!r}")
        return None
    return source_path


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
