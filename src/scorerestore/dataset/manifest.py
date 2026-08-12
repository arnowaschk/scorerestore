"""Human-inspectable JSONL dataset manifest validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .config import SPLIT_NAMES

DATASET_MANIFEST_SCHEMA_VERSION = 1
REQUIRED_SAMPLE_FIELDS = frozenset(
    {
        "sample_id",
        "dataset_id",
        "source_id",
        "source_path",
        "source_sha256",
        "source_license_status",
        "source_provenance",
        "page",
        "split",
        "generator_version",
        "lilypond_version",
        "seed",
        "render_parameters",
        "degradation_preset",
        "degradations",
        "input_path",
        "clean_target_path",
        "mask_paths",
        "recipe_path",
        "hashes",
        "dimensions",
        "dpi",
        "created_at",
    }
)


class DatasetManifestError(ValueError):
    """Raised with every detected dataset manifest problem."""

    def __init__(self, errors: list[str] | tuple[str, ...]) -> None:
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True, slots=True)
class DatasetManifestValidationReport:
    """Validated JSONL records and source-level split assignments."""

    manifest_path: Path
    records: tuple[dict[str, Any], ...]
    source_splits: dict[str, str]


def validate_dataset_manifest(
    path: str | Path,
    *,
    verify_hashes: bool = True,
) -> DatasetManifestValidationReport:
    """Validate schema, target files, hashes, dimensions, and split isolation."""

    manifest_path = Path(path).resolve()
    records = _read_jsonl(manifest_path)
    dataset_root = manifest_path.parent.parent
    errors: list[str] = []
    metadata = _read_dataset_metadata(dataset_root, errors)
    seen_samples: set[str] = set()
    source_splits: dict[str, str] = {}
    dataset_id: str | None = None
    for index, record in enumerate(records, start=1):
        prefix = f"line {index}"
        missing = REQUIRED_SAMPLE_FIELDS - set(record)
        if missing:
            errors.append(f"{prefix}: missing fields: {', '.join(sorted(missing))}")
            continue
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"{prefix}: sample_id must be a nonempty string")
        elif sample_id in seen_samples:
            errors.append(f"{prefix}: duplicate sample_id {sample_id!r}")
        else:
            seen_samples.add(sample_id)
        current_dataset_id = record.get("dataset_id")
        if not isinstance(current_dataset_id, str) or not current_dataset_id:
            errors.append(f"{prefix}: dataset_id must be a nonempty string")
        elif dataset_id is None:
            dataset_id = current_dataset_id
        elif current_dataset_id != dataset_id:
            errors.append(f"{prefix}: dataset_id differs from earlier records")

        source_id = record.get("source_id")
        split = record.get("split")
        if not isinstance(source_id, str) or not source_id:
            errors.append(f"{prefix}: source_id must be a nonempty string")
        if split not in SPLIT_NAMES:
            errors.append(f"{prefix}: split must be one of {', '.join(SPLIT_NAMES)}")
        elif isinstance(source_id, str):
            previous = source_splits.setdefault(source_id, split)
            if previous != split:
                errors.append(
                    f"{prefix}: source {source_id!r} spans splits {previous!r} and {split!r}"
                )

        dimensions = record.get("dimensions")
        if not _valid_dimensions(dimensions):
            errors.append(f"{prefix}: dimensions must contain positive integer width and height")
            dimensions = None
        if not isinstance(record.get("degradations"), list):
            errors.append(f"{prefix}: degradations must be a list")
        _validate_scalar_fields(record, prefix, errors)
        _validate_timestamp(record.get("created_at"), prefix, errors)
        _validate_provenance(record.get("source_provenance"), prefix, errors)
        _validate_artifacts(
            record,
            dataset_root=dataset_root,
            dimensions=dimensions,
            prefix=prefix,
            verify_hashes=verify_hashes,
            errors=errors,
        )

    if metadata is not None:
        if metadata.get("sample_count") != len(records):
            errors.append("dataset metadata sample_count does not match JSONL records")
        if dataset_id is not None and metadata.get("dataset_id") != dataset_id:
            errors.append("dataset metadata dataset_id does not match JSONL records")
        if metadata.get("source_splits") != source_splits:
            errors.append("dataset metadata source_splits does not match JSONL records")

    if errors:
        raise DatasetManifestError(errors)
    return DatasetManifestValidationReport(
        manifest_path=manifest_path,
        records=tuple(records),
        source_splits=source_splits,
    )


def read_manifest_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read JSONL records after strict structural parsing."""

    return tuple(_read_jsonl(Path(path).resolve()))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DatasetManifestError([f"cannot read dataset manifest {path}: {error}"]) from error
    if not lines:
        raise DatasetManifestError(["dataset manifest must contain at least one record"])
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"line {index}: blank JSONL lines are not allowed")
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {index}: invalid JSON: {error.msg}")
            continue
        if not isinstance(record, dict):
            errors.append(f"line {index}: record must be an object")
            continue
        records.append(record)
    if errors:
        raise DatasetManifestError(errors)
    return records


def _validate_artifacts(
    record: dict[str, Any],
    *,
    dataset_root: Path,
    dimensions: dict[str, int] | None,
    prefix: str,
    verify_hashes: bool,
    errors: list[str],
) -> None:
    mask_paths = record.get("mask_paths")
    required_masks = {"background", "staff", "notation", "text"}
    if not isinstance(mask_paths, dict) or set(mask_paths) != required_masks:
        required_names = ", ".join(sorted(required_masks))
        errors.append(f"{prefix}: mask_paths must contain exactly {required_names}")
        return
    artifact_paths: dict[str, Any] = {
        "input": record.get("input_path"),
        "clean": record.get("clean_target_path"),
        "recipe": record.get("recipe_path"),
        **{f"mask_{name}": mask_paths[name] for name in required_masks},
    }
    hashes = record.get("hashes")
    if not isinstance(hashes, dict):
        errors.append(f"{prefix}: hashes must be a mapping")
        hashes = {}
    for name, raw_path in artifact_paths.items():
        path = _resolve_artifact(dataset_root, raw_path, prefix, name, errors)
        if path is None:
            continue
        expected_hash = hashes.get(name)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            errors.append(f"{prefix}: hashes.{name} must be a SHA-256 string")
        elif verify_hashes and sha256_file(path) != expected_hash:
            errors.append(f"{prefix}: hashes.{name} mismatch for {raw_path}")
        if dimensions is not None and name != "recipe":
            try:
                with Image.open(path) as image:
                    if image.size != (dimensions["width"], dimensions["height"]):
                        errors.append(f"{prefix}: {name} dimensions differ from manifest")
            except OSError as error:
                errors.append(f"{prefix}: cannot open {name} image: {error}")


def _resolve_artifact(
    dataset_root: Path,
    raw_path: Any,
    prefix: str,
    name: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        errors.append(f"{prefix}: {name} path must be a nonempty string")
        return None
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        errors.append(f"{prefix}: {name} path must stay inside the dataset")
        return None
    path = (dataset_root / relative).resolve()
    if not path.is_relative_to(dataset_root) or not path.is_file():
        errors.append(f"{prefix}: {name} artifact does not exist: {raw_path}")
        return None
    return path


def _valid_dimensions(raw: Any) -> bool:
    return (
        isinstance(raw, dict)
        and type(raw.get("width")) is int
        and raw["width"] > 0
        and type(raw.get("height")) is int
        and raw["height"] > 0
    )


def _validate_timestamp(raw: Any, prefix: str, errors: list[str]) -> None:
    if not isinstance(raw, str):
        errors.append(f"{prefix}: created_at must be an ISO-8601 string")
        return
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        errors.append(f"{prefix}: created_at must be a timezone-aware ISO-8601 timestamp")


def _validate_provenance(raw: Any, prefix: str, errors: list[str]) -> None:
    if not isinstance(raw, dict):
        errors.append(f"{prefix}: source_provenance must be a mapping")
        return
    for field in ("source_url", "composition_rights", "source_file_rights", "verified_date"):
        if field not in raw:
            errors.append(f"{prefix}: source_provenance.{field} is required")
    for rights_name in ("composition_rights", "source_file_rights"):
        rights = raw.get(rights_name)
        if not isinstance(rights, dict):
            errors.append(f"{prefix}: source_provenance.{rights_name} must be a mapping")
        elif rights.get("status") not in {"public_domain", "compatible_license"}:
            errors.append(f"{prefix}: source_provenance.{rights_name}.status is not distributable")
        elif not isinstance(rights.get("basis"), str) or not rights["basis"].strip():
            errors.append(f"{prefix}: source_provenance.{rights_name}.basis is required")


def _validate_scalar_fields(record: dict[str, Any], prefix: str, errors: list[str]) -> None:
    if record.get("schema_version") != DATASET_MANIFEST_SCHEMA_VERSION:
        errors.append(f"{prefix}: schema_version must be {DATASET_MANIFEST_SCHEMA_VERSION}")
    if record.get("source_license_status") not in {"public_domain", "compatible_license"}:
        errors.append(f"{prefix}: source_license_status is not distributable")
    source_sha256 = record.get("source_sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        errors.append(f"{prefix}: source_sha256 must be lowercase SHA-256")
    for field in ("page", "dpi"):
        value = record.get(field)
        if type(value) is not int or value < 1:
            errors.append(f"{prefix}: {field} must be a positive integer")
    if type(record.get("seed")) is not int:
        errors.append(f"{prefix}: seed must be an integer")
    for field in ("generator_version", "lilypond_version", "degradation_preset"):
        value = record.get(field)
        if not isinstance(value, str) or not value:
            errors.append(f"{prefix}: {field} must be a nonempty string")
    render_parameters = record.get("render_parameters")
    required_render = {
        "staff_size",
        "paper_format",
        "orientation",
        "margins_mm",
        "layout_seed",
        "dpi",
        "mask_threshold",
        "strict_unknown_grobs",
    }
    if not isinstance(render_parameters, dict) or not required_render <= set(render_parameters):
        errors.append(f"{prefix}: render_parameters is incomplete")


def _read_dataset_metadata(dataset_root: Path, errors: list[str]) -> dict[str, Any] | None:
    metadata_path = dataset_root / "manifests/dataset.json"
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except OSError as error:
        errors.append(f"cannot read dataset metadata {metadata_path}: {error}")
        return None
    except json.JSONDecodeError as error:
        errors.append(f"invalid dataset metadata JSON: {error.msg}")
        return None
    if not isinstance(raw, dict):
        errors.append("dataset metadata root must be an object")
        return None
    if raw.get("schema_version") != DATASET_MANIFEST_SCHEMA_VERSION:
        errors.append(f"dataset metadata schema_version must be {DATASET_MANIFEST_SCHEMA_VERSION}")
    return raw
