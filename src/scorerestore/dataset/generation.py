"""Materialized V1 dataset generation and sample reproduction."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import random
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import Image

from scorerestore import __version__
from scorerestore.degradation import DegradationConfig, degrade, resolve_degradation_config
from scorerestore.lilypond import LilyPondLayoutConfig, LilyPondRenderConfig, render_score
from scorerestore.lilypond.constants import LILYPOND_VERSION
from scorerestore.lilypond.renderer import detect_lilypond_version
from scorerestore.provenance import RightsRecord, ScoreAsset, validate_score_manifest
from scorerestore.storage import FilesystemStorage

from .config import DatasetGenerationConfig
from .manifest import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    read_manifest_records,
    sha256_file,
    validate_dataset_manifest,
)
from .sources import CuratedLilyPondDatasetSource, assign_source_splits

DATASET_GENERATOR_SCHEMA_VERSION = 1


class DatasetGenerationError(RuntimeError):
    """Raised when materialization cannot produce a valid dataset."""


class DatasetReproductionError(RuntimeError):
    """Raised when a manifest sample cannot be validated or reproduced."""


@dataclass(frozen=True, slots=True)
class DatasetGenerationResult:
    dataset_directory: Path
    manifest_path: Path
    metadata_path: Path
    sample_count: int
    split_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class ReproductionResult:
    sample_id: str
    exact_environment: bool
    output_matches: bool
    clean_matches: bool
    masks_match: bool
    output_path: Path | None
    differences: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RenderedCandidate:
    asset: ScoreAsset
    split: str
    render_id: str
    page: int
    width: int
    height: int
    layout_seed: int
    render_parameters: dict[str, object]
    clean_path: str
    clean_sha256: str
    mask_paths: dict[str, str]
    mask_sha256: dict[str, str]


def generate_dataset(
    config: DatasetGenerationConfig,
    *,
    output_root: str | Path = "data/generated",
    lilypond_binary: str | Path = "lilypond",
) -> DatasetGenerationResult:
    """Render, degrade, and atomically materialize one configured V1 dataset."""

    output_root_path = Path(output_root).resolve()
    final_directory = output_root_path / config.dataset_id
    if final_directory.exists():
        raise DatasetGenerationError(f"dataset directory already exists: {final_directory}")
    source_manifest = config.source_manifest.resolve()
    source_provider = CuratedLilyPondDatasetSource(source_manifest, config.source_ids)
    assets = source_provider.assets()
    if not assets:
        raise DatasetGenerationError("dataset source selection is empty")
    detected_lilypond = detect_lilypond_version(lilypond_binary)
    if detected_lilypond != LILYPOND_VERSION:
        raise DatasetGenerationError(
            f"LilyPond version mismatch: expected {LILYPOND_VERSION}, got {detected_lilypond}"
        )

    assignments = assign_source_splits(
        [asset.id for asset in assets], weights=config.split_weights, seed=config.seed
    )
    degradation_configs = tuple(
        resolve_degradation_config(_resolve_config_reference(value))
        for value in config.degradation_configs
    )
    challenge_config = resolve_degradation_config(
        _resolve_config_reference(config.challenge_degradation_config)
    )

    output_root_path.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{config.dataset_id}-", dir=output_root_path)
    )
    temporary_directory.chmod(0o755)
    storage = FilesystemStorage(temporary_directory)
    try:
        for directory in (
            "inputs",
            "clean",
            "masks/background",
            "masks/staff",
            "masks/notation",
            "masks/text",
            "manifests/recipes",
            "reports/renders",
            "reports/qa",
            ".render-cache",
        ):
            storage.make_directory(directory)
        candidates = _render_candidates(
            assets,
            assignments=assignments,
            config=config,
            storage=storage,
            lilypond_binary=lilypond_binary,
        )
        if not candidates:
            raise DatasetGenerationError("LilyPond rendering produced no candidate pages")
        records = _materialize_samples(
            candidates,
            config=config,
            storage=storage,
            degradation_configs=degradation_configs,
            challenge_config=challenge_config,
        )
        manifest_text = "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records
        )
        manifest_path = storage.write_text("manifests/samples.jsonl", manifest_text)
        split_counts = {split: 0 for split in config.split_weights}
        for record in records:
            split_counts[record["split"]] += 1
        metadata = _dataset_metadata(
            config,
            assignments=assignments,
            sample_count=len(records),
            split_counts=split_counts,
            source_manifest=source_manifest,
        )
        storage.write_text("manifests/dataset.json", _json_text(metadata))
        shutil.rmtree(storage.path(".render-cache"))
        validate_dataset_manifest(manifest_path)
        temporary_directory.replace(final_directory)
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise

    return DatasetGenerationResult(
        dataset_directory=final_directory,
        manifest_path=final_directory / "manifests/samples.jsonl",
        metadata_path=final_directory / "manifests/dataset.json",
        sample_count=len(records),
        split_counts=split_counts,
    )


def reproduce_sample(
    sample_id: str,
    *,
    data_root: str | Path = "data/generated",
    dataset_id: str | None = None,
    source_manifest: str | Path = "assets/scores/manifest.yaml",
    lilypond_binary: str | Path = "lilypond",
    output_path: str | Path | None = None,
) -> ReproductionResult:
    """Regenerate one manifest sample and compare exact target/output hashes."""

    manifest_path, record = _find_sample(data_root, sample_id, dataset_id)
    provenance = validate_score_manifest(source_manifest)
    by_id = {asset.id: asset for asset in provenance.assets}
    asset = by_id.get(record["source_id"])
    if asset is None:
        raise DatasetReproductionError(
            f"source {record['source_id']!r} is absent from {source_manifest}"
        )
    if asset.source_sha256 != record["source_sha256"]:
        raise DatasetReproductionError("source SHA-256 differs from the sample manifest")

    detected_lilypond = detect_lilypond_version(lilypond_binary)
    if detected_lilypond != record["lilypond_version"]:
        raise DatasetReproductionError(
            "required LilyPond version is unavailable; exact or best-effort rendering cannot "
            f"continue (required {record['lilypond_version']}, got {detected_lilypond})"
        )
    current_versions = _software_versions()
    recipe_path = manifest_path.parent.parent / record["recipe_path"]
    if sha256_file(recipe_path) != record["hashes"]["recipe"]:
        raise DatasetReproductionError("degradation recipe hash differs from the sample manifest")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recorded_versions = recipe.get("software_versions", {})
    exact_environment = (
        record["generator_version"] == __version__
        and record["dataset_generator_schema_version"] == DATASET_GENERATOR_SCHEMA_VERSION
        and record["lilypond_version"] == LILYPOND_VERSION
        and recorded_versions == current_versions
    )

    layout = _layout_from_manifest(record["render_parameters"])
    with tempfile.TemporaryDirectory(prefix="scorerestore-reproduce-") as temporary:
        render_result = render_score(
            asset.source_path,
            Path(temporary) / "render",
            config=LilyPondRenderConfig(
                lilypond_binary=lilypond_binary,
                dpi=record["dpi"],
                mask_threshold=record["render_parameters"]["mask_threshold"],
                strict_unknown_grobs=record["render_parameters"]["strict_unknown_grobs"],
                layout=layout,
            ),
            expected_source_sha256=asset.source_sha256,
        )
        page_number = record["page"]
        if page_number < 1 or page_number > len(render_result.pages):
            raise DatasetReproductionError(
                f"rendered source no longer contains manifest page {page_number}"
            )
        page = render_result.pages[page_number - 1]
        with Image.open(page.pristine_path) as opened:
            clean = opened.copy()
        degradation_config = recipe["resolved_config"]
        degradation_result = degrade(clean, config=degradation_config, seed=record["seed"])
        encoded_output = _png_bytes(degradation_result.image)
        reproduced_output_hash = hashlib.sha256(encoded_output).hexdigest()
        clean_matches = sha256_file(page.pristine_path) == record["hashes"]["clean"]
        masks_match = all(
            sha256_file(page.mask_paths[name]) == record["hashes"][f"mask_{name}"]
            for name in ("background", "staff", "notation", "text")
        )
        output_matches = reproduced_output_hash == record["hashes"]["input"]

    final_output_path: Path | None = None
    if output_path is not None:
        final_output_path = Path(output_path).resolve()
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        final_output_path.write_bytes(encoded_output)
        final_output_path.chmod(0o644)

    differences: list[str] = []
    if not exact_environment:
        differences.append("software versions differ; reproduction is best-effort")
    if not clean_matches:
        differences.append("pristine render hash differs")
    if not masks_match:
        differences.append("one or more semantic mask hashes differ")
    if not output_matches:
        differences.append("degraded output hash differs")
    if exact_environment and differences:
        raise DatasetReproductionError("exact reproduction failed: " + "; ".join(differences))
    return ReproductionResult(
        sample_id=sample_id,
        exact_environment=exact_environment,
        output_matches=output_matches,
        clean_matches=clean_matches,
        masks_match=masks_match,
        output_path=final_output_path,
        differences=tuple(differences),
    )


def _render_candidates(
    assets: tuple[ScoreAsset, ...],
    *,
    assignments: dict[str, str],
    config: DatasetGenerationConfig,
    storage: FilesystemStorage,
    lilypond_binary: str | Path,
) -> list[_RenderedCandidate]:
    candidates: list[_RenderedCandidate] = []
    for asset in assets:
        for layout_seed, layout in _layout_variants(asset.id, config):
            render_parameters = {
                **layout.to_dict(),
                "layout_seed": layout_seed,
                "dpi": config.dpi,
                "mask_threshold": config.mask_threshold,
                "strict_unknown_grobs": config.strict_unknown_grobs,
            }
            render_id = _stable_id(
                "render", {"source_sha256": asset.source_sha256, **render_parameters}
            )
            render_directory = storage.path(f".render-cache/{render_id}")
            result = render_score(
                asset.source_path,
                render_directory,
                config=LilyPondRenderConfig(
                    lilypond_binary=lilypond_binary,
                    dpi=config.dpi,
                    mask_threshold=config.mask_threshold,
                    strict_unknown_grobs=config.strict_unknown_grobs,
                    layout=layout,
                ),
                expected_source_sha256=asset.source_sha256,
            )
            storage.copy_file(result.metadata_path, f"reports/renders/{render_id}.json")
            for page in result.pages:
                page_stem = f"{render_id}-p{page.page:03d}"
                clean_path = f"clean/{page_stem}.png"
                storage.copy_file(page.pristine_path, clean_path)
                mask_paths: dict[str, str] = {}
                mask_hashes: dict[str, str] = {}
                for name in ("background", "staff", "notation", "text"):
                    mask_path = f"masks/{name}/{page_stem}.png"
                    copied = storage.copy_file(page.mask_paths[name], mask_path)
                    mask_paths[name] = mask_path
                    mask_hashes[name] = sha256_file(copied)
                storage.copy_file(page.qa_panel_path, f"reports/qa/{page_stem}.png")
                candidates.append(
                    _RenderedCandidate(
                        asset=asset,
                        split=assignments[asset.id],
                        render_id=render_id,
                        page=page.page,
                        width=page.width,
                        height=page.height,
                        layout_seed=layout_seed,
                        render_parameters=render_parameters,
                        clean_path=clean_path,
                        clean_sha256=sha256_file(storage.path(clean_path)),
                        mask_paths=mask_paths,
                        mask_sha256=mask_hashes,
                    )
                )
    return candidates


def _materialize_samples(
    candidates: list[_RenderedCandidate],
    *,
    config: DatasetGenerationConfig,
    storage: FilesystemStorage,
    degradation_configs: tuple[DegradationConfig, ...],
    challenge_config: DegradationConfig,
) -> list[dict[str, Any]]:
    rng = random.Random(config.seed)
    rng.shuffle(candidates)
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    records: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for sample_index in range(config.target_samples):
        candidate = candidates[sample_index % len(candidates)]
        selected_config = (
            challenge_config if candidate.split == "challenge" else rng.choice(degradation_configs)
        )
        sample_seed = rng.getrandbits(63)
        identity = {
            "dataset_id": config.dataset_id,
            "render_id": candidate.render_id,
            "page": candidate.page,
            "degradation_config": selected_config.to_dict(),
            "seed": sample_seed,
        }
        sample_id = _stable_id("sample", identity)
        if sample_id in seen_sample_ids:
            raise DatasetGenerationError(f"stable sample ID collision: {sample_id}")
        seen_sample_ids.add(sample_id)
        with Image.open(storage.path(candidate.clean_path)) as opened:
            clean = opened.copy()
        degraded = degrade(clean, config=selected_config, seed=sample_seed)
        input_path = f"inputs/{sample_id}.png"
        input_file = storage.write_bytes(input_path, _png_bytes(degraded.image))
        recipe = dict(degraded.recipe)
        recipe["dataset"] = {
            "dataset_id": config.dataset_id,
            "sample_id": sample_id,
            "source_id": candidate.asset.id,
            "render_id": candidate.render_id,
            "page": candidate.page,
            "split": candidate.split,
        }
        recipe_path = f"manifests/recipes/{sample_id}.json"
        recipe_file = storage.write_text(recipe_path, _json_text(recipe))
        record = {
            "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
            "sample_id": sample_id,
            "dataset_id": config.dataset_id,
            "source_id": candidate.asset.id,
            "source_path": candidate.asset.source_file,
            "source_sha256": candidate.asset.source_sha256,
            "source_license_status": candidate.asset.source_file_rights.status,
            "source_provenance": _source_provenance(candidate.asset),
            "page": candidate.page,
            "split": candidate.split,
            "generator_version": __version__,
            "dataset_generator_schema_version": DATASET_GENERATOR_SCHEMA_VERSION,
            "lilypond_version": LILYPOND_VERSION,
            "seed": sample_seed,
            "render_id": candidate.render_id,
            "render_parameters": candidate.render_parameters,
            "degradation_preset": selected_config.preset,
            "degradation_config": selected_config.to_dict(),
            "degradations": degraded.recipe["operations"],
            "input_path": input_path,
            "clean_target_path": candidate.clean_path,
            "mask_paths": candidate.mask_paths,
            "recipe_path": recipe_path,
            "hashes": {
                "input": sha256_file(input_file),
                "clean": candidate.clean_sha256,
                "recipe": sha256_file(recipe_file),
                **{f"mask_{name}": digest for name, digest in candidate.mask_sha256.items()},
            },
            "dimensions": {"width": candidate.width, "height": candidate.height},
            "dpi": config.dpi,
            "created_at": created_at,
        }
        records.append(record)
    return records


def _layout_variants(
    source_id: str,
    config: DatasetGenerationConfig,
) -> list[tuple[int, LilyPondLayoutConfig]]:
    variants: list[tuple[int, LilyPondLayoutConfig]] = []
    combinations = product(
        config.layout.staff_sizes,
        config.layout.paper_formats,
        config.layout.orientations,
        range(config.layout.variants_per_combination),
    )
    for staff_size, paper_format, orientation, variant_index in combinations:
        seed_material = (
            f"{config.seed}:{source_id}:{staff_size}:{paper_format}:{orientation}:{variant_index}"
        )
        layout_seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
        rng = random.Random(layout_seed)
        low, high = config.layout.margin_range_mm
        margins = [round(rng.uniform(low, high), 3) for _ in range(4)]
        variants.append(
            (
                layout_seed,
                LilyPondLayoutConfig(
                    staff_size=staff_size,
                    paper_format=paper_format,  # type: ignore[arg-type]
                    orientation=orientation,  # type: ignore[arg-type]
                    top_margin_mm=margins[0],
                    bottom_margin_mm=margins[1],
                    left_margin_mm=margins[2],
                    right_margin_mm=margins[3],
                ),
            )
        )
    return variants


def _dataset_metadata(
    config: DatasetGenerationConfig,
    *,
    assignments: dict[str, str],
    sample_count: int,
    split_counts: dict[str, int],
    source_manifest: Path,
) -> dict[str, object]:
    config_dict = config.to_dict()
    return {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "dataset_generator_schema_version": DATASET_GENERATOR_SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "generator_version": __version__,
        "lilypond_version": LILYPOND_VERSION,
        "sample_count": sample_count,
        "split_counts": split_counts,
        "source_splits": assignments,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "config": config_dict,
        "config_sha256": hashlib.sha256(_canonical_json(config_dict)).hexdigest(),
        "software_versions": _software_versions(),
    }


def _find_sample(
    data_root: str | Path,
    sample_id: str,
    dataset_id: str | None,
) -> tuple[Path, dict[str, Any]]:
    root = Path(data_root).resolve()
    manifests = (
        [root / dataset_id / "manifests/samples.jsonl"]
        if dataset_id is not None
        else sorted(root.glob("*/manifests/samples.jsonl"))
    )
    matches: list[tuple[Path, dict[str, Any]]] = []
    for manifest in manifests:
        if not manifest.is_file():
            continue
        for record in read_manifest_records(manifest):
            if record.get("sample_id") == sample_id:
                matches.append((manifest, record))
    if not matches:
        raise DatasetReproductionError(f"sample {sample_id!r} was not found below {root}")
    if len(matches) > 1:
        raise DatasetReproductionError(
            f"sample {sample_id!r} occurs in multiple datasets; pass --dataset-id"
        )
    return matches[0]


def _layout_from_manifest(raw: dict[str, Any]) -> LilyPondLayoutConfig:
    margins = raw["margins_mm"]
    return LilyPondLayoutConfig(
        staff_size=raw["staff_size"],
        paper_format=raw["paper_format"],
        orientation=raw["orientation"],
        top_margin_mm=margins["top"],
        bottom_margin_mm=margins["bottom"],
        left_margin_mm=margins["left"],
        right_margin_mm=margins["right"],
    )


def _source_provenance(asset: ScoreAsset) -> dict[str, object]:
    return {
        "source_url": asset.source_url,
        "composition_rights": _rights_dict(asset.composition_rights),
        "source_file_rights": _rights_dict(asset.source_file_rights),
        "verified_date": asset.verified_date.isoformat(),
        "work_identifier": asset.work_identifier,
        "title": asset.title,
        "composer": asset.composer,
    }


def _rights_dict(rights: RightsRecord) -> dict[str, object]:
    return {
        "status": rights.status,
        "basis": rights.basis,
        "license": rights.license,
        "license_url": rights.license_url,
    }


def _resolve_config_reference(value: str) -> str | Path:
    path = Path(value)
    return path.resolve() if path.is_file() else value


def _stable_id(prefix: str, payload: dict[str, object]) -> str:
    return f"{prefix}-{hashlib.sha256(_canonical_json(payload)).hexdigest()[:24]}"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _png_bytes(image: Image.Image) -> bytes:
    encoded = io.BytesIO()
    image.save(encoded, format="PNG", compress_level=9)
    return encoded.getvalue()


def _software_versions() -> dict[str, str]:
    return {
        "scorerestore": __version__,
        "python": platform.python_version(),
        "pillow": PIL.__version__,
        "numpy": np.__version__,
    }
