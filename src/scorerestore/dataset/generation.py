"""Materialized V1 dataset generation and sample reproduction."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm

from scorerestore import __version__
from scorerestore.degradation import DegradationConfig, degrade, resolve_degradation_config
from scorerestore.lilypond import (
    LilyPondLayoutConfig,
    LilyPondRenderConfig,
    LilyPondRenderError,
    render_score,
)
from scorerestore.lilypond.constants import LILYPOND_VERSION
from scorerestore.lilypond.renderer import detect_lilypond_version
from scorerestore.provenance import RightsRecord, ScoreAsset, validate_score_manifest
from scorerestore.storage import FilesystemStorage

from .config import DatasetGenerationConfig
from .manifest import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    DatasetManifestError,
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
    skipped_page_count: int


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


@dataclass(slots=True)
class _ProgressReporter:
    """Drive a frequently refreshed tqdm indicator for a parallel generation phase."""

    label: str
    total: int
    enabled: bool
    _bar: Any | None = field(default=None, init=False)

    def start(self) -> None:
        if self.enabled:
            self._bar = tqdm(
                total=self.total,
                desc=self.label,
                unit="task",
                dynamic_ncols=True,
                mininterval=1.0,
                smoothing=0.15,
            )

    def advance(self) -> None:
        if self._bar is not None:
            self._bar.update()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    def message(self, text: str) -> None:
        if self.enabled:
            tqdm.write(text)


def generate_dataset(
    config: DatasetGenerationConfig,
    *,
    output_root: str | Path = "data/generated",
    lilypond_binary: str | Path = "lilypond",
    progress: bool = False,
    update: bool = False,
) -> DatasetGenerationResult:
    """Render, degrade, and materialize one configured V1 dataset.

    Normal generation is atomic and refuses an existing dataset directory. ``update`` instead
    writes in place so an interrupted run can be resumed; every pre-existing artifact is
    hash-checked and preserved rather than replaced.
    """

    output_root_path = Path(output_root).resolve()
    final_directory = output_root_path / config.dataset_id
    if final_directory.exists() and not update:
        raise DatasetGenerationError(f"dataset directory already exists: {final_directory}")
    if final_directory.exists() and not final_directory.is_dir():
        raise DatasetGenerationError(f"dataset path is not a directory: {final_directory}")
    if update:
        _validate_existing_metadata(final_directory, config)
        existing = _complete_existing_result(final_directory, config)
        if existing is not None:
            return existing
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
    temporary_directory: Path | None = None
    if update:
        final_directory.mkdir(parents=True, exist_ok=True)
        final_directory.chmod(0o755)
        storage = FilesystemStorage(final_directory)
    else:
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
        candidates, skipped_pages = _render_candidates(
            assets,
            assignments=assignments,
            config=config,
            storage=storage,
            lilypond_binary=lilypond_binary,
            workers=config.workers,
            progress=progress,
            update=update,
        )
        if not candidates:
            raise DatasetGenerationError("LilyPond rendering produced no candidate pages")
        storage.write_text("reports/skipped_pages.json", _json_text(skipped_pages))
        records = _materialize_samples(
            candidates,
            config=config,
            storage=storage,
            degradation_configs=degradation_configs,
            challenge_config=challenge_config,
            workers=config.workers,
            progress=progress,
            update=update,
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
            skipped_page_count=len(skipped_pages),
        )
        storage.write_text("manifests/dataset.json", _json_text(metadata))
        shutil.rmtree(storage.path(".render-cache"))
        shutil.rmtree(storage.path(".resume-render-cache"), ignore_errors=True)
        validate_dataset_manifest(manifest_path)
        if temporary_directory is not None:
            temporary_directory.replace(final_directory)
    except Exception:
        if temporary_directory is not None:
            shutil.rmtree(temporary_directory, ignore_errors=True)
        raise

    return DatasetGenerationResult(
        dataset_directory=final_directory,
        manifest_path=final_directory / "manifests/samples.jsonl",
        metadata_path=final_directory / "manifests/dataset.json",
        sample_count=len(records),
        split_counts=split_counts,
        skipped_page_count=len(skipped_pages),
    )


def _validate_existing_metadata(directory: Path, config: DatasetGenerationConfig) -> None:
    """Reject an update that would combine artifacts from different configurations."""

    metadata_path = directory / "manifests" / "dataset.json"
    if not metadata_path.is_file():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetGenerationError(f"cannot read existing dataset metadata: {error}") from error
    expected = hashlib.sha256(_canonical_json(config.to_dict())).hexdigest()
    if not isinstance(metadata, dict) or metadata.get("config_sha256") != expected:
        raise DatasetGenerationError(
            "existing dataset metadata is incompatible with this configuration; "
            "use a new dataset_id or output root"
        )


def _complete_existing_result(
    directory: Path, config: DatasetGenerationConfig
) -> DatasetGenerationResult | None:
    """Return an existing valid dataset result so update is a no-op when nothing is missing."""

    manifest_path = directory / "manifests" / "samples.jsonl"
    metadata_path = directory / "manifests" / "dataset.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        return None
    try:
        report = validate_dataset_manifest(manifest_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (DatasetManifestError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    split_counts = metadata.get("split_counts")
    if not isinstance(split_counts, dict):
        return None
    try:
        counts = {name: int(split_counts.get(name, 0)) for name in config.split_weights}
    except (TypeError, ValueError):
        return None
    if len(report.records) != config.target_samples:
        return None
    skipped = metadata.get("skipped_page_count", 0)
    if isinstance(skipped, bool) or not isinstance(skipped, int):
        return None
    return DatasetGenerationResult(
        dataset_directory=directory,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        sample_count=len(report.records),
        split_counts=counts,
        skipped_page_count=skipped,
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
    workers: int,
    progress: bool,
    update: bool,
) -> tuple[list[_RenderedCandidate], list[dict[str, object]]]:
    tasks = [
        (asset, layout_seed, layout)
        for asset in assets
        for layout_seed, layout in _layout_variants(asset.id, config)
    ]
    reporter = _ProgressReporter(
        f"Rendering {len(tasks)} source/layout task(s) with {min(workers, len(tasks))} worker(s)",
        len(tasks),
        progress,
    )
    reporter.start()
    completed_candidates: list[tuple[list[_RenderedCandidate], list[dict[str, object]]] | None] = [
        None
    ] * len(tasks)
    try:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(tasks)), thread_name_prefix="render"
        ) as pool:
            futures = {
                pool.submit(
                    _render_candidate,
                    asset,
                    assignments[asset.id],
                    layout_seed,
                    layout,
                    config,
                    storage,
                    lilypond_binary,
                    update,
                ): index
                for index, (asset, layout_seed, layout) in enumerate(tasks)
            }
            for future in as_completed(futures):
                completed_candidates[futures[future]] = future.result()
                reporter.advance()
    finally:
        reporter.close()
    candidates: list[_RenderedCandidate] = []
    skipped_pages: list[dict[str, object]] = []
    for result in completed_candidates:
        if result is not None:
            candidates.extend(result[0])
            skipped_pages.extend(result[1])
    if skipped_pages:
        reporter.message(
            f"Skipped {len(skipped_pages)} non-trainable rendered page(s); "
            "see reports/skipped_pages.json"
        )
    return candidates, skipped_pages


def _render_candidate(
    asset: ScoreAsset,
    split: str,
    layout_seed: int,
    layout: LilyPondLayoutConfig,
    config: DatasetGenerationConfig,
    storage: FilesystemStorage,
    lilypond_binary: str | Path,
    update: bool,
) -> tuple[list[_RenderedCandidate], list[dict[str, object]]]:
    """Render one independent source/layout task; safe to run alongside other tasks."""

    render_parameters = {
        **layout.to_dict(),
        "layout_seed": layout_seed,
        "dpi": config.dpi,
        "mask_threshold": config.mask_threshold,
        "strict_unknown_grobs": config.strict_unknown_grobs,
    }
    render_id = _stable_id("render", {"source_sha256": asset.source_sha256, **render_parameters})
    cache_path = storage.path(f".render-cache/{render_id}")
    temporary_root: tempfile.TemporaryDirectory[str] | None = None
    if update and cache_path.exists():
        resume_root = storage.make_directory(".resume-render-cache")
        temporary_root = tempfile.TemporaryDirectory(prefix=f"{render_id}-", dir=resume_root)
        cache_path = Path(temporary_root.name) / "render"
    try:
        try:
            result = render_score(
                asset.source_path,
                cache_path,
                config=LilyPondRenderConfig(
                    lilypond_binary=lilypond_binary,
                    dpi=config.dpi,
                    mask_threshold=config.mask_threshold,
                    strict_unknown_grobs=config.strict_unknown_grobs,
                    expected_nonempty=(),
                    layout=layout,
                ),
                expected_source_sha256=asset.source_sha256,
            )
        except LilyPondRenderError as error:
            raise DatasetGenerationError(
                f"render failed for source {asset.id}, layout seed {layout_seed}: {error}"
            ) from error
        _copy_or_verify(storage, result.metadata_path, f"reports/renders/{render_id}.json")
        candidates: list[_RenderedCandidate] = []
        skipped_pages: list[dict[str, object]] = []
        for page in result.pages:
            missing_masks = _missing_trainable_masks(page.mask_paths)
            if missing_masks:
                skipped_pages.append(
                    {
                        "source_id": asset.id,
                        "render_id": render_id,
                        "page": page.page,
                        "layout_seed": layout_seed,
                        "missing_masks": missing_masks,
                    }
                )
                continue
            page_stem = f"{render_id}-p{page.page:03d}"
            clean_path = f"clean/{page_stem}.png"
            _copy_or_verify(storage, page.pristine_path, clean_path)
            mask_paths: dict[str, str] = {}
            mask_hashes: dict[str, str] = {}
            for name in ("background", "staff", "notation", "text"):
                mask_path = f"masks/{name}/{page_stem}.png"
                copied = _copy_or_verify(storage, page.mask_paths[name], mask_path)
                mask_paths[name] = mask_path
                mask_hashes[name] = sha256_file(copied)
            _copy_or_verify(storage, page.qa_panel_path, f"reports/qa/{page_stem}.png")
            candidates.append(
                _RenderedCandidate(
                    asset=asset,
                    split=split,
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
        return candidates, skipped_pages
    finally:
        if temporary_root is not None:
            temporary_root.cleanup()


def _materialize_samples(
    candidates: list[_RenderedCandidate],
    *,
    config: DatasetGenerationConfig,
    storage: FilesystemStorage,
    degradation_configs: tuple[DegradationConfig, ...],
    challenge_config: DegradationConfig,
    workers: int,
    progress: bool,
    update: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(config.seed)
    rng.shuffle(candidates)
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    tasks: list[tuple[_RenderedCandidate, DegradationConfig, int, str]] = []
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
        tasks.append((candidate, selected_config, sample_seed, sample_id))
    reporter = _ProgressReporter(
        f"Degrading and writing {len(tasks)} sample(s) with {min(workers, len(tasks))} worker(s)",
        len(tasks),
        progress,
    )
    existing_records = _existing_records(storage) if update else {}
    reporter.start()
    records: list[dict[str, Any] | None] = [None] * len(tasks)
    try:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(tasks)), thread_name_prefix="degrade"
        ) as pool:
            futures = {}
            for index, (candidate, selected_config, sample_seed, sample_id) in enumerate(tasks):
                existing = existing_records.get(sample_id)
                if existing is not None and _record_artifacts_are_valid(storage, existing):
                    records[index] = existing
                    reporter.advance()
                    continue
                futures[
                    pool.submit(
                        _materialize_sample,
                        candidate,
                        selected_config,
                        sample_seed,
                        sample_id,
                        config,
                        storage,
                        created_at,
                    )
                ] = index
            for future in as_completed(futures):
                records[futures[future]] = future.result()
                reporter.advance()
    finally:
        reporter.close()
    return [record for record in records if record is not None]


def _materialize_sample(
    candidate: _RenderedCandidate,
    selected_config: DegradationConfig,
    sample_seed: int,
    sample_id: str,
    config: DatasetGenerationConfig,
    storage: FilesystemStorage,
    created_at: str,
) -> dict[str, Any]:
    """Create one deterministic degradation/materialization task concurrently."""

    with Image.open(storage.path(candidate.clean_path)) as opened:
        clean = opened.copy()
    degraded = degrade(clean, config=selected_config, seed=sample_seed)
    input_path = f"inputs/{sample_id}.png"
    input_file = _write_or_verify(storage, input_path, _png_bytes(degraded.image))
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
    recipe_file = _write_or_verify(storage, recipe_path, _json_text(recipe).encode("utf-8"))
    return {
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


def _copy_or_verify(storage: FilesystemStorage, source: Path, relative_path: str) -> Path:
    """Copy a new artifact, or preserve an existing byte-identical one during update."""

    destination = storage.path(relative_path)
    if destination.exists():
        if not destination.is_file():
            raise DatasetGenerationError(f"existing artifact is not a file: {relative_path}")
        if sha256_file(source) != sha256_file(destination):
            raise DatasetGenerationError(
                f"existing artifact differs from regenerated content: {relative_path}"
            )
        return destination
    return storage.copy_file(source, relative_path)


def _write_or_verify(storage: FilesystemStorage, relative_path: str, content: bytes) -> Path:
    """Write a missing artifact without replacing an existing artifact during update."""

    destination = storage.path(relative_path)
    if destination.exists():
        if not destination.is_file():
            raise DatasetGenerationError(f"existing artifact is not a file: {relative_path}")
        if sha256_file(destination) != hashlib.sha256(content).hexdigest():
            raise DatasetGenerationError(
                f"existing artifact differs from regenerated content: {relative_path}"
            )
        return destination
    return storage.write_bytes(relative_path, content)


def _existing_records(storage: FilesystemStorage) -> dict[str, dict[str, Any]]:
    """Load any validly encoded records retained before an interrupted update."""

    manifest_path = storage.path("manifests/samples.jsonl")
    if not manifest_path.is_file():
        return {}
    try:
        records = read_manifest_records(manifest_path)
    except DatasetManifestError as error:
        raise DatasetGenerationError(
            "existing update manifest is malformed; repair or remove it before resuming: "
            f"{error}"
        ) from error
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise DatasetGenerationError("existing update manifest contains an invalid sample_id")
        if sample_id in by_id:
            raise DatasetGenerationError(
                f"existing update manifest repeats sample_id {sample_id!r}"
            )
        by_id[sample_id] = record
    return by_id


def _record_artifacts_are_valid(storage: FilesystemStorage, record: dict[str, Any]) -> bool:
    """Check a record's materialized files without rejecting unrelated incomplete records."""

    mask_paths = record.get("mask_paths")
    hashes = record.get("hashes")
    if not isinstance(mask_paths, dict) or not isinstance(hashes, dict):
        return False
    artifacts = {
        "input": record.get("input_path"),
        "clean": record.get("clean_target_path"),
        "recipe": record.get("recipe_path"),
        **{
            f"mask_{name}": mask_paths.get(name)
            for name in ("background", "staff", "notation", "text")
        },
    }
    for name, relative_path in artifacts.items():
        expected = hashes.get(name)
        if not isinstance(relative_path, str) or not relative_path:
            return False
        if not isinstance(expected, str) or len(expected) != 64:
            return False
        try:
            path = storage.path(relative_path)
        except ValueError:
            return False
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


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


def _missing_trainable_masks(mask_paths: dict[str, Path]) -> list[str]:
    """Return required semantic masks that are empty on an otherwise valid rendered page."""

    missing: list[str] = []
    for name in ("staff", "notation"):
        with Image.open(mask_paths[name]) as mask:
            if mask.getbbox() is None:
                missing.append(name)
    return missing


def _dataset_metadata(
    config: DatasetGenerationConfig,
    *,
    assignments: dict[str, str],
    sample_count: int,
    split_counts: dict[str, int],
    source_manifest: Path,
    skipped_page_count: int,
) -> dict[str, object]:
    config_dict = config.to_dict()
    return {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "dataset_generator_schema_version": DATASET_GENERATOR_SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "generator_version": __version__,
        "lilypond_version": LILYPOND_VERSION,
        "sample_count": sample_count,
        "skipped_page_count": skipped_page_count,
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
