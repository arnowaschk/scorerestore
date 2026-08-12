"""Command-line scaffold for the public ScoreRestore V1 interface."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from scorerestore import __version__
from scorerestore.dataset import (
    DatasetManifestError,
    generate_dataset,
    load_dataset_config,
    reproduce_sample,
    validate_dataset_manifest,
)
from scorerestore.dataset.config import DatasetConfigError
from scorerestore.dataset.generation import DatasetGenerationError, DatasetReproductionError
from scorerestore.degradation import (
    PRESET_NAMES,
    DegradationConfigError,
    degrade,
    recipe_json,
)
from scorerestore.lilypond.constants import (
    DEFAULT_DPI,
    DEFAULT_MASK_THRESHOLD,
    LILYPOND_VERSION,
)
from scorerestore.lilypond.renderer import (
    LilyPondRenderConfig,
    LilyPondRenderError,
    detect_lilypond_version,
    render_score,
)
from scorerestore.provenance import ProvenanceValidationError, validate_score_manifest

_NOT_IMPLEMENTED_MESSAGE = (
    "{command} is part of the public ScoreRestore V1 interface but is reserved for a later "
    "milestone and is not implemented yet."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the public V1 command tree without implementing later milestones."""

    parser = argparse.ArgumentParser(
        prog="scorerestore",
        description=(
            "ScoreRestore V1 proof of concept for sheet-music restoration and semantic analysis. "
            "ScoreRestore is not an OMR system."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    for name, help_text in (
        ("train", "Train a ScoreRestore model (Milestone 6)"),
        ("evaluate", "Evaluate a model and create reports (Milestone 9)"),
        ("infer", "Restore and segment input pages (Milestone 8)"),
        ("baseline", "Run the classical computer-vision baseline (Milestone 5)"),
    ):
        command_parser = commands.add_parser(name, help=help_text, description=help_text)
        command_parser.set_defaults(command_path=name)

    generation = commands.add_parser("generate", help="Generate a materialized synthetic dataset")
    generation.add_argument("-c", "--config", type=Path, required=True)
    generation.add_argument("--output-root", type=Path, default=Path("data/generated"))
    generation.add_argument(
        "--lilypond", default="lilypond", help="LilyPond executable (default: lilypond)"
    )
    generation.set_defaults(command_path="generate", handler=_generate_dataset)

    degradation = commands.add_parser(
        "degrade", help="Apply reproducible pixel-aligned synthetic degradation"
    )
    degradation.add_argument("input", type=Path, help="single-page raster image")
    degradation.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output PNG (default: INPUT.degraded.png)",
    )
    config_selection = degradation.add_mutually_exclusive_group()
    config_selection.add_argument(
        "-c", "--config", type=Path, help="degradation YAML configuration"
    )
    config_selection.add_argument("--preset", choices=PRESET_NAMES, default=None)
    degradation.add_argument("--seed", type=int, default=0)
    degradation.add_argument(
        "--recipe", type=Path, help="recipe JSON path (default: OUTPUT with .recipe.json)"
    )
    degradation.set_defaults(command_path="degrade", handler=_degrade_image)

    inspect = commands.add_parser("inspect", help="Inspect provenance and later V1 artifacts")
    inspect.set_defaults(command_path="inspect")
    inspect_commands = inspect.add_subparsers(dest="inspect_command", metavar="COMMAND")
    provenance = inspect_commands.add_parser(
        "provenance", help="Validate bundled score provenance, rights, and hashes"
    )
    provenance.add_argument(
        "--manifest",
        type=Path,
        default=Path("assets/scores/manifest.yaml"),
        help="score manifest path (default: assets/scores/manifest.yaml)",
    )
    provenance.set_defaults(command_path="inspect provenance", handler=_validate_provenance)
    environment = inspect_commands.add_parser(
        "environment", help="Print pinned and detected runtime versions"
    )
    environment.add_argument(
        "--lilypond", default="lilypond", help="LilyPond executable (default: lilypond)"
    )
    environment.set_defaults(command_path="inspect environment", handler=_inspect_environment)

    masks = inspect_commands.add_parser(
        "masks", help="Render aligned masks and a visual QA panel for a LilyPond source"
    )
    masks.add_argument("source", type=Path, help="ordinary LilyPond .ly source")
    masks.add_argument("-o", "--output", type=Path, required=True, help="new output directory")
    masks.add_argument(
        "--lilypond", default="lilypond", help="LilyPond executable (default: lilypond)"
    )
    masks.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    masks.add_argument("--mask-threshold", type=float, default=DEFAULT_MASK_THRESHOLD)
    masks.add_argument("--strict-unknown-grobs", action="store_true")
    masks.add_argument(
        "--manifest",
        type=Path,
        default=Path("assets/scores/manifest.yaml"),
        help="manifest used to verify bundled source hashes",
    )
    masks.set_defaults(command_path="inspect masks", handler=_inspect_masks)

    dataset = commands.add_parser("dataset", help="Dataset maintenance commands")
    dataset_commands = dataset.add_subparsers(
        dest="dataset_command", metavar="COMMAND", required=True
    )
    reproduce = dataset_commands.add_parser(
        "reproduce", help="Reproduce and hash-check a materialized sample"
    )
    reproduce.add_argument("sample_id", metavar="SAMPLE_ID")
    reproduce.add_argument("--data-root", type=Path, default=Path("data/generated"))
    reproduce.add_argument("--dataset-id")
    reproduce.add_argument(
        "--source-manifest", type=Path, default=Path("assets/scores/manifest.yaml")
    )
    reproduce.add_argument(
        "--lilypond", default="lilypond", help="LilyPond executable (default: lilypond)"
    )
    reproduce.add_argument("-o", "--output", type=Path)
    reproduce.set_defaults(command_path="dataset reproduce", handler=_reproduce_sample)
    validate = dataset_commands.add_parser(
        "validate", help="Validate a materialized JSONL manifest and artifacts"
    )
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--skip-hashes", action="store_true")
    validate.set_defaults(command_path="dataset validate", handler=_validate_dataset)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ScoreRestore command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is not None:
        return handler(args)
    command_path = args.command_path
    print(_NOT_IMPLEMENTED_MESSAGE.format(command=command_path), file=sys.stderr)
    return 2


def _validate_provenance(args: argparse.Namespace) -> int:
    try:
        report = validate_score_manifest(args.manifest)
    except ProvenanceValidationError as error:
        print(f"Provenance validation failed for {args.manifest}:", file=sys.stderr)
        for detail in error.errors:
            print(f"- {detail}", file=sys.stderr)
        return 1

    print(
        f"Validated {len(report.assets)} score source(s) and "
        f"{report.verified_hashes} SHA-256 hash(es) from {args.manifest}."
    )
    return 0


def _inspect_environment(args: argparse.Namespace) -> int:
    try:
        detected_lilypond = detect_lilypond_version(args.lilypond)
        lilypond_error = None
    except LilyPondRenderError as error:
        detected_lilypond = None
        lilypond_error = str(error)
    environment = {
        "scorerestore_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "lilypond": {
            "binary": str(args.lilypond),
            "required_version": LILYPOND_VERSION,
            "detected_version": detected_lilypond,
            "matches_required": detected_lilypond == LILYPOND_VERSION,
            "error": lilypond_error,
        },
    }
    print(json.dumps(environment, indent=2, sort_keys=True))
    return 0 if detected_lilypond is not None else 1


def _inspect_masks(args: argparse.Namespace) -> int:
    expected_sha256: str | None = None
    manifest_path: Path = args.manifest
    if manifest_path.is_file():
        try:
            report = validate_score_manifest(manifest_path)
        except ProvenanceValidationError as error:
            print(f"Provenance validation failed for {manifest_path}:", file=sys.stderr)
            for detail in error.errors:
                print(f"- {detail}", file=sys.stderr)
            return 1
        source_path = args.source.resolve()
        for asset in report.assets:
            if asset.source_path == source_path:
                expected_sha256 = asset.source_sha256
                break

    try:
        result = render_score(
            args.source,
            args.output,
            config=LilyPondRenderConfig(
                lilypond_binary=args.lilypond,
                dpi=args.dpi,
                mask_threshold=args.mask_threshold,
                strict_unknown_grobs=args.strict_unknown_grobs,
            ),
            expected_source_sha256=expected_sha256,
        )
    except (LilyPondRenderError, ValueError) as error:
        print(f"Mask inspection failed: {error}", file=sys.stderr)
        return 1

    verification = "verified against manifest" if result.source_hash_verified else "recorded only"
    print(
        f"Rendered {len(result.pages)} page(s) with LilyPond {result.lilypond_version}; "
        f"source hash {verification}; output: {result.output_directory}"
    )
    if result.unknown_grobs:
        print(f"Unknown grobs defaulted to notation: {', '.join(result.unknown_grobs)}")
    return 0


def _degrade_image(args: argparse.Namespace) -> int:
    input_path: Path = args.input.resolve()
    output_argument = args.output or input_path.with_name(f"{input_path.stem}.degraded.png")
    output_path: Path = output_argument.resolve()
    recipe_argument = args.recipe or output_path.with_suffix(".recipe.json")
    recipe_path: Path = recipe_argument.resolve()

    if output_path.suffix.lower() != ".png":
        print("Degradation output must use the .png suffix.", file=sys.stderr)
        return 1
    if input_path in {output_path, recipe_path} or output_path == recipe_path:
        print("Input, degraded image, and recipe paths must be distinct.", file=sys.stderr)
        return 1

    try:
        with Image.open(input_path) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("degrade accepts one raster page at a time")
            source = opened.copy()
        config = args.config if args.config is not None else (args.preset or "medium")
        result = degrade(source, config=config, seed=args.seed)
        _write_degradation_artifacts(result.image, output_path, recipe_json(result), recipe_path)
    except (DegradationConfigError, OSError, UnidentifiedImageError, ValueError) as error:
        print(f"Degradation failed: {error}", file=sys.stderr)
        return 1

    operations = result.recipe["operations"]
    print(
        f"Applied {len(operations)} degradation(s) with seed {args.seed}; "
        f"image: {output_path}; recipe: {recipe_path}"
    )
    return 0


def _write_degradation_artifacts(
    image: Image.Image,
    output_path: Path,
    serialized_recipe: str,
    recipe_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    image_temporary: Path | None = None
    recipe_temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}-", suffix=".tmp", dir=output_path.parent, delete=False
        ) as temporary:
            image_temporary = Path(temporary.name)
        image.save(image_temporary, format="PNG", compress_level=9)
        image_temporary.chmod(0o644)
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f".{recipe_path.name}-",
            suffix=".tmp",
            dir=recipe_path.parent,
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary.write(serialized_recipe)
            recipe_temporary = Path(temporary.name)
        recipe_temporary.chmod(0o644)
        image_temporary.replace(output_path)
        image_temporary = None
        recipe_temporary.replace(recipe_path)
        recipe_temporary = None
    finally:
        if image_temporary is not None:
            image_temporary.unlink(missing_ok=True)
        if recipe_temporary is not None:
            recipe_temporary.unlink(missing_ok=True)


def _generate_dataset(args: argparse.Namespace) -> int:
    try:
        config = load_dataset_config(args.config)
        result = generate_dataset(
            config,
            output_root=args.output_root,
            lilypond_binary=args.lilypond,
        )
    except (
        DatasetConfigError,
        DatasetGenerationError,
        LilyPondRenderError,
        ProvenanceValidationError,
        ValueError,
    ) as error:
        print(f"Dataset generation failed: {error}", file=sys.stderr)
        return 1
    counts = ", ".join(f"{name}={count}" for name, count in result.split_counts.items())
    print(
        f"Generated {result.sample_count} sample(s) at {result.dataset_directory}; "
        f"splits: {counts}; manifest: {result.manifest_path}"
    )
    return 0


def _reproduce_sample(args: argparse.Namespace) -> int:
    try:
        result = reproduce_sample(
            args.sample_id,
            data_root=args.data_root,
            dataset_id=args.dataset_id,
            source_manifest=args.source_manifest,
            lilypond_binary=args.lilypond,
            output_path=args.output,
        )
    except (
        DatasetReproductionError,
        LilyPondRenderError,
        ProvenanceValidationError,
        ValueError,
    ) as error:
        print(f"Dataset reproduction failed: {error}", file=sys.stderr)
        return 1
    mode = "exact" if result.exact_environment else "best-effort"
    hashes = result.output_matches and result.clean_matches and result.masks_match
    print(
        f"{mode.capitalize()} reproduction for {result.sample_id}: "
        f"{'all hashes match' if hashes else 'hash differences detected'}"
    )
    for difference in result.differences:
        print(f"- {difference}")
    if result.output_path is not None:
        print(f"Reproduced input: {result.output_path}")
    return 0 if hashes else 1


def _validate_dataset(args: argparse.Namespace) -> int:
    try:
        report = validate_dataset_manifest(
            args.manifest,
            verify_hashes=not args.skip_hashes,
        )
    except DatasetManifestError as error:
        print(f"Dataset manifest validation failed for {args.manifest}:", file=sys.stderr)
        for detail in error.errors:
            print(f"- {detail}", file=sys.stderr)
        return 1
    print(
        f"Validated {len(report.records)} sample(s) from {len(report.source_splits)} "
        f"source(s) in {report.manifest_path}."
    )
    return 0
