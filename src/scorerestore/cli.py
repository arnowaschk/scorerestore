"""Command-line scaffold for the public ScoreRestore V1 interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from scorerestore import __version__
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
        ("generate", "Generate a materialized synthetic dataset (Milestone 4)"),
        ("train", "Train a ScoreRestore model (Milestone 6)"),
        ("evaluate", "Evaluate a model and create reports (Milestone 9)"),
        ("infer", "Restore and segment input pages (Milestone 8)"),
        ("baseline", "Run the classical computer-vision baseline (Milestone 5)"),
        ("degrade", "Apply reproducible synthetic degradation (Milestone 3)"),
    ):
        command_parser = commands.add_parser(name, help=help_text, description=help_text)
        command_parser.set_defaults(command_path=name)

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

    dataset = commands.add_parser("dataset", help="Dataset maintenance commands")
    dataset_commands = dataset.add_subparsers(
        dest="dataset_command", metavar="COMMAND", required=True
    )
    reproduce = dataset_commands.add_parser(
        "reproduce", help="Reproduce a materialized sample (Milestone 4)"
    )
    reproduce.add_argument("sample_id", metavar="SAMPLE_ID")
    reproduce.set_defaults(command_path="dataset reproduce")
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
