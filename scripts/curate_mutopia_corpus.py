#!/usr/bin/env python3
"""Materialize the reviewed 40-file Mutopia V1 corpus into a new strict score-source directory.

This intentionally writes only to a caller-selected *new* directory. Review the resulting sources
and `manifest.yaml`, then copy them into a release branch deliberately; never replace bundled assets
implicitly. The upstream Git commit, every raw URL, and every resulting SHA-256 are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

_INCLUDE_PATTERN = re.compile(r'\\include\s+"([^"\\]+)"')
_LILYPOND_BUILTIN_INCLUDES = frozenset(
    {"articulate.ly", "deutsch.ly", "english.ly", "italiano.ly", "nederlands.ly"}
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--catalog",
        type=Path,
        default=Path("assets/scores/corpus-40.yaml"),
        help="reviewed corpus catalogue",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="new output directory")
    parser.add_argument(
        "--from-local-repository",
        type=Path,
        help="read pinned paths from a local Mutopia checkout instead of downloading",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to replace existing output directory: {output}")
    catalog = _catalog(args.catalog)
    output.mkdir(parents=True)
    try:
        assets = []
        reports = []
        for item in catalog["scores"]:
            source_tree = _source_tree(
                item["path"], catalog["upstream"], args.from_local_repository
            )
            source = source_tree[item["path"]]
            _assert_public_domain_header(source, item["id"])
            source_directory = output / "sources" / item["id"]
            source_directory.mkdir(parents=True)
            source_parent = PurePosixPath(item["path"]).parent
            for repository_path, source_bytes in source_tree.items():
                relative_path = PurePosixPath(repository_path).relative_to(source_parent)
                destination = source_directory / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source_bytes)
            source_path = source_directory / PurePosixPath(item["path"]).name
            source_url = _source_url(catalog["upstream"], item["path"])
            assets.append(
                {
                    "id": item["id"],
                    "title": item["work"],
                    "composer": item["composer"],
                    "work_identifier": item["work"],
                    "source_url": source_url,
                    "source_file": str(source_path.relative_to(output)),
                    "source_sha256": hashlib.sha256(source).hexdigest(),
                    "composition_rights": {
                        "status": "public_domain",
                        "basis": (
                            f"{item['composer']} died in {item['death_year']}; the composition "
                            "is public domain."
                        ),
                    },
                    "source_file_rights": {
                        "status": "public_domain",
                        "basis": catalog["upstream"]["source_file_rights_basis"],
                    },
                    "verified_date": "2026-08-13",
                    "notes": f"Curated V1 coverage: {item['coverage']}.",
                }
            )
            reports.append(_source_report(item["id"], source_tree, catalog["upstream"]))
        manifest = {"schema_version": 1, "scores": assets}
        (output / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        (output / "CURATION_REPORT.json").write_text(
            json.dumps(
                {
                    "catalog": str(args.catalog.resolve()),
                    "upstream": catalog["upstream"],
                    "score_count": len(assets),
                    "source_file_count": sum(len(item["files"]) for item in reports),
                    "scores": reports,
                    "review_required": (
                        "Inspect rendered output and provenance before committing this corpus."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    print(f"Materialized {len(assets)} curated score sources at {output}")
    return 0


def _catalog(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("catalog schema_version must be 1")
    upstream, scores = raw.get("upstream"), raw.get("scores")
    if not isinstance(upstream, dict) or not isinstance(scores, list) or len(scores) != 40:
        raise ValueError("catalog must contain upstream metadata and exactly 40 scores")
    required = {"id", "composer", "death_year", "work", "path", "coverage"}
    if any(not isinstance(item, dict) or required - set(item) for item in scores):
        raise ValueError("each catalog score must provide all required curation fields")
    if len({item["id"] for item in scores}) != len(scores):
        raise ValueError("catalog score ids must be unique")
    return raw


def _source_tree(root_path: str, upstream: dict[str, Any], local: Path | None) -> dict[str, bytes]:
    """Fetch a score's main file and every repository-local ``\\include`` recursively."""

    pending = [root_path]
    tree: dict[str, bytes] = {}
    while pending:
        repository_path = pending.pop()
        if repository_path in tree:
            continue
        source = _source_bytes(repository_path, upstream, local)
        tree[repository_path] = source
        parent = PurePosixPath(repository_path).parent
        for include in _INCLUDE_PATTERN.findall(source.decode("utf-8", errors="replace")):
            if include in _LILYPOND_BUILTIN_INCLUDES:
                continue
            include_path = _included_repository_path(parent, include)
            pending.append(str(include_path))
    return tree


def _included_repository_path(parent: PurePosixPath, include: str) -> PurePosixPath:
    candidate = PurePosixPath(include)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsupported unsafe LilyPond include path: {include!r}")
    return parent / candidate


def _source_bytes(path: str, upstream: dict[str, Any], local: Path | None) -> bytes:
    if local is not None:
        return (local / path).read_bytes()
    try:
        with urllib.request.urlopen(_source_url(upstream, path), timeout=30) as response:
            return response.read()
    except urllib.error.URLError as error:
        raise RuntimeError(f"cannot download {path}: {error}") from error


def _source_report(
    identifier: str, source_tree: dict[str, bytes], upstream: dict[str, Any]
) -> dict[str, Any]:
    """Record all copied support files and hashes for review alongside the strict main manifest."""

    return {
        "id": identifier,
        "files": [
            {
                "repository_path": path,
                "source_url": _source_url(upstream, path),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for path, content in sorted(source_tree.items())
        ],
    }


def _source_url(upstream: dict[str, Any], path: str) -> str:
    repository, commit = upstream.get("repository"), upstream.get("commit")
    if not isinstance(repository, str) or not isinstance(commit, str):
        raise ValueError("catalog upstream repository and commit are required")
    return f"https://raw.githubusercontent.com/MutopiaProject/MutopiaProject/{commit}/{path}"


def _assert_public_domain_header(source: bytes, identifier: str) -> None:
    text = source.decode("utf-8", errors="replace").lower()
    if "mutopia" not in text or "public domain" not in text:
        raise ValueError(
            f"{identifier}: upstream source lacks required Mutopia public-domain markers"
        )


if __name__ == "__main__":
    raise SystemExit(main())
