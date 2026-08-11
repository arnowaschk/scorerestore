from __future__ import annotations

from pathlib import Path

import yaml

from scorerestore.lilypond.constants import (
    LILYPOND_LINUX_X86_64_SHA256,
    LILYPOND_VERSION,
)

PROJECT_ROOT = Path(__file__).parents[2]


def test_compose_has_one_cpu_service_and_canonical_mounts() -> None:
    compose = yaml.safe_load((PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8"))

    assert set(compose["services"]) == {"scorerestore"}
    service = compose["services"]["scorerestore"]
    assert service["image"] == "scorerestore:0.1.0"
    assert set(service["volumes"]) == {
        "./data:/data",
        "./models:/models",
        "./runs:/runs",
    }
    assert "gpus" not in service


def test_gpu_override_only_enables_gpu_access() -> None:
    override = yaml.safe_load((PROJECT_ROOT / "compose.gpu.yaml").read_text(encoding="utf-8"))

    assert override == {"services": {"scorerestore": {"gpus": "all"}}}


def test_dockerfile_uses_versioned_images() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM ghcr.io/astral-sh/uv:0.11.20 AS uv" in dockerfile
    assert "FROM python:3.12.11-slim-bookworm" in dockerfile
    assert f"ARG LILYPOND_VERSION={LILYPOND_VERSION}" in dockerfile
    assert f"ARG LILYPOND_SHA256={LILYPOND_LINUX_X86_64_SHA256}" in dockerfile
    assert "COPY --from=lilypond /opt/lilypond-2.26.0" in dockerfile
    assert "COPY assets ./assets" in dockerfile
    assert ":latest" not in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile


def test_private_and_generated_paths_are_ignored() -> None:
    ignore_lines = set((PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())

    assert {"/SPEC.md", "/.local/", "/data/*", "/models/*", "/runs/*"} <= ignore_lines


def test_private_paths_are_excluded_from_docker_context() -> None:
    ignore_lines = set((PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())

    assert {".git", ".local", ".venv", "SPEC.md"} <= ignore_lines
