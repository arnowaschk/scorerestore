from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageOps

from scorerestore.cli import main
from scorerestore.lilypond.constants import LILYPOND_VERSION
from scorerestore.lilypond.renderer import (
    LilyPondRenderConfig,
    LilyPondRenderError,
    LilyPondRenderResult,
    detect_lilypond_version,
    render_score,
)
from scorerestore.provenance import validate_score_manifest

PROJECT_ROOT = Path(__file__).parents[2]
FIXTURE = PROJECT_ROOT / "tests/fixtures/lilypond/overlap.ly"
MANIFEST = PROJECT_ROOT / "assets/scores/manifest.yaml"


@pytest.fixture(scope="session")
def lilypond_binary() -> Path:
    configured = os.environ.get("SCORERESTORE_TEST_LILYPOND")
    candidate = Path(configured) if configured else Path(shutil.which("lilypond") or "")
    if not candidate.is_file():
        pytest.skip("LilyPond is not installed")
    try:
        version = detect_lilypond_version(candidate)
    except LilyPondRenderError as error:
        pytest.skip(str(error))
    if version != LILYPOND_VERSION:
        pytest.skip(f"integration tests require LilyPond {LILYPOND_VERSION}, found {version}")
    return candidate


@pytest.fixture(scope="session")
def rendered_fixture(
    tmp_path_factory: pytest.TempPathFactory, lilypond_binary: Path
) -> LilyPondRenderResult:
    output = tmp_path_factory.mktemp("lilypond-render") / "result"
    expected_sha256 = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    return render_score(
        FIXTURE,
        output,
        config=LilyPondRenderConfig(
            lilypond_binary=lilypond_binary,
            dpi=96,
            strict_unknown_grobs=True,
        ),
        expected_source_sha256=expected_sha256,
    )


def test_ordinary_lilypond_file_renders_all_aligned_passes(
    rendered_fixture: LilyPondRenderResult,
) -> None:
    assert rendered_fixture.lilypond_version == LILYPOND_VERSION
    assert rendered_fixture.source_hash_verified is True
    assert rendered_fixture.unknown_grobs == ()
    assert len(rendered_fixture.pages) == 1
    page = rendered_fixture.pages[0]
    required_paths = {
        page.pristine_path,
        page.qa_panel_path,
        *page.mask_paths.values(),
    }
    assert all(path.is_file() for path in required_paths)

    with Image.open(page.pristine_path) as pristine:
        expected_size = pristine.size
    assert expected_size == (page.width, page.height)
    for path in page.mask_paths.values():
        with Image.open(path) as mask:
            assert mask.size == expected_size


def test_staff_and_notation_masks_overlap_without_background_overlap(
    rendered_fixture: LilyPondRenderResult,
) -> None:
    page = rendered_fixture.pages[0]
    with (
        Image.open(page.mask_paths["staff"]) as staff,
        Image.open(page.mask_paths["notation"]) as notation,
        Image.open(page.mask_paths["text"]) as text,
        Image.open(page.mask_paths["background"]) as background,
    ):
        assert ImageChops.multiply(staff, notation).getbbox() is not None
        foreground_union = ImageChops.lighter(staff, ImageChops.lighter(notation, text))
        assert (
            ImageChops.difference(background, ImageOps.invert(foreground_union)).getbbox() is None
        )
        assert ImageChops.multiply(background, foreground_union).getbbox() is None


def test_visual_qa_panel_and_metadata_are_materialized(
    rendered_fixture: LilyPondRenderResult,
) -> None:
    page = rendered_fixture.pages[0]
    with Image.open(page.qa_panel_path) as panel:
        assert panel.width > page.width
        assert panel.height > 0
    metadata = json.loads(rendered_fixture.metadata_path.read_text(encoding="utf-8"))
    assert metadata["lilypond_version"] == LILYPOND_VERSION
    assert metadata["pages"][0]["qa"] == {
        "background_foreground_overlap_pixels": 0,
        "background_is_foreground_complement": True,
        "dimensions_match": True,
    }


def test_bundled_legacy_source_renders_without_editing_source(
    tmp_path: Path, lilypond_binary: Path
) -> None:
    report = validate_score_manifest(MANIFEST)
    asset = next(item for item in report.assets if item.id == "bach-bwv773-invention-02")
    before = hashlib.sha256(asset.source_path.read_bytes()).hexdigest()

    result = render_score(
        asset.source_path,
        tmp_path / "bach",
        config=LilyPondRenderConfig(
            lilypond_binary=lilypond_binary,
            dpi=48,
            strict_unknown_grobs=True,
        ),
        expected_source_sha256=asset.source_sha256,
    )

    assert len(result.pages) == 2
    assert result.source_hash_verified is True
    assert hashlib.sha256(asset.source_path.read_bytes()).hexdigest() == before


def test_source_hash_mismatch_is_rejected(tmp_path: Path, lilypond_binary: Path) -> None:
    with pytest.raises(LilyPondRenderError, match="source SHA-256 mismatch"):
        render_score(
            FIXTURE,
            tmp_path / "mismatch",
            config=LilyPondRenderConfig(lilypond_binary=lilypond_binary, dpi=48),
            expected_source_sha256="0" * 64,
        )


def test_inspect_commands_report_environment_and_generate_panel(
    tmp_path: Path,
    lilypond_binary: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["inspect", "environment", "--lilypond", str(lilypond_binary)]) == 0
    environment = json.loads(capsys.readouterr().out)
    assert environment["lilypond"]["matches_required"] is True

    output = tmp_path / "cli-masks"
    assert (
        main(
            [
                "inspect",
                "masks",
                str(FIXTURE),
                "-o",
                str(output),
                "--lilypond",
                str(lilypond_binary),
                "--dpi",
                "48",
                "--strict-unknown-grobs",
            ]
        )
        == 0
    )
    assert (output / "page-001/qa.png").is_file()
