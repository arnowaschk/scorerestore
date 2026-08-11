from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from scorerestore.cli import main

PROJECT_ROOT = Path(__file__).parents[2]


def test_cli_materializes_reproducible_png_and_recipe(tmp_path: Path) -> None:
    source_path = tmp_path / "clean.png"
    first_output = tmp_path / "first.png"
    second_output = tmp_path / "second.png"
    source = Image.new("L", (144, 88), 255)
    draw = ImageDraw.Draw(source)
    for y in range(12, 73, 10):
        draw.line((4, y, 139, y), fill=0, width=1)
    draw.ellipse((50, 34, 58, 41), fill=0)
    source.save(source_path)
    config_path = PROJECT_ROOT / "configs/degradation/medium.yaml"

    assert (
        main(
            [
                "degrade",
                str(source_path),
                "-o",
                str(first_output),
                "-c",
                str(config_path),
                "--seed",
                "1234",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "degrade",
                str(source_path),
                "-o",
                str(second_output),
                "-c",
                str(config_path),
                "--seed",
                "1234",
            ]
        )
        == 0
    )

    first_recipe_path = tmp_path / "first.recipe.json"
    second_recipe_path = tmp_path / "second.recipe.json"
    assert first_output.is_file()
    assert first_recipe_path.is_file()
    assert first_output.stat().st_mode & 0o777 == 0o644
    assert first_recipe_path.stat().st_mode & 0o777 == 0o644
    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_recipe_path.read_bytes() == second_recipe_path.read_bytes()
    with Image.open(first_output) as first, Image.open(second_output) as second:
        assert first.mode == second.mode == "L"
        assert first.size == second.size == source.size
        assert first.tobytes() == second.tobytes()
        assert first.tobytes() != source.tobytes()
    assert json.loads(first_recipe_path.read_text(encoding="utf-8")) == json.loads(
        second_recipe_path.read_text(encoding="utf-8")
    )


def test_cli_default_names_and_builtin_preset(tmp_path: Path) -> None:
    source_path = tmp_path / "page.png"
    Image.new("L", (32, 24), 220).save(source_path)

    assert main(["degrade", str(source_path), "--preset", "light", "--seed", "7"]) == 0

    assert (tmp_path / "page.degraded.png").is_file()
    assert (tmp_path / "page.degraded.recipe.json").is_file()
