from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from scorerestore.cli import main
from scorerestore.inference import read_input_pages
from scorerestore.models import build_model


def test_compare_real_world_auto_selects_models_and_preserves_full_resolution(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "real_world"
    source_root.mkdir()
    source = source_root / "example.pdf"
    first, second = Image.new("L", (64, 64), 190), Image.new("L", (64, 64), 220)
    first.save(source, format="PDF", save_all=True, append_images=[second], resolution=72)
    runs_root = tmp_path / "runs"
    _checkpoint(runs_root / "custom-worse/checkpoints/best.pt", "unet", 0.9)
    chosen = runs_root / "custom-best/checkpoints/best.pt"
    _checkpoint(chosen, "unet", 0.1)
    _checkpoint(runs_root / "resnet/checkpoints/best.pt", "resnet18", 0.2)
    config = _comparison_config(tmp_path, source_root, runs_root)
    output = tmp_path / "comparison"

    assert (
        main(
            [
                "compare-real-world",
                "-o",
                str(output),
                "-c",
                str(config),
            ]
        )
        == 0
    )

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["models"]["model_cleaned"]["selection"] == {
        "backend": "unet",
        "path": str(chosen.resolve()),
        "selection": "automatic_lowest_validation_loss",
        "validation_loss": 0.1,
    }
    assert metadata["models"]["resnet_cleaned"]["selection"]["backend"] == "resnet18"
    assert metadata["inference"]["resampling"] == "none; panels retain native raster dimensions"
    assert metadata["inference"]["classical_baseline"]["variant"] == "otsu"
    assert metadata["inference"]["panel_layout"] == [
        "Original",
        "OpenCV classical cleaned",
        "ResNet-18 cleaned",
        "Custom model cleaned",
    ]
    assert len(metadata["pages"]) == 2
    for page_number in (1, 2):
        for directory in ("original", "classical_cleaned", "resnet_cleaned", "model_cleaned"):
            page = output / directory / "example" / f"page-{page_number:04d}.png"
            with Image.open(page) as image:
                assert image.size == (64, 64)

    sheets = read_input_pages(output / "comparison.pdf", pdf_dpi=72)
    assert len(sheets) == 2
    assert all(page.image.size == (256, 92) for page in sheets)


def test_compare_real_world_allows_custom_checkpoint_override(tmp_path: Path) -> None:
    source_root = tmp_path / "real_world"
    source_root.mkdir()
    Image.new("L", (64, 64), 190).save(source_root / "example.pdf", format="PDF", resolution=72)
    runs_root = tmp_path / "runs"
    custom = runs_root / "custom/checkpoints/best.pt"
    _checkpoint(custom, "unet", 0.5)
    resnet = runs_root / "resnet/checkpoints/best.pt"
    _checkpoint(resnet, "resnet18", 0.2)
    override = tmp_path / "override.pt"
    _checkpoint(override, "unet", 1.0)
    config = _comparison_config(tmp_path, source_root, runs_root)
    output = tmp_path / "comparison"

    assert (
        main(
            [
                "compare-real-world",
                "-o",
                str(output),
                "-c",
                str(config),
                "--checkpoint",
                f"model_cleaned={override}",
            ]
        )
        == 0
    )
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["models"]["model_cleaned"]["selection"]["path"] == str(override.resolve())
    assert metadata["models"]["model_cleaned"]["selection"]["selection"] == "explicit_override"


def test_compare_real_world_places_each_yaml_model_in_order(tmp_path: Path) -> None:
    source_root = tmp_path / "real_world"
    source_root.mkdir()
    Image.new("L", (64, 64), 190).save(source_root / "example.pdf", format="PDF", resolution=72)
    runs_root = tmp_path / "runs"
    custom = runs_root / "custom/checkpoints/best.pt"
    resnet = runs_root / "resnet/checkpoints/best.pt"
    second_custom = runs_root / "second-custom/checkpoints/best.pt"
    _checkpoint(custom, "unet", 0.5)
    _checkpoint(resnet, "resnet18", 0.2)
    _checkpoint(second_custom, "unet", 0.6)
    config = _comparison_config(
        tmp_path,
        source_root,
        runs_root,
        additional_models=(("training_demo_2", "Training demo 2 cleaned", "unet", second_custom),),
    )
    output = tmp_path / "comparison"

    assert main(["compare-real-world", "-o", str(output), "-c", str(config)]) == 0

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["inference"]["panel_layout"][-1] == "Training demo 2 cleaned"
    assert metadata["pages"][0]["models"]["training_demo_2"]["cleaned"].startswith(
        "training_demo_2/"
    )
    sheet = read_input_pages(output / "comparison.pdf", pdf_dpi=72)
    assert sheet[0].image.size == (320, 92)


def _comparison_config(
    tmp_path: Path,
    source_root: Path,
    runs_root: Path,
    *,
    additional_models: tuple[tuple[str, str, str, Path], ...] = (),
) -> Path:
    config = tmp_path / "comparison.yaml"
    lines = [
        f"input_root: {source_root}",
        f"runs_root: {runs_root}",
        "inference:",
        "  device: cpu",
        "  tile_size: 64",
        "  overlap: 16",
        "  pdf_dpi: 72",
        "  cleaning_threshold: 0.5",
        "  segmentation_threshold: 0.5",
        "classical:",
        "  config: configs/baseline.yaml",
        "  variant: otsu",
        "models:",
        "  - id: resnet_cleaned",
        "    label: ResNet-18 cleaned",
        "    backend: resnet18",
        "    checkpoint: auto",
        "  - id: model_cleaned",
        "    label: Custom model cleaned",
        "    backend: unet",
        "    checkpoint: auto",
    ]
    for identifier, label, backend, checkpoint in additional_models:
        lines.extend(
            (
                f"  - id: {identifier}",
                f"    label: {label}",
                f"    backend: {backend}",
                f"    checkpoint: {checkpoint}",
            )
        )
    lines.append("")
    config.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return config


def _checkpoint(path: Path, backend: str, validation_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1)
    model = build_model(backend, base_channels=2, pretrained=False, freeze_batch_norm=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "task": "multitask",
                "model_backend": backend,
                "base_channels": 2,
                "pretrained": False,
                "freeze_batch_norm": True,
            },
            "epoch": 1,
            "validation_loss": validation_loss,
        },
        path,
    )
