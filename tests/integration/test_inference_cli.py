from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from scorerestore.cli import main
from scorerestore.models import build_model


def test_infer_cli_writes_exact_dimension_page_outputs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    model = build_model(base_channels=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_backend": "unet",
                "base_channels": 2,
                "pretrained": False,
                "freeze_batch_norm": True,
            },
            "epoch": 1,
            "validation_loss": 0.2,
        },
        checkpoint,
    )
    config = tmp_path / "infer.yaml"
    config.write_text(
        "\n".join(
            (
                f"checkpoint: {checkpoint}",
                "device: cpu",
                "tile_size: 32",
                "overlap: 8",
                "cleaning_threshold: 0.5",
                "segmentation_threshold: 0.5",
                "pdf_dpi: 72",
                "overlay: true",
                "",
            )
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.png"
    Image.new("L", (71, 43), 190).save(source)
    output = tmp_path / "output"

    assert main(["infer", str(source), "-c", str(config), "-o", str(output)]) == 0

    metadata = json.loads((output / "page-0001/metadata.json").read_text())
    assert metadata["inference"]["input_dimensions"] == {"width": 71, "height": 43}
    for name in ("cleaned", "cleaning_probability", "staff_mask", "staff_probability", "overlay"):
        with Image.open(output / metadata["outputs"][name]) as result:
            assert result.size == (71, 43)


def test_infer_cli_preserves_multipage_tiff_and_pdf_order(tmp_path: Path) -> None:
    checkpoint, config = _checkpoint_and_config(tmp_path)
    del checkpoint
    first, second = Image.new("L", (37, 35), 20), Image.new("L", (37, 35), 220)
    tiff = tmp_path / "input.tiff"
    first.save(tiff, save_all=True, append_images=[second])
    pdf = tmp_path / "input.pdf"
    first.save(pdf, save_all=True, append_images=[second], resolution=72)

    for source, name in ((tiff, "tiff"), (pdf, "pdf")):
        output = tmp_path / name
        assert main(["infer", str(source), "-c", str(config), "-o", str(output)]) == 0
        run_metadata = json.loads((output / "metadata.json").read_text())
        assert len(run_metadata["pages"]) == 2
        for page_number in (1, 2):
            metadata = json.loads((output / f"page-{page_number:04d}/metadata.json").read_text())
            assert metadata["page_number"] == page_number
            with Image.open(output / metadata["outputs"]["cleaned"]) as result:
                assert result.size == (37, 35)


def _checkpoint_and_config(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal trained-model shaped checkpoint and CPU inference config."""

    checkpoint = tmp_path / "weights.pt"
    model = build_model(base_channels=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_backend": "unet",
                "base_channels": 2,
                "pretrained": False,
                "freeze_batch_norm": True,
            },
            "epoch": 1,
            "validation_loss": 0.2,
        },
        checkpoint,
    )
    config = tmp_path / "infer.yaml"
    config.write_text(
        "\n".join(
            (
                f"checkpoint: {checkpoint}",
                "device: cpu",
                "tile_size: 32",
                "overlap: 8",
                "cleaning_threshold: 0.5",
                "segmentation_threshold: 0.5",
                "pdf_dpi: 72",
                "overlay: true",
                "",
            )
        ),
        encoding="utf-8",
    )
    return checkpoint, config
