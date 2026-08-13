from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from scorerestore.inference.io import read_input_pages
from scorerestore.inference.tiled import (
    _padded_tile,
    _predict_tiled,
    _tile_starts,
    clean,
    load_checkpoint_model,
)
from scorerestore.models import build_model


@dataclass(frozen=True, slots=True)
class _Output:
    cleaning: Tensor
    segmentation: Tensor


class _PointwiseModel(nn.Module):
    """A seam-sensitive pointwise model: tiled logits must equal whole-page logits."""

    def __init__(self) -> None:
        super().__init__()
        self.maximum_shape = (0, 0)

    def forward(self, value: Tensor) -> _Output:
        self.maximum_shape = tuple(
            max(a, b) for a, b in zip(self.maximum_shape, value.shape[-2:], strict=True)
        )
        return _Output(
            cleaning=4 * value - 2,
            segmentation=torch.cat((value, value * 2, value * 3, value * 4), dim=1),
        )


def test_tile_starts_cover_boundaries_without_duplicate_final_tile() -> None:
    assert _tile_starts(32, 32, 8) == (0,)
    assert _tile_starts(79, 32, 8) == (0, 24, 47)


def test_reflection_padding_extends_a_small_page() -> None:
    image = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    tile = _padded_tile(image, 0, 0, 4)

    assert tile.shape == (1, 1, 4, 4)
    assert torch.equal(
        tile[0, 0],
        torch.tensor(
            [[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0], [1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]]
        ),
    )


def test_overlap_blending_has_no_seam_for_pointwise_logits() -> None:
    image = torch.linspace(0, 1, 91 * 75).reshape(1, 1, 91, 75)
    model = _PointwiseModel()
    tiled_clean, tiled_segment, tile_count = _predict_tiled(
        model, image, torch.device("cpu"), 32, 8
    )
    direct = model(image)

    assert tile_count > 1
    assert torch.allclose(tiled_clean, direct.cleaning, atol=1e-6)
    assert torch.allclose(tiled_segment, direct.segmentation, atol=1e-6)


def test_clean_preserves_dimensions_and_bounds_large_page_memory() -> None:
    image = np.full((701, 983), 180, dtype=np.uint8)
    model = _PointwiseModel()
    result = clean(image, model=model, device="cpu", tile_size=128, overlap=32)

    assert result.cleaned.size == (983, 701)
    assert result.probability.size == (983, 701)
    assert {mask.size for mask in result.masks.values()} == {(983, 701)}
    assert set(np.unique(np.asarray(result.cleaned))) <= {0, 255}
    assert model.maximum_shape == (128, 128)
    assert result.metadata["tile_count"] > 1


def test_checkpoint_model_round_trip(tmp_path: Path) -> None:
    original = build_model(base_channels=2)
    checkpoint_path = tmp_path / "weights.pt"
    torch.save(
        {
            "model_state_dict": original.state_dict(),
            "config": {
                "model_backend": "unet",
                "base_channels": 2,
                "pretrained": False,
                "freeze_batch_norm": True,
            },
            "epoch": 1,
            "validation_loss": 0.2,
        },
        checkpoint_path,
    )

    loaded, metadata = load_checkpoint_model(checkpoint_path, device="cpu")
    result = clean(
        Image.new("L", (64, 64), 200), model=loaded, device="cpu", tile_size=64, overlap=0
    )

    assert metadata["checkpoint_epoch"] == 1
    assert result.cleaned.size == (64, 64)


def test_original_custom_unet_checkpoint_without_transfer_fields_still_loads(
    tmp_path: Path,
) -> None:
    original = build_model(base_channels=2)
    checkpoint_path = tmp_path / "legacy-weights.pt"
    torch.save(
        {
            "model_state_dict": original.state_dict(),
            "config": {"base_channels": 2, "task": "multitask"},
            "epoch": 1,
            "validation_loss": 0.2,
        },
        checkpoint_path,
    )

    _, metadata = load_checkpoint_model(checkpoint_path, device="cpu")

    assert metadata["training_config"]["model_backend"] == "unet"
    assert metadata["training_config"]["pretrained"] is False


def test_input_adapters_preserve_multipage_tiff_order_and_rasterize_pdf(tmp_path: Path) -> None:
    first, second = Image.new("L", (17, 11), 10), Image.new("L", (17, 11), 230)
    tiff = tmp_path / "pages.tiff"
    first.save(tiff, save_all=True, append_images=[second])
    tiff_pages = read_input_pages(tiff)
    pdf = tmp_path / "pages.pdf"
    first.save(pdf, save_all=True, append_images=[second], resolution=72)
    pdf_pages = read_input_pages(pdf, pdf_dpi=72)

    assert [page.page_number for page in tiff_pages] == [1, 2]
    assert [np.asarray(page.image).mean() for page in tiff_pages] == [10, 230]
    assert [page.page_number for page in pdf_pages] == [1, 2]
    assert [page.image.size for page in pdf_pages] == [(17, 11), (17, 11)]
