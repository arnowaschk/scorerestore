"""Input adapters for ordinary rasters, multipage TIFFs, and PDFs.

Core tiled inference intentionally receives only Pillow/NumPy images. PDF rasterization is isolated
here from the cleaned-PDF output adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageSequence, UnidentifiedImageError


class InputReadError(ValueError):
    """Raised when an unsupported or unreadable inference input is supplied."""


@dataclass(frozen=True, slots=True)
class InputPage:
    """A rasterized input page, retaining input order and known DPI where available."""

    image: Image.Image
    page_number: int
    dpi: int | None


def read_input_pages(path: str | Path, *, pdf_dpi: int = 300) -> tuple[InputPage, ...]:
    """Read PNG/JPEG/TIFF (including multipage TIFF) or rasterize each PDF page in order."""

    source = Path(path)
    if source.suffix.lower() == ".pdf":
        return _read_pdf(source, pdf_dpi)
    try:
        with Image.open(source) as opened:
            dpi = _image_dpi(opened)
            return tuple(
                InputPage(frame.convert("L").copy(), number, dpi)
                for number, frame in enumerate(ImageSequence.Iterator(opened), start=1)
            )
    except (OSError, UnidentifiedImageError) as error:
        raise InputReadError(f"cannot read raster input {source}: {error}") from error


def _read_pdf(path: Path, dpi: int) -> tuple[InputPage, ...]:
    try:
        document = pdfium.PdfDocument(path)
    except Exception as error:
        raise InputReadError(f"cannot read PDF input {path}: {error}") from error
    scale = dpi / 72.0
    pages: list[InputPage] = []
    try:
        for index in range(len(document)):
            page = document[index]
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil().convert("L").copy()
            pages.append(InputPage(image, index + 1, dpi))
    except Exception as error:
        raise InputReadError(f"cannot rasterize PDF input {path}: {error}") from error
    finally:
        document.close()
    return tuple(pages)


def _image_dpi(image: Image.Image) -> int | None:
    raw = image.info.get("dpi")
    if not isinstance(raw, tuple) or not raw:
        return None
    value = raw[0]
    return round(value) if isinstance(value, (int, float)) and value > 0 else None
