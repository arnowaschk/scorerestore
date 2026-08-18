from __future__ import annotations

from pathlib import Path

from PIL import Image

import scorerestore.dataset.generation as generation


def test_progress_reporter_uses_frequently_refreshed_tqdm(monkeypatch) -> None:
    created: list[FakeTqdm] = []

    def create_bar(**kwargs):
        bar = FakeTqdm(kwargs)
        created.append(bar)
        return bar

    monkeypatch.setattr(generation, "tqdm", create_bar)
    reporter = generation._ProgressReporter("Rendering", 640, True)

    reporter.start()
    reporter.advance()
    reporter.advance()
    reporter.close()

    assert created[0].kwargs["total"] == 640
    assert created[0].kwargs["mininterval"] == 1.0
    assert created[0].updates == 2
    assert created[0].closed is True


def test_missing_trainable_masks_identifies_blank_semantic_layers(tmp_path: Path) -> None:
    populated = tmp_path / "populated.png"
    blank = tmp_path / "blank.png"
    image = Image.new("L", (2, 2), 0)
    image.putpixel((0, 0), 255)
    image.save(populated)
    Image.new("L", (2, 2), 0).save(blank)

    missing = generation._missing_trainable_masks({"staff": blank, "notation": populated})

    assert missing == ["staff"]


class FakeTqdm:
    def __init__(self, kwargs: dict[str, object]) -> None:
        self.kwargs = kwargs
        self.updates = 0
        self.closed = False

    def update(self) -> None:
        self.updates += 1

    def close(self) -> None:
        self.closed = True
