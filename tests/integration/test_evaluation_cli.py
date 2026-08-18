from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from scorerestore.cli import main
from scorerestore.dataset.manifest import sha256_file
from scorerestore.models import build_model


def test_evaluate_keeps_challenge_separate_and_writes_deterministic_artifacts(
    tmp_path: Path,
) -> None:
    manifest, checkpoint, config = _evaluation_fixture(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"

    assert main(["evaluate", "-c", str(config), "-o", str(first)]) == 0
    assert main(["evaluate", "-c", str(config), "-o", str(second)]) == 0

    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["splits"] == ["validation", "test", "challenge"]
    assert summary["models"]["clean_unet"]["splits"]["test"]["sample_count"] == 1
    assert summary["models"]["clean_unet"]["splits"]["challenge"]["sample_count"] == 1
    assert "non_challenge" not in summary["models"]["clean_unet"]
    comparison = summary["comparisons"]["clean_unet_vs_multitask_unet"]
    assert comparison["comparison_type"] == "cleaning_only_vs_multitask_custom_unet"
    assert comparison["controlled"] is True
    assert (
        summary["report_selection"]
        == json.loads((second / "summary.json").read_text(encoding="utf-8"))["report_selection"]
    )

    metrics = (first / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert len(metrics) == 10  # Three actual OpenCV rows and three rows per neural model.
    assert "cleaning_ssim" in metrics[0]
    assert "segmentation_notation_dice" in metrics[0]
    assert "segmentation_foreground_macro_dice" in metrics[0]
    assert "background_overlap_rate" in metrics[0]
    assert "accuracy" not in metrics[0]
    assert (first / "report/summary.md").is_file()
    visual = next((first / "comparisons/clean_unet").glob("*.png"))
    matching = second / "comparisons/clean_unet" / visual.name
    assert sha256_file(visual) == sha256_file(matching)

    assert checkpoint.is_file()
    assert manifest.is_file()
    assert main(["evaluate", "-c", str(config), "-o", str(first), "--update"]) == 0


def test_benchmark_writes_only_actual_measurements(tmp_path: Path) -> None:
    _, _, config = _evaluation_fixture(tmp_path)
    input_path = tmp_path / "dataset/inputs/page.png"
    output = tmp_path / "benchmark.json"

    assert (
        main(
            [
                "benchmark",
                str(input_path),
                "-c",
                str(config),
                "-o",
                str(output),
                "--model",
                "clean_unet",
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["label"] == "MEASURED"
    assert report["hardware"]["device"] == "cpu"
    assert report["tile_size"] == 64
    assert report["overlap"] == 16
    assert report["precision_mode"] == "float32"
    assert report["pages"][0]["dimensions"] == {"width": 64, "height": 64}
    assert report["total_latency_seconds"] > 0
    assert report["total_megapixels_per_second"] > 0
    assert report["peak_gpu_memory_bytes"] is None
    assert (
        main(
            [
                "benchmark",
                str(input_path),
                "-c",
                str(config),
                "-o",
                str(output),
                "--model",
                "clean_unet",
                "--update",
            ]
        )
        == 0
    )


def _evaluation_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "dataset"
    for directory in (
        dataset / "inputs",
        dataset / "clean",
        dataset / "masks/background",
        dataset / "masks/staff",
        dataset / "masks/notation",
        dataset / "masks/text",
        dataset / "manifests/recipes",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    input_path = dataset / "inputs/page.png"
    clean_path = dataset / "clean/page.png"
    _page(210).save(input_path)
    _page(255).save(clean_path)
    masks = _masks()
    mask_paths: dict[str, str] = {}
    for name, image in masks.items():
        path = dataset / f"masks/{name}/page.png"
        image.save(path)
        mask_paths[name] = str(path.relative_to(dataset))
    recipe_path = dataset / "manifests/recipes/page.json"
    recipe_path.write_text("{}\n", encoding="utf-8")

    records = []
    for split in ("validation", "test", "challenge"):
        records.append(
            {
                "schema_version": 1,
                "sample_id": f"sample-{split}",
                "dataset_id": "evaluation-fixture",
                "source_id": f"source-{split}",
                "source_path": f"sources/{split}.ly",
                "source_sha256": "0" * 64,
                "source_license_status": "public_domain",
                "source_provenance": _provenance(),
                "page": 1,
                "split": split,
                "generator_version": "0.1.0",
                "lilypond_version": "2.26.0",
                "seed": 1,
                "render_parameters": _render_parameters(),
                "degradation_preset": "light",
                "degradations": [],
                "input_path": str(input_path.relative_to(dataset)),
                "clean_target_path": str(clean_path.relative_to(dataset)),
                "mask_paths": mask_paths,
                "recipe_path": str(recipe_path.relative_to(dataset)),
                "hashes": {
                    "input": sha256_file(input_path),
                    "clean": sha256_file(clean_path),
                    "recipe": sha256_file(recipe_path),
                    **{
                        f"mask_{name}": sha256_file(dataset / relative_path)
                        for name, relative_path in mask_paths.items()
                    },
                },
                "dimensions": {"width": 64, "height": 64},
                "dpi": 300,
                "created_at": "2026-08-13T00:00:00Z",
            }
        )
    manifest = dataset / "manifests/samples.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    (dataset / "manifests/dataset.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "evaluation-fixture",
                "sample_count": len(records),
                "source_splits": {record["source_id"]: record["split"] for record in records},
            }
        ),
        encoding="utf-8",
    )

    torch.manual_seed(1)
    checkpoint = tmp_path / "weights.pt"
    model = build_model(base_channels=2)
    training = {
        "dataset_manifest": str(manifest),
        "model_backend": "unet",
        "base_channels": 2,
        "pretrained": False,
        "freeze_batch_norm": True,
        "crop_size": 64,
        "train_crops_per_epoch": 3,
        "validation_crops": 3,
        "foreground_fraction": 0.8,
        "minimum_foreground_occupancy": 0.01,
        "batch_size": 1,
        "epochs": 5,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "gradient_accumulation": 1,
        "early_stopping_patience": None,
        "device": "cpu",
        "seed": 1,
        "num_workers": 0,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {**training, "task": "clean"},
            "epoch": 1,
            "validation_loss": 0.2,
        },
        checkpoint,
    )
    multitask = tmp_path / "multitask.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {**training, "task": "multitask"},
            "epoch": 1,
            "validation_loss": 0.2,
        },
        multitask,
    )
    config = tmp_path / "evaluation.yaml"
    config.write_text(
        "\n".join(
            (
                f"dataset_manifest: {manifest}",
                "models:",
                f"  - name: clean_unet\n    checkpoint: {checkpoint}",
                f"  - name: multitask_unet\n    checkpoint: {multitask}",
                "splits: [validation, test, challenge]",
                "inference:",
                "  device: cpu",
                "  tile_size: 64",
                "  overlap: 16",
                "  pdf_dpi: 300",
                "  cleaning_threshold: 0.5",
                "  segmentation_threshold: 0.5",
                "baseline:",
                "  config: configs/baseline.yaml",
                "  variant: otsu",
                "report:",
                "  seed: 9",
                "  samples: 2",
                "",
            )
        ),
        encoding="utf-8",
    )
    return manifest, checkpoint, config


def _page(background: int) -> Image.Image:
    image = Image.new("L", (64, 64), background)
    ImageDraw.Draw(image).rectangle((16, 16, 48, 22), fill=0)
    return image


def _masks() -> dict[str, Image.Image]:
    background = Image.new("L", (64, 64), 255)
    staff = Image.new("L", (64, 64), 0)
    notation = Image.new("L", (64, 64), 0)
    text = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(staff).line((8, 20, 56, 20), fill=255, width=1)
    ImageDraw.Draw(notation).ellipse((28, 24, 36, 32), fill=255)
    ImageDraw.Draw(text).rectangle((12, 40, 30, 45), fill=255)
    return {"background": background, "staff": staff, "notation": notation, "text": text}


def _provenance() -> dict[str, object]:
    rights = {"status": "public_domain", "basis": "fixture"}
    return {
        "source_url": "https://example.invalid/fixture",
        "composition_rights": rights,
        "source_file_rights": rights,
        "verified_date": "2026-08-13",
    }


def _render_parameters() -> dict[str, object]:
    return {
        "staff_size": 16.0,
        "paper_format": "a4",
        "orientation": "portrait",
        "margins_mm": {"top": 10, "right": 10, "bottom": 10, "left": 10},
        "layout_seed": 1,
        "dpi": 300,
        "mask_threshold": 0.5,
        "strict_unknown_grobs": True,
    }
