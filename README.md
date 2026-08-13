# ScoreRestore

**ScoreRestore — Deep-learning restoration and semantic analysis of scanned sheet music**

ScoreRestore V1 is a proof-of-concept project for cleaning degraded sheet-music pages and
predicting independent background, staff, notation, and text masks. **ScoreRestore is not an OMR
system.**

The repository implements a small rights-cleared score-source corpus, exact semantic LilyPond mask
rendering, reusable pixel-aligned synthetic degradation, materialized dataset generation with exact
sample reproduction, neural training, tiled inference, and measured evaluation reports. No quality
or performance claim is made unless it is backed by a generated report labelled **MEASURED**.

Validate the bundled score sources, rights declarations, and SHA-256 hashes:

```bash
uv run scorerestore inspect provenance
```

Inspect the renderer environment and render a score into aligned masks plus a visual QA panel:

```bash
uv run scorerestore inspect environment
uv run scorerestore inspect masks tests/fixtures/lilypond/overlap.ly -o /tmp/score-masks
```

Rendering requires exactly LilyPond 2.26.0. The Docker image installs and checksums that release;
native installations with another LilyPond version are reported but refused for rendering.

Canonical Docker rendering writes inspectable results through the `/runs` bind mount:

```bash
docker compose build
docker compose run --rm scorerestore scorerestore inspect environment
docker compose run --rm scorerestore scorerestore inspect masks \
  assets/scores/sources/bach-bwv773-invention-02.ly \
  -o /runs/milestone-2-bach --strict-unknown-grobs
```

## Synthetic degradation

Milestone 3 implements exactly five composable V1 families: Gaussian blur, Gaussian image noise,
smooth uneven illumination/shadows, JPEG artifacts, and procedural stains/speckles. The `light`,
`medium`, `heavy`, and seeded `random` presets retain the original pixel dimensions and never apply
rotation, skew, perspective, or page warping. `heavy` is intentionally bounded to keep notation
recoverable.

CLI usage writes a lossless degraded PNG and a sibling machine-readable recipe:

```bash
uv run scorerestore degrade input.png -o degraded.png \
  -c configs/degradation/medium.yaml --seed 1234
```

The recipe records the source/output pixel hashes, resolved severity bounds, actual operation
selection and order, sampled parameters, master seed, per-operation seeds, dimensions, and relevant
software versions. The same normalized image, configuration, environment, and seed produce the same
pixels and recipe.

The degradation subsystem is also a standalone public Python API and does not import training code:

```python
from PIL import Image
from scorerestore import degrade

source = Image.open("input.png")
result = degrade(source, config="medium", seed=1234)
result.image.save("degraded.png")
print(result.recipe)
print(result.metadata)
```

ScoreRestore normalizes degradation inputs to grayscale intensity (`0 = black`, `255 = white`).
Preset YAML files live under `configs/degradation/` and may be copied and adjusted within the
validated recoverable V1 bounds.

Docker Compose usage:

```bash
docker compose build
docker compose run --rm scorerestore scorerestore degrade \
  /runs/input.png -o /runs/degraded.png \
  -c configs/degradation/medium.yaml --seed 1234
```

## Materialized datasets

Milestone 4 assigns each underlying musical source to one deterministic
train/validation/test/challenge split before rendering or degradation. Every layout and degraded
variant of that source stays in the same split. Layout generation deterministically varies staff
size, A4/Letter paper, portrait/landscape orientation, and modest margins; the exact values are
recorded in each JSONL sample record.

Generate and validate the two-sample native smoke dataset:

```bash
uv run scorerestore generate \
  -c configs/dataset/smoke.yaml \
  --output-root data/generated
uv run scorerestore dataset validate \
  data/generated/scorerestore-smoke-v1/manifests/samples.jsonl
```

Generation refuses to replace an existing dataset directory. Remove or choose a different output
root deliberately before regenerating. The canonical `demo.yaml` targets 1,000 degraded page
samples; `challenge.yaml` keeps a separate recoverable challenge set using the same five V1
degradation families. These larger configurations are intentionally not part of the quick smoke
command.

Each dataset contains degraded inputs, pristine targets, four independent semantic masks, render
and QA reports, per-sample degradation recipes, `manifests/samples.jsonl`, and resolved dataset
metadata. Validate one sample by regenerating its source, layout, masks, and degradation:

```bash
uv run scorerestore dataset reproduce SAMPLE_ID \
  --dataset-id scorerestore-smoke-v1 \
  --data-root data/generated \
  -o /tmp/reproduced.png
```

Matching LilyPond, ScoreRestore, Python, Pillow, and NumPy versions produce an exact hash check.
Compatible version differences are reported as best-effort rather than silently presented as
exact.

The lightweight Python loader opens the degraded input and all five targets without introducing a
training-framework dependency before the training milestones:

```python
from scorerestore.dataset import MaterializedDataset

dataset = MaterializedDataset("data/generated/scorerestore-smoke-v1/manifests/samples.jsonl")
sample = dataset[0]
print(sample.image.size, sample.clean.size, sorted(sample.masks))
```

Docker Compose uses the same configs and writes through the `/data` mount:

```bash
docker compose build
docker compose run --rm scorerestore scorerestore generate \
  -c configs/dataset/smoke.yaml --output-root /data/generated
docker compose run --rm scorerestore scorerestore dataset validate \
  /data/generated/scorerestore-smoke-v1/manifests/samples.jsonl
```

## Classical cleaning baseline

Milestone 5 provides a deliberately understandable, non-deep-learning reference pipeline:

```text
grayscale → smooth illumination estimate → normalization → Otsu/adaptive threshold
          → optional light morphology → binary cleaned PNG
```

Every run evaluates the same samples with four fixed variants: Otsu, adaptive thresholding, Otsu
plus morphology, and adaptive thresholding plus the same morphology. The shared defaults are a
fixed, untuned starting point and were not adjusted using test results. The default light
morphology is a 3x3 open-then-close cleanup with one iteration per operation; bounded `open`,
`close`, or `open_close` remains configurable through strict YAML.

Run the baseline over a generated smoke dataset:

```bash
uv run scorerestore baseline \
  data/generated/scorerestore-smoke-v1/manifests/samples.jsonl \
  -c configs/baseline.yaml \
  -o runs/baseline-smoke
```

Configuration remains YAML-first. Repeat `--set FIELD=VALUE` for explicit overrides:

```bash
uv run scorerestore baseline \
  data/generated/scorerestore-smoke-v1/manifests/samples.jsonl \
  -c configs/baseline.yaml \
  -o runs/baseline-open \
  --set morphology.operation=open
```

Each run saves four binary cleaned PNGs per sample under `results/<variant>/<split>/`, plus resolved
`config.yaml`, `environment.json`, variant-labelled per-result `metrics.jsonl` and `metrics.csv`,
and a per-variant `summary.json`. Cleaning evaluation reports foreground precision, recall,
F1/Dice, IoU, and scale-adaptive SSIM. Overall pixel accuracy is deliberately omitted because
white page background would dominate it. Split summaries remain separate, especially challenge
versus test.

Docker Compose uses the same interface and writes results through `/runs`:

```bash
docker compose run --rm scorerestore scorerestore baseline \
  /data/generated/scorerestore-smoke-v1/manifests/samples.jsonl \
  -c configs/baseline.yaml \
  -o /runs/baseline-smoke
```

The command computes real metrics for the selected dataset. Neural and baseline comparisons are
produced by the measured evaluation workflow below rather than copied into this README as an
unverified static table.

## Custom U-Net training

Milestone 6 adds the readable, in-repository PyTorch U-Net: a shared four-level encoder/decoder,
bilinear skip connections, GroupNorm, a one-channel ink-coverage cleaning head, and a four-channel
independent-sigmoid semantic head in the fixed `background, staff, notation, text` order. The
same model supports `clean`, `segment`, and `multitask`; inactive tasks do not contribute loss or
checkpoint selection. Clean targets preserve antialiased ink coverage (`1 = ink`) while the input
remains grayscale intensity (`0 = black`, `1 = white`).

The foreground-aware crop sampler targets 80% occupied crops by default and uses uniform crops for
the remaining 20%; no mirrored score augmentation is used. `configs/training/default.yaml` uses
the V1 default 1024px crop and a 32-channel model, with conservative 4060 Ti 16 GB-oriented batch
and accumulation settings. The `smoke.yaml` configuration uses a tiny model and 64px crops solely
to validate CPU forward/backward/checkpoint paths. It falls back
to the training split for validation only when a deliberately tiny dataset has no validation split.

```bash
uv run scorerestore train -c configs/training/default.yaml -o runs/training-demo
uv run scorerestore train -c configs/training/short.yaml -o runs/training-short
uv run scorerestore train -c configs/training/smoke.yaml -o runs/training-smoke
uv run scorerestore train -c configs/training/resnet18.yaml -o runs/training-resnet18
uv run scorerestore train -c configs/training/smoke.yaml -o runs/training-clean --set task=clean
uv run scorerestore train -c configs/training/smoke.yaml -o runs/training-segment --set task=segment
```

For a real run, use a materialized dataset with a source-isolated validation split and set
`training.device=auto` (CUDA is selected when available). CUDA uses AMP; CPU remains fully
supported. Each run creates resolved configuration and environment provenance, append-only JSONL
metrics, CSV metrics, checkpoint(s), and reserved plots/comparisons/report directories. The run
records actual environment values only; unavailable Git metadata is recorded as `null` rather than
guessed. CUDA uses BF16 AMP when the GPU supports it (including the RTX 4060 Ti), otherwise FP16
with gradient scaling. This milestone does not yet provide tiled inference, final evaluation
reports, or quality claims.

### Transfer-learning comparison

Milestone 7 adds the `resnet18` backend as the V1 transfer-learning comparison. It initializes a
TorchVision `ResNet18_Weights.IMAGENET1K_V1` ImageNet encoder, replaces its first RGB convolution
with the arithmetic mean of its pretrained RGB kernels for grayscale input, and uses a small
U-Net-like bilinear decoder with the same cleaning and four-channel semantic heads. For small tile
batches, encoder BatchNorm running statistics are frozen while their affine parameters remain
trainable. Every transfer run records the exact weights enum and download URL, grayscale adaptation,
BatchNorm policy, architecture, and parameter count in `environment.json`. The weights download on
first use; ScoreRestore does not redistribute them.

Use the same dataset, task, split, crop, seed, and training settings as `default.yaml` when
comparing the custom U-Net and ResNet-18. The measured evaluation workflow below records the actual
comparison; pretrained natural-image features are not assumed to win.

## Tiled inference

Milestone 8 provides bounded-memory inference for PNG, JPEG, TIFF (including multipage TIFF), and
PDF inputs. `infer` keeps every page independent and in order; PDFs are rasterized with pypdfium2
at the configured DPI (300 by default). V1 writes a directory of per-page PNG outputs rather than
reconstructing a PDF.

```bash
uv run scorerestore infer input.pdf \
  -c configs/inference/default.yaml \
  -o runs/inference-example \
  --set checkpoint=runs/training-demo/checkpoints/best.pt
```

Each page directory contains a binary `cleaned.png`, cleaning probability, four independent semantic
probability maps, four thresholded masks, an optional inspection overlay, and `metadata.json`.
Outputs preserve the input raster dimensions exactly. The API is also available directly:

```python
from PIL import Image
from scorerestore import clean
from scorerestore.inference import load_checkpoint_model

model, _ = load_checkpoint_model("runs/training-demo/checkpoints/best.pt")
result = clean(Image.open("input.png"), model=model)
result.cleaned.save("cleaned.png")
```

The default uses 1024px tiles with 128px overlap. Tiles are reflection-padded at boundaries and
their logits are blended with a smooth raised-cosine window before thresholding, so neither cleaning
nor segmentation results have tile seams. Whole pages are never resized to fit memory.

Training writes dependency-free progress lines to the terminal (and Docker logs): start settings,
epoch/phase, batch count and percentage, running loss, elapsed time, ETA, and end-of-epoch losses.
Docker Compose mounts `./data` at both `/data` and the repository-relative `/app/data`, so the
default training configuration works unchanged in the container.

`assets/scores/real_world/` is the canonical local location for unannotated real-score examples.
They are useful for practical visual demonstrations, but are never V1 training, validation, or
quantitative evaluation data because they have no ground truth.

The Compose service reserves a 1 GiB RAM-backed `/dev/shm` area for PyTorch DataLoader workers.
This is distinct from free disk space: Docker otherwise defaults to just 64 MiB of shared memory,
which is insufficient for prefetched 1024px batches. On memory-constrained hosts, use synchronous
loading instead (slower, but it needs no worker shared-memory queue):

```bash
docker compose run --rm scorerestore scorerestore train \
  -c configs/training/default.yaml -o /runs/training-demo \
  --set training.num_workers=0
```

## Measured evaluation and benchmark reports

Milestone 9 evaluates named checkpoints using `configs/evaluation.yaml`. Its canonical three-model
template covers a cleaning-only custom U-Net, a multitask custom U-Net, and a pretrained ResNet-18
encoder model. For a controlled comparison, train the selected candidates with the same dataset
manifest, source-isolated split, budget, seed, and major hyperparameters. The generated summary
marks whether checkpoint provenance satisfies those comparability checks; it does not claim a
winner.

```bash
uv run scorerestore evaluate \
  -c configs/evaluation.yaml \
  -o runs/evaluation-demo
```

The command refuses to overwrite an existing output directory and writes `metrics.jsonl`,
`metrics.csv`, `summary.json`, and `report/summary.md`. It reports thresholded cleaning precision,
recall, F1/Dice, IoU, and continuous SSIM; per-class segmentation precision, recall, Dice, and IoU;
macro and foreground-only macro segmentation metrics; plus `all_false_rate` and
`background_overlap_rate`. It deliberately omits overall pixel accuracy. The selected OpenCV
variant is measured alongside neural cleaning output, but has no invented segmentation metrics.

Validation, test, and challenge are separate report sections; challenge is never folded into test.
The report also produces seeded, deterministic five-panel sheets:

```text
degraded | OpenCV baseline | DL cleaned | pristine target | segmentation overlay
```

Use `report.seed` and `report.samples` in the evaluation configuration to regenerate the same
visual selection. A report is a measurement of the listed checkpoints and dataset manifest, not a
general quality claim; repeated-seed comparisons require separate runs.

Measure actual tiled inference on a named checkpoint without estimating values:

```bash
uv run scorerestore benchmark input.pdf \
  -c configs/evaluation.yaml \
  -o runs/benchmark-demo.json \
  --model multitask_unet
```

The resulting JSON is labelled **MEASURED** and records hardware, tile size, overlap, precision
mode, PDF DPI, page dimensions, per-page and total latency, megapixels/second, and CUDA peak memory
when it is measurable. No benchmark table is published here because the project has not committed a
specific measured hardware run.

## Real-world visual comparison

Run the default four-panel comparison on every PDF in `assets/scores/real_world/`:

```bash
uv run scorerestore compare-real-world \
  -c configs/real_world/default.yaml \
  -o runs/real-world-comparison
```

The YAML configuration specifies the real-world input folder, tiled inference settings, classical
OpenCV variant, and the ordered neural `models` panels. The default discovers the eligible
ResNet-18 and custom U-Net checkpoints with the lowest saved training validation losses. This is a
deterministic local selection heuristic, not a cross-dataset quality claim; the chosen checkpoints
and losses are recorded in `metadata.json`.

Create another YAML file to compare multiple explicit training runs side by side:

```yaml
# configs/real_world/demo-comparison.yaml
input_root: assets/scores/real_world
runs_root: runs
inference: {device: auto, tile_size: 1024, overlap: 128, pdf_dpi: 300,
            cleaning_threshold: 0.5, segmentation_threshold: 0.5}
classical: {config: configs/baseline.yaml, variant: otsu}
models:
  - {id: resnet_cleaned, label: ResNet-18 cleaned, backend: resnet18, checkpoint: auto}
  - {id: demo_1, label: Training demo 1, backend: unet,
     checkpoint: runs/training-demo/checkpoints/best.pt}
  - {id: demo_2, label: Training demo 2, backend: unet,
     checkpoint: runs/training-demo2/checkpoints/best.pt}
```

The panels follow the `models` list exactly after Original and the classical result. Use a temporary
command-line checkpoint override without changing YAML when useful:

```bash
uv run scorerestore compare-real-world \
  -o runs/real-world-comparison \
  -c configs/real_world/demo-comparison.yaml \
  --checkpoint demo_2=runs/training-demo2/checkpoints/best.pt
```

It saves original, `classical_cleaned`, `resnet_cleaned`, and `model_cleaned` PNG pages and one
`comparison.pdf`; each configured model additionally has its own directory named after its `id`.
Every PDF page becomes one landscape page with exact native-resolution raster panels. No page is
downscaled for the comparison PDF. All reusable settings—including the classical variant, DPI, and
tile settings—belong in YAML.

## Native scaffold check

```bash
uv sync --frozen --group dev
uv run scorerestore --help
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Python 3.11 or 3.12 is supported. The lockfile selects Python 3.12 for reproducible development.

## Docker scaffold check

CPU:

```bash
docker compose build
docker compose run --rm scorerestore scorerestore --help
./scripts/docker/scorerestore --help
```

Optional NVIDIA GPU access:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm scorerestore scorerestore --help
./scripts/docker/scorerestore-gpu --help
```

The GPU override requires a compatible NVIDIA driver and NVIDIA Container Toolkit. It exposes the
same PyTorch training commands as the CPU image to CUDA.

## License

ScoreRestore source code is licensed under Apache-2.0. Third-party components and bundled assets are
listed in `THIRD_PARTY_NOTICES.md`. Each score source has separate composition and source-file rights
metadata in `assets/scores/manifest.yaml` and is integrity-checked before use.
