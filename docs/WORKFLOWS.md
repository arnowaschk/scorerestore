# Workflows

This guide collects the operational commands that are intentionally kept out of the root README.
Run commands from the repository root. Configuration is YAML-first; see the
[configuration reference](../configs/README.md) for every shipped preset.

## Environment and provenance

Create the locked native development environment and run the quality gate:

```bash
uv sync --frozen --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Validate bundled sources and their rights declarations, then inspect the renderer and masks:

```bash
uv run scorerestore inspect provenance
uv run scorerestore inspect environment
uv run scorerestore inspect masks tests/fixtures/lilypond/overlap.ly -o /tmp/score-masks
```

Rendering requires LilyPond 2.26.0. The Docker image installs and verifies that exact version;
unsupported native versions are reported but refused for rendering.

## Generate data

Generate a reproducible smoke dataset and validate its manifest:

```bash
uv run scorerestore generate -c configs/dataset/smoke.yaml --output-root data/generated
uv run scorerestore dataset validate data/generated/scorerestore-smoke-v1/manifests/samples.jsonl
```

Reproduce one materialized sample to check its source, layout, masks, and degradation:

```bash
uv run scorerestore dataset reproduce SAMPLE_ID \
  --dataset-id scorerestore-smoke-v1 \
  --data-root data/generated \
  -o /tmp/reproduced.png
```

To apply only a V1 degradation preset to an image:

```bash
uv run scorerestore degrade input.png -o degraded.png \
  -c configs/degradation/medium.yaml --seed 1234
```

The recipe records source and output hashes, sampled operations, parameters, seeds, dimensions,
and relevant software versions. See [Dataset and provenance](DATASET.md) for data format, source
splits, and curation.

## Baseline, training, and inference

Run the fixed classical baseline over a generated dataset:

```bash
uv run scorerestore baseline \
  data/generated/scorerestore-smoke-v1/manifests/samples.jsonl \
  -c configs/baseline.yaml -o runs/baseline-smoke
```

Train the default custom U-Net or the transfer-learning comparison:

```bash
uv run scorerestore train -c configs/training/default.yaml -o runs/training-demo
uv run scorerestore train -c configs/training/resnet18.yaml -o runs/training-resnet18
```

Run bounded-memory tiled inference on a PDF or raster input:

```bash
uv run scorerestore infer input.pdf \
  -c configs/inference/default.yaml \
  -o runs/inference-example \
  --set checkpoint=runs/training-demo/checkpoints/best.pt
```

The default configuration uses 1024px tiles with 128px overlap and preserves each input page's
native raster dimensions. See [Architecture](ARCHITECTURE.md) for model and inference design.

## Evaluation and real-world comparisons

Evaluate named checkpoints with separate validation, test, and challenge reports:

```bash
uv run scorerestore evaluate -c configs/evaluation.yaml -o runs/evaluation-demo
```

`evaluate` refuses an existing output directory. The checked-in evaluation configuration points to
the retained full-run checkpoints and manifest; copy it and replace paths when evaluating another
experiment.

Measure tiled inference for one configured model:

```bash
uv run scorerestore benchmark input.pdf \
  -c configs/evaluation.yaml \
  -o runs/benchmark-demo.json \
  --model multitask_unet
```

Run a qualitative real-world comparison. These inputs have no ground truth, so the resulting
quality diagnostics are not quantitative restoration claims:

```bash
uv run scorerestore compare-real-world \
  -c configs/real_world/default.yaml \
  -o runs/real-world-comparison
```

Copy `configs/real_world/default.yaml` to compare explicit checkpoints or more model panels. Use
`bash scripts/release/compare_real_world_gpu.sh` after a completed full CUDA run to reproduce its
real-world comparison. See [Benchmarks](BENCHMARKS.md) for metric definitions and reporting rules.

## Docker and release gates

Build the CPU image and check the CLI:

```bash
docker compose build
docker compose run --rm scorerestore scorerestore --help
./scripts/docker/scorerestore --help
```

On a host with NVIDIA Container Toolkit, use the GPU override:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm scorerestore scorerestore --help
./scripts/docker/scorerestore-gpu --help
```

For reproducible end-to-end checks from a fresh clone:

```bash
bash scripts/release/fresh_clone_check.sh
bash scripts/release/fresh_clone_check_gpu.sh
```

The GPU gate requires CUDA in the container. Dataset rendering and the OpenCV baseline remain CPU
operations; CUDA is used for neural training and inference.
