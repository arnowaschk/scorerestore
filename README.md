# ScoreRestore

**Reproducible deep-learning restoration and semantic analysis for scanned sheet music.**

ScoreRestore V1 turns rights-cleared LilyPond score sources into pixel-aligned training data, trains
page-cleaning and semantic-segmentation models, evaluates them against an OpenCV baseline, and
creates full-resolution visual comparisons. It predicts independent background, staff, notation,
and text masks alongside page cleaning. **It is not an OMR system and does not emit MusicXML or
MIDI.**

## Why ScoreRestore

- **Score-aware supervision:** LilyPond supplies exact masks; every generated page retains its
  source, layout, recipe, and hashes.
- **Reproducible experiments:** source-isolated splits, YAML configuration, deterministic data
  generation, and recorded environments make results inspectable.
- **Complete restoration workflow:** data generation, PyTorch training, tiled full-page inference,
  classical comparison, evaluation, and real-world visual reports are included.
- **Evidence before claims:** test and challenge remain separate; unannotated scans are never given
  invented quantitative metrics.

## Measured V1 result snapshot

The canonical CUDA run (`bash scripts/release/full_run_gpu.sh`) trains on the 40-source curated
corpus and evaluates source-isolated synthetic pages. Its held-out **test** report (46 pages) is:

| Method | Cleaning Dice | Cleaning SSIM | Foreground segmentation macro Dice |
| --- | ---: | ---: | ---: |
| OpenCV Otsu baseline | 0.9400 | 0.9887 | — |
| Custom clean-only U-Net | **0.9946** | 0.9982 | — |
| Custom multitask U-Net | 0.9943 | 0.9981 | 0.9688 |
| ImageNet-pretrained ResNet-18 | 0.9926 | **0.9982** | **0.9767** |

Bold marks the column best using unrounded report values. The clean-only U-Net's SSIM rounds to
`0.9982` but is marginally lower than the ResNet-18's; a dash means the method produces no semantic
masks. Full methodology and artifacts are described in the [benchmark guide](docs/BENCHMARKS.md).

### Validation model-selection snapshot

Use validation—not the held-out test table—to select future multitask models. `T` is the best
validation score among the compared models; `r` is the current multitask score divided by `T`.

| Validation metric | Model setting `T` | Best score `T` | Multitask score | Relative score `r` |
| --- | --- | ---: | ---: | ---: |
| Cleaning Dice | Clean-only U-Net | 0.9959 | 0.9953 | 99.93% |
| Cleaning SSIM | Clean-only U-Net | 0.9982 | 0.9980 | 99.98% |
| Foreground segmentation macro Dice | ImageNet-pretrained ResNet-18 | 0.9765 | 0.9657 | 98.89% |

Segmentation is the current multitask model's limiting metric. These are selection diagnostics,
not held-out test results.

## Quick start

Install the locked development environment and run the native checks:

```bash
uv sync --frozen --group dev
uv run ruff check .
uv run pytest
```

Generate and validate a tiny reproducible dataset:

```bash
uv run scorerestore generate -c configs/dataset/smoke.yaml --output-root data/generated
uv run scorerestore dataset validate data/generated/scorerestore-smoke-v1/manifests/samples.jsonl
```

For the complete CUDA experiment and retained reports, use a Docker host with NVIDIA Container
Toolkit configured:

```bash
bash scripts/release/full_run_gpu.sh
```

It writes data below `data/full-run-gpu/` and reports below `runs/full-run-gpu/`. See the
[workflow guide](docs/WORKFLOWS.md) for training, inference, evaluation, Docker, and release
commands.

## Workflow at a glance

```text
rights-cleared .ly sources → pristine render + semantic masks → seeded degradation
                         → source-isolated dataset → train → tiled inference → evaluate
```

| Need | Start here |
| --- | --- |
| Source rights, deterministic splits, data layout, and degradation | [Dataset and provenance](docs/DATASET.md) |
| Models, semantic channels, and tile blending | [Architecture](docs/ARCHITECTURE.md) |
| Training, inference, baselines, real-world comparisons, and Docker | [Workflow guide](docs/WORKFLOWS.md) |
| Metrics, controlled comparisons, and benchmark provenance | [Benchmarks](docs/BENCHMARKS.md) |
| YAML presets and their purpose | [Configuration reference](configs/README.md) |

## Scope and evidence

ScoreRestore is a synthetic-data V1 proof of concept. It has not established quantitative quality
on annotated real scans, handwriting, skew or perspective distortion, historical material, or
arbitrary source editions. PDFs in `assets/scores/real_world/` are qualitative inspection inputs
only. Trust a reported score or performance figure only when it comes from a retained artifact
labelled **MEASURED**.

The real-world comparison intentionally retains diagnostic failures—for example, tiled-inference
seams—rather than presenting them as hidden exceptions. For model limitations and intended use,
read the [model card](docs/MODEL_CARD.md); for planned work, see the [roadmap](docs/ROADMAP.md).

## Release and license

Run `bash scripts/release/fresh_clone_check.sh` for the end-to-end CPU release gate, or
`bash scripts/release/fresh_clone_check_gpu.sh` on a CUDA-capable Docker host. The
[demo-checkpoint guide](docs/DEMO_CHECKPOINT.md) describes publishing a verified release asset.

ScoreRestore source code is licensed under Apache-2.0. Third-party components and bundled assets
are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Score-specific composition and
source-file rights metadata live in `assets/scores/manifest.yaml` and are integrity-checked before
use.
