# ScoreRestore

**ScoreRestore — Deep-learning restoration and semantic analysis of scanned sheet music**

ScoreRestore V1 is a proof-of-concept project for cleaning degraded sheet-music pages and
predicting independent background, staff, notation, and text masks. **ScoreRestore is not an OMR
system.**

The repository currently implements the project scaffold, a small rights-cleared score-source
corpus, exact semantic LilyPond mask rendering, and reusable pixel-aligned synthetic degradation.
Dataset generation, model training, evaluation, and inference remain explicit CLI placeholders
until their specified V1 milestones. No benchmark or quality claims are made at this stage.

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

The GPU override requires a compatible NVIDIA driver and NVIDIA Container Toolkit. The current
image contains no PyTorch or model code yet, so this command only validates the container interface.

## License

ScoreRestore source code is licensed under Apache-2.0. Third-party components and bundled assets are
listed in `THIRD_PARTY_NOTICES.md`. Each score source has separate composition and source-file rights
metadata in `assets/scores/manifest.yaml` and is integrity-checked before use.
