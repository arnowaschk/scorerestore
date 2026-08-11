# ScoreRestore

**ScoreRestore — Deep-learning restoration and semantic analysis of scanned sheet music**

ScoreRestore V1 is a proof-of-concept project for cleaning degraded sheet-music pages and
predicting independent background, staff, notation, and text masks. **ScoreRestore is not an OMR
system.**

This repository currently contains the Milestone 0 project scaffold. Dataset generation, model
training, evaluation, and inference commands are visible in the CLI but intentionally report that
they are unavailable until their specified V1 milestones are implemented. No benchmark or quality
claims are made at this stage.

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

The GPU override requires a compatible NVIDIA driver and NVIDIA Container Toolkit. The Milestone 0
image contains no PyTorch or model code yet, so this command only validates the container interface.

## License

ScoreRestore source code is licensed under Apache-2.0. Third-party components are listed in
`THIRD_PARTY_NOTICES.md`. Training assets and their provenance will be introduced and validated in
Milestone 1; none are bundled in this scaffold.

