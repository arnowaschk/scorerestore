#!/usr/bin/env bash
set -euo pipefail

# End-to-end CPU release check. Run from a fresh clone; it never deletes existing host data/runs.
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
tag="fresh-$(date +%Y%m%d%H%M%S)"
data_root="/data/${tag}"
runs_root="/runs/${tag}"
manifest="${data_root}/scorerestore-smoke-v1/manifests/samples.jsonl"

cd "$root"
uv sync --frozen --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
docker compose build
docker compose run --rm scorerestore scorerestore inspect provenance
docker compose run --rm scorerestore scorerestore generate \
  -c configs/dataset/smoke.yaml --output-root "$data_root"
docker compose run --rm scorerestore scorerestore baseline "$manifest" \
  -c configs/baseline.yaml -o "${runs_root}/baseline"
docker compose run --rm scorerestore scorerestore train \
  -c configs/training/smoke.yaml -o "${runs_root}/training" \
  --set dataset_manifest="$manifest"
docker compose run --rm scorerestore sh -ceu '
  input=$(find "'$data_root'/scorerestore-smoke-v1/inputs" -name "*.png" -print -quit)
  scorerestore infer "$input" -c configs/inference/default.yaml -o "'$runs_root'/inference" \
    --set checkpoint="'$runs_root'/training/checkpoints/best.pt"
'

echo "Fresh-clone check completed. Host outputs: data/${tag} and runs/${tag}"
