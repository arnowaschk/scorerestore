#!/usr/bin/env bash
set -euo pipefail

# End-to-end CUDA release check. Run from a fresh clone on a host with the NVIDIA
# Container Toolkit configured. It refuses a CPU fallback for neural work and never
# deletes existing host data/runs.
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
tag="fresh-gpu-$(date +%Y%m%d%H%M%S)"
data_root="/data/${tag}"
runs_root="/runs/${tag}"
manifest="${data_root}/scorerestore-smoke-v1/manifests/samples.jsonl"
compose=(docker compose -f compose.yaml -f compose.gpu.yaml)

cd "$root"
uv sync --frozen --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
"${compose[@]}" build
"${compose[@]}" run --rm scorerestore python -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the scorerestore container")
print(f"CUDA available: {torch.cuda.get_device_name(0)} ({torch.version.cuda})")
'
"${compose[@]}" run --rm scorerestore scorerestore inspect provenance
"${compose[@]}" run --rm scorerestore scorerestore generate \
  -c configs/dataset/smoke.yaml --output-root "$data_root"
"${compose[@]}" run --rm scorerestore scorerestore baseline "$manifest" \
  -c configs/baseline.yaml -o "${runs_root}/baseline"
"${compose[@]}" run --rm scorerestore scorerestore train \
  -c configs/training/smoke.yaml -o "${runs_root}/training" \
  --set dataset_manifest="$manifest" \
  --set training.device=cuda
"${compose[@]}" run --rm scorerestore sh -ceu '
  input=$(find "'$data_root'/scorerestore-smoke-v1/inputs" -name "*.png" -print -quit)
  scorerestore infer "$input" -c configs/inference/default.yaml -o "'$runs_root'/inference" \
    --set checkpoint="'$runs_root'/training/checkpoints/best.pt" \
    --set device=cuda
'

echo "CUDA fresh-clone check completed. Host outputs: data/${tag} and runs/${tag}"
