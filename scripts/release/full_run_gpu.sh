#!/usr/bin/env bash
set -euo pipefail

# Complete V1 CUDA showcase run. This creates the canonical 1,000-sample demo
# dataset, trains all three evaluated models for their full configured budgets,
# generates measured reports/benchmarks, and creates the real-world PDF comparison.
# It deliberately uses fixed output locations so their provenance is easy to find. A complete
# generated dataset is reused by default; pass --force to regenerate that dataset explicitly.
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="/data/full-run-gpu"
runs_root="/runs/full-run-gpu"
manifest="${data_root}/scorerestore-demo-v1/manifests/samples.jsonl"
source_manifest="${data_root}/curated-sources/manifest.yaml"
evaluation_config="configs/evaluation/full_run_gpu.yaml"
compose=(docker compose -f compose.yaml -f compose.gpu.yaml)
dataset_directory="$root/data/full-run-gpu/scorerestore-demo-v1"
reuse_dataset=false

case "${1:-}" in
  "") ;;
  --force) force=true ;;
  *)
    echo "Usage: $0 [--force]" >&2
    exit 2
    ;;
esac
force="${force:-false}"

if [[ -e "$root/runs/full-run-gpu" ]]; then
  echo "Refusing to overwrite runs/full-run-gpu." >&2
  echo "Review, archive, or deliberately remove the existing full-run output first." >&2
  exit 1
fi
if [[ -e "$dataset_directory" ]]; then
  if [[ ! -f "$dataset_directory/manifests/samples.jsonl" ]]; then
    echo "Refusing to use incomplete dataset directory: $dataset_directory" >&2
    echo "Review or deliberately remove it before restarting the full run." >&2
    exit 1
  fi
  if [[ "$force" == true ]]; then
    echo "--force: removing the existing generated dataset at $dataset_directory"
    rm -rf -- "$dataset_directory"
  else
    reuse_dataset=true
    echo "Reusing generated dataset at data/full-run-gpu/scorerestore-demo-v1"
  fi
fi

cd "$root"
if [[ "$reuse_dataset" == false ]]; then
  "${compose[@]}" build
  "${compose[@]}" run --rm scorerestore python -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the scorerestore container")
print(f"CUDA available: {torch.cuda.get_device_name(0)} ({torch.version.cuda})")
'
  "${compose[@]}" run --rm scorerestore scorerestore inspect provenance
  if [[ ! -f "$root/data/full-run-gpu/curated-sources/manifest.yaml" ]]; then
    "${compose[@]}" run --rm scorerestore python scripts/curate_mutopia_corpus.py \
      -o "${data_root}/curated-sources"
  else
    echo "Reusing existing curated source staging at data/full-run-gpu/curated-sources"
  fi
  "${compose[@]}" run --rm scorerestore scorerestore inspect provenance --manifest "$source_manifest"
  "${compose[@]}" run --rm scorerestore python scripts/preflight_lilypond_sources.py \
    --manifest "$source_manifest" --workers auto
  "${compose[@]}" run --rm scorerestore scorerestore generate \
    -c configs/dataset/demo.yaml --output-root "$data_root" \
    --set source_manifest="$source_manifest"
fi
"${compose[@]}" run --rm scorerestore scorerestore dataset validate "$manifest"

# All runs use the same source-isolated materialized dataset, full 50-epoch budget,
# seed, crop settings, and CUDA device. The task/backend are the only intended changes.
"${compose[@]}" run --rm scorerestore scorerestore train \
  -c configs/training/default.yaml -o "${runs_root}/training-clean" \
  --set dataset_manifest="$manifest" \
  --set task=clean \
  --set training.device=cuda
"${compose[@]}" run --rm scorerestore scorerestore train \
  -c configs/training/default.yaml -o "${runs_root}/training-multitask" \
  --set dataset_manifest="$manifest" \
  --set training.device=cuda
"${compose[@]}" run --rm scorerestore scorerestore train \
  -c configs/training/resnet18.yaml -o "${runs_root}/training-resnet18" \
  --set dataset_manifest="$manifest" \
  --set training.device=cuda

"${compose[@]}" run --rm scorerestore scorerestore evaluate \
  -c "$evaluation_config" -o "${runs_root}/evaluation"
"${compose[@]}" run --rm scorerestore sh -ceu '
  input=$(find "'$data_root'/scorerestore-demo-v1/inputs" -name "*.png" -print -quit)
  test -n "$input"
  scorerestore benchmark "$input" -c "'$evaluation_config'" \
    -o "'$runs_root'/benchmarks/clean_unet.json" --model clean_unet
  scorerestore benchmark "$input" -c "'$evaluation_config'" \
    -o "'$runs_root'/benchmarks/multitask_unet.json" --model multitask_unet
  scorerestore benchmark "$input" -c "'$evaluation_config'" \
    -o "'$runs_root'/benchmarks/pretrained_resnet18.json" --model pretrained_resnet18
'
"${compose[@]}" run --rm scorerestore scorerestore compare-real-world \
  -c configs/real_world/default.yaml -o "${runs_root}/real-world" \
  --set runs_root="$runs_root" \
  --set inference.device=cuda \
  --checkpoint resnet_cleaned="${runs_root}/training-resnet18/checkpoints/best.pt" \
  --checkpoint model_cleaned="${runs_root}/training-multitask/checkpoints/best.pt"

echo "Full CUDA run completed. Host outputs: data/full-run-gpu and runs/full-run-gpu"
