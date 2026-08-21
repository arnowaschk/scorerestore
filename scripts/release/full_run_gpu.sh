#!/usr/bin/env bash
set -euo pipefail

# Complete resumable CUDA showcase run. A profile isolates data and reports so multiple
# experiments can coexist. Use --update from the first invocation to make every materializing
# command resume-safe after an interruption.
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
profile=""
update=false

usage() {
  echo "Usage: $0 [--profile NAME] [--update]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      profile="$2"
      shift 2
      ;;
    --update)
      update=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

[[ -z "$profile" || "$profile" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || {
  echo "Profile must use lowercase letters, digits, dots, underscores, or hyphens." >&2
  exit 2
}

run_name="full-run-gpu"
if [[ -n "$profile" ]]; then
  run_name+="-${profile}"
fi
data_root="/data/${run_name}"
runs_root="/runs/${run_name}"
manifest="${data_root}/scorerestore-demo-v1/manifests/samples.jsonl"
source_manifest="${data_root}/curated-sources/manifest.yaml"
compose=(docker compose -f compose.yaml -f compose.gpu.yaml)
update_args=()
if [[ "$update" == true ]]; then
  update_args=(--update)
fi

cd "$root"
# Configuration is copied into the image, so always rebuild before the run.
"${compose[@]}" build
"${compose[@]}" run --rm scorerestore python -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the scorerestore container")
print(f"CUDA available: {torch.cuda.get_device_name(0)} ({torch.version.cuda})")
'
"${compose[@]}" run --rm scorerestore scorerestore inspect provenance

if [[ ! -f "$root/data/${run_name}/curated-sources/manifest.yaml" ]]; then
  "${compose[@]}" run --rm scorerestore python scripts/curate_mutopia_corpus.py \
    -o "${data_root}/curated-sources" "${update_args[@]}"
fi
"${compose[@]}" run --rm scorerestore scorerestore inspect provenance --manifest "$source_manifest"
"${compose[@]}" run --rm scorerestore python scripts/preflight_lilypond_sources.py \
  --manifest "$source_manifest" --workers auto
"${compose[@]}" run --rm scorerestore scorerestore generate \
  -c configs/dataset/demo.yaml --output-root "$data_root" \
  --set source_manifest="$source_manifest" "${update_args[@]}"
"${compose[@]}" run --rm scorerestore scorerestore dataset validate "$manifest"

# The same data, crop, seed, and budget are used for all compared models. Each train command
# persists last.pt after every epoch, so --update resumes the exact optimizer and AMP state.
"${compose[@]}" run --rm scorerestore scorerestore train \
  -c configs/training/default.yaml -o "${runs_root}/training-clean" \
  --set dataset_manifest="$manifest" --set task=clean --set training.device=cuda \
  "${update_args[@]}"
"${compose[@]}" run --rm scorerestore scorerestore train \
  -c configs/training/default.yaml -o "${runs_root}/training-multitask" \
  --set dataset_manifest="$manifest" --set training.device=cuda "${update_args[@]}"
"${compose[@]}" run --rm scorerestore scorerestore train \
  -c configs/training/resnet18.yaml -o "${runs_root}/training-resnet18" \
  --set dataset_manifest="$manifest" --set training.device=cuda "${update_args[@]}"

"${compose[@]}" run --rm scorerestore sh -ceu '
  config=/tmp/scorerestore-evaluation.yaml
  sed -e "s|/data/full-run-gpu|"'"$data_root"'|g" \
      -e "s|/runs/full-run-gpu|"'"$runs_root"'|g" \
      configs/evaluation/full_run_gpu.yaml > "$config"
  scorerestore evaluate -c "$config" -o "'"$runs_root"'/evaluation" '"${update_args[*]}"'
'
"${compose[@]}" run --rm scorerestore sh -ceu '
  config=/tmp/scorerestore-evaluation.yaml
  sed -e "s|/data/full-run-gpu|"'"$data_root"'|g" \
      -e "s|/runs/full-run-gpu|"'"$runs_root"'|g" \
      configs/evaluation/full_run_gpu.yaml > "$config"
  input=$(find "'"$data_root"'/scorerestore-demo-v1/inputs" -name "*.png" -print -quit)
  test -n "$input"
  scorerestore benchmark "$input" -c "$config" \
    -o "'"$runs_root"'/benchmarks/clean_unet.json" --model clean_unet '"${update_args[*]}"'
  scorerestore benchmark "$input" -c "$config" \
    -o "'"$runs_root"'/benchmarks/multitask_unet.json" --model multitask_unet '"${update_args[*]}"'
  scorerestore benchmark "$input" -c "$config" \
    -o "'"$runs_root"'/benchmarks/pretrained_resnet18.json" --model pretrained_resnet18 '"${update_args[*]}"'
'
"${compose[@]}" run --rm scorerestore scorerestore compare-real-world \
  -c configs/real_world/default.yaml -o "${runs_root}/real-world" \
  --set runs_root="$runs_root" --set inference.device=cuda \
  --checkpoint resnet_cleaned="${runs_root}/training-resnet18/checkpoints/best.pt" \
  --checkpoint model_cleaned="${runs_root}/training-multitask/checkpoints/best.pt" \
  "${update_args[@]}"

echo "Full CUDA run completed. Host outputs: data/${run_name} and runs/${run_name}"
