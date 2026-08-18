#!/usr/bin/env bash
set -euo pipefail

# Native equivalent of full_run_gpu.sh for CUDA hosts that cannot launch Docker containers.
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
profile="default"
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

[[ "$profile" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || {
  echo "Profile must use lowercase letters, digits, dots, underscores, or hyphens." >&2
  exit 2
}

data_root="data/full-run-native-${profile}"
runs_root="runs/full-run-native-${profile}"
manifest="${data_root}/scorerestore-demo-v1/manifests/samples.jsonl"
source_manifest="${data_root}/curated-sources/manifest.yaml"
update_args=()
if [[ "$update" == true ]]; then
  update_args=(--update)
fi

cd "$root"
uv sync --frozen --group dev
uv run python -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable for the native full run")
print(f"CUDA available: {torch.cuda.get_device_name(0)} ({torch.version.cuda})")
'
uv run scorerestore inspect provenance

if [[ ! -f "$source_manifest" ]]; then
  uv run python scripts/curate_mutopia_corpus.py -o "${data_root}/curated-sources" \
    "${update_args[@]}"
fi
uv run scorerestore inspect provenance --manifest "$source_manifest"
uv run python scripts/preflight_lilypond_sources.py --manifest "$source_manifest" --workers auto
uv run scorerestore generate -c configs/dataset/demo.yaml --output-root "$data_root" \
  --set source_manifest="$source_manifest" "${update_args[@]}"
uv run scorerestore dataset validate "$manifest"

uv run scorerestore train -c configs/training/default.yaml -o "${runs_root}/training-clean" \
  --set dataset_manifest="$manifest" --set task=clean --set training.device=cuda \
  "${update_args[@]}"
uv run scorerestore train -c configs/training/default.yaml -o "${runs_root}/training-multitask" \
  --set dataset_manifest="$manifest" --set training.device=cuda "${update_args[@]}"
uv run scorerestore train -c configs/training/resnet18.yaml -o "${runs_root}/training-resnet18" \
  --set dataset_manifest="$manifest" --set training.device=cuda "${update_args[@]}"

evaluation_config="$(mktemp)"
trap 'rm -f "$evaluation_config"' EXIT
sed -e "s|/data/full-run-gpu|${data_root}|g" -e "s|/runs/full-run-gpu|${runs_root}|g" \
  configs/evaluation/full_run_gpu.yaml > "$evaluation_config"
uv run scorerestore evaluate -c "$evaluation_config" -o "${runs_root}/evaluation" "${update_args[@]}"

input="$(find "${data_root}/scorerestore-demo-v1/inputs" -name '*.png' -print -quit)"
test -n "$input"
uv run scorerestore benchmark "$input" -c "$evaluation_config" \
  -o "${runs_root}/benchmarks/clean_unet.json" --model clean_unet "${update_args[@]}"
uv run scorerestore benchmark "$input" -c "$evaluation_config" \
  -o "${runs_root}/benchmarks/multitask_unet.json" --model multitask_unet "${update_args[@]}"
uv run scorerestore benchmark "$input" -c "$evaluation_config" \
  -o "${runs_root}/benchmarks/pretrained_resnet18.json" --model pretrained_resnet18 \
  "${update_args[@]}"
uv run scorerestore compare-real-world -c configs/real_world/default.yaml \
  -o "${runs_root}/real-world" --set runs_root="$runs_root" --set inference.device=cuda \
  --checkpoint resnet_cleaned="${runs_root}/training-resnet18/checkpoints/best.pt" \
  --checkpoint model_cleaned="${runs_root}/training-multitask/checkpoints/best.pt" \
  "${update_args[@]}"

echo "Native full CUDA run completed. Outputs: ${data_root} and ${runs_root}"
