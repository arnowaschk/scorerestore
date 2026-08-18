#!/usr/bin/env bash
set -euo pipefail

# Re-run only the canonical CUDA real-world comparison after a full GPU training run.  It leaves
# datasets and checkpoints untouched and writes a fresh comparison plus quality-proxy report.
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
runs_root="/runs/full-run-gpu"
output_name="real-world-quality"
force=false
compose=(docker compose -f compose.yaml -f compose.gpu.yaml)

case "${1:-}" in
  "") ;;
  --force)
    force=true
    ;;
  *)
    echo "Usage: $0 [--force]" >&2
    exit 2
    ;;
esac

host_output="$root/runs/full-run-gpu/$output_name"
if [[ -e "$host_output" ]]; then
  if [[ "$force" != true ]]; then
    echo "Refusing to overwrite $host_output. Pass --force to replace only this comparison output." >&2
    exit 1
  fi
  echo "--force: removing existing comparison output at $host_output"
  rm -rf -- "$host_output"
fi

cd "$root"
"${compose[@]}" build
"${compose[@]}" run --rm scorerestore scorerestore compare-real-world \
  -c configs/real_world/default.yaml -o "${runs_root}/${output_name}" \
  --set runs_root="$runs_root" \
  --set inference.device=cuda \
  --checkpoint resnet_cleaned="${runs_root}/training-resnet18/checkpoints/best.pt" \
  --checkpoint model_cleaned="${runs_root}/training-multitask/checkpoints/best.pt"

echo "CUDA real-world comparison completed. Host output: runs/full-run-gpu/$output_name"
