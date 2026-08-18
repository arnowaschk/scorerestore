# Demo checkpoint release workflow

Do not commit model binaries to Git. Publish a small V1 checkpoint as a GitHub Release asset together
with `demo-checkpoint.json` and its SHA-256 file. The metadata must satisfy the model card’s release
requirements and state the checkpoint license explicitly.

Before publishing:

1. Run `uv run scorerestore inspect provenance` and retain its successful output.
2. Train or select a named run; retain resolved config, environment, and evaluation output.
3. Run `sha256sum CHECKPOINT > CHECKPOINT.sha256`.
4. Create `demo-checkpoint.json` with the artifact URL, SHA-256, architecture, data manifest hash,
   source manifest hash, metrics artifact path, input convention, ScoreRestore version, and license.
5. In a clean checkout, download the asset visibly, verify `sha256sum --check`, then use it with
   `scorerestore infer -c configs/inference/default.yaml --set checkpoint=PATH`.

Until a release asset exists, a fresh clone has a fully reproducible alternative: generate the smoke
dataset, train `configs/training/smoke.yaml`, then run inference using its `checkpoints/best.pt`.
`scripts/release/fresh_clone_check.sh` performs that path with Docker Compose.
