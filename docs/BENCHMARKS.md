# Benchmarks and reporting

ScoreRestore publishes no invented performance values. Run `scorerestore evaluate` to create the
machine-readable CSV/JSONL metrics and deterministic visual sheets, then cite that artifact.

Cleaning reports foreground precision, recall, F1/Dice, IoU, and continuous SSIM; it intentionally
does not headline background-dominated pixel accuracy. Segmentation reports precision, recall,
Dice, and IoU per independent channel, macro mean, and foreground-only macro mean. Challenge is
always reported separately from test.

```bash
uv run scorerestore evaluate -c configs/evaluation.yaml -o runs/evaluation-demo
uv run scorerestore benchmark input.pdf -c configs/evaluation.yaml \
  -o runs/benchmark-demo.json --model multitask_unet
```

`benchmark` labels output **MEASURED** and records hardware, model/checkpoint provenance, tile size,
overlap, precision mode, image dimensions, latency, megapixels/second, and CUDA peak memory when
available. A public table must be transcribed from a retained measured JSON artifact, identify the
dataset manifest and split, and never merge challenge into test.

The evaluation configuration supports cleaning-only versus multitask U-Net and custom U-Net versus
ResNet-18 comparisons. A comparison is controlled only when the report’s recorded provenance shows
matching data, split, budget, and major training settings. One seed is descriptive, not a causal or
general performance claim.
