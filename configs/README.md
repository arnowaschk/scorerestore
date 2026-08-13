# Configuration

ScoreRestore uses YAML-first configuration. The `degradation/` directory contains the four V1
photometric presets: `light`, `medium`, `heavy`, and seeded `random`. Each file defines an inclusive
operation-count range and severity bounds for exactly the five V1 degradation families. The actual
selection, order, sampled parameters, and per-operation seeds are recorded in every output recipe.

The `dataset/` directory contains materialized generation presets:

- `smoke.yaml`: two low-resolution samples for quick native and container checks;
- `demo.yaml`: the canonical approximately 1,000-sample demonstration dataset;
- `challenge.yaml`: a separate recoverable challenge dataset using held-out degradation patterns.

Dataset configuration controls the deterministic source split, layout grid, margin range, target
sample count, mask rendering, and degradation preset selection.

The `training/` directory contains custom U-Net and transfer-learning configurations:

- `default.yaml`: the standard 50-epoch V1 demo training run;
- `short.yaml`: the canonical five-epoch practical test run using the same defaults;
- `smoke.yaml`: a tiny CPU forward/backward/checkpoint verification run.
- `resnet18.yaml`: the comparable ImageNet-pretrained ResNet-18 transfer-learning run.

`short.yaml` is useful for validating an end-to-end training setup, but five epochs are not a
quality benchmark. Benchmark configurations will be added only with the milestones that consume
them.

`baseline.yaml` configures the fixed four-variant classical comparison: shared smooth
illumination-field estimation, Otsu and adaptive threshold parameters, one bounded light
morphology, and the pristine-target ink threshold. Every run covers thresholding with and without
that morphology on identical samples. CLI overrides use repeatable dotted assignments such as
`--set morphology.operation=open`.
