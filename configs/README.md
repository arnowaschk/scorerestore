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
sample count, mask rendering, and degradation preset selection. Training and benchmark
configurations will be added only with the milestones that consume them.

`baseline.yaml` configures the fixed four-variant classical comparison: shared smooth
illumination-field estimation, Otsu and adaptive threshold parameters, one bounded light
morphology, and the pristine-target ink threshold. Every run covers thresholding with and without
that morphology on identical samples. CLI overrides use repeatable dotted assignments such as
`--set morphology.operation=open`.
