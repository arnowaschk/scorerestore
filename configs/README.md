# Configuration

ScoreRestore uses YAML-first configuration. The `degradation/` directory contains the four V1
photometric presets: `light`, `medium`, `heavy`, and seeded `random`. Each file defines an inclusive
operation-count range and severity bounds for exactly the five V1 degradation families. The actual
selection, order, sampled parameters, and per-operation seeds are recorded in every output recipe.

Dataset, training, and benchmark configurations will be added only with the milestones that
consume them.
