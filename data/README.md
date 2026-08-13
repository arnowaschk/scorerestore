# Generated data

`scorerestore generate` writes materialized datasets below `data/generated/<dataset-id>/` by
default. Each dataset includes degraded inputs, pristine targets, background/staff/notation/text
masks, JSONL manifests, recipes, and QA/render reports. Generated contents are intentionally
ignored by Git; this documentation file is the only tracked exception.

Real-world demonstration PDFs are kept separately in `assets/scores/real_world/`. They have no
clean or semantic annotations and therefore MUST NOT be used as V1 training targets, validation
data, or quantitative evaluation data.
