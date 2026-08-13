# Generated data

`scorerestore generate` writes materialized datasets below `data/generated/<dataset-id>/` by
default. Each dataset includes degraded inputs, pristine targets, background/staff/notation/text
masks, JSONL manifests, recipes, and QA/render reports. Generated contents are intentionally
ignored by Git; this documentation file is the only tracked exception.

## Real-world examples

`data/real_world/` is the canonical local location for unannotated example score pages. These
files are practical demonstration inputs only: they MUST NOT be used as V1 training targets,
validation data, or quantitative evaluation data because no clean or semantic annotations exist.
They are intentionally ignored by Git along with other local data.
