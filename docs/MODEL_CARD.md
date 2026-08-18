# Model card — ScoreRestore V1 demonstration checkpoint

## Intended use

The demonstration model cleans synthetic degradations of engraved sheet music and predicts four
inspection masks. It is suitable for exercising the CLI, evaluating the reproducible synthetic
pipeline, and creating qualitative demonstrations. It is not OMR and must not be used to make
editorial, archival, or accessibility claims about historical scans without human review.

## Inputs and outputs

Input is grayscale (`0 = black`, `1 = white` internally). The cleaning head predicts ink coverage;
final cleaned output is binary black ink on white. Semantic outputs are four independent sigmoid
channels: background, staff, notation, text. Tiles preserve original raster size.

## Data, evaluation, and limitations

Training data is synthetic degradation of provenance-checked LilyPond renders. It does not establish
performance on photographs, skewed pages, handwriting, unusual historical notation, or real scans.
Real-world PDFs have no target annotations and are qualitative only. Reported metrics apply only to
the named generated manifest/checkpoint and its separate splits.

## Release requirements

Every released checkpoint must have a companion `demo-checkpoint.json` containing ScoreRestore
version, SHA-256, architecture/backend, resolved training config, dataset ID and manifest hash,
source-provenance manifest hash, measured metrics artifact, input convention, and checkpoint license.
The release workflow is in `docs/DEMO_CHECKPOINT.md`.
