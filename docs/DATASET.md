# Data and provenance

V1 uses existing score sources engraved in LilyPond, never procedurally composed music. For every
materialized sample, the source-level split is fixed before rendering: all layouts and degradation
variants of that source remain in train, validation, test, or challenge together.

```text
curated .ly source → pinned LilyPond render → pristine + 4 independent masks
                    → seeded degradation → input/targets/recipe/manifest
```

The pristine grayscale render remains the continuous cleaning target. The five V1 degradation
families are blur, Gaussian image noise, uneven illumination, JPEG artifacts, and stains/speckles;
they never add geometric warping. Layout variation is deterministic and records staff size, paper,
orientation, margins, DPI, and seed.

`assets/scores/manifest.yaml` is the strict manifest for distributed source bytes: provenance for
both composition and the specific typesetting, a source URL, and SHA-256 are mandatory. CI validates
all three. `assets/scores/corpus-40.yaml` is the reviewed, pinned 40-file Mutopia expansion
catalogue. Materialize it into a new review directory with:

```bash
uv run python scripts/curate_mutopia_corpus.py -o /tmp/scorerestore-curated-40
uv run scorerestore inspect provenance --manifest /tmp/scorerestore-curated-40/manifest.yaml
```

The script refuses existing output, checks each upstream header for Mutopia and Public Domain
markers, records immutable raw-GitHub URLs and SHA-256s, and emits a manifest that undergoes the
same strict validation. Review rendered results and provenance before promoting the staged corpus
to a release branch. Unannotated files in `assets/scores/real_world/` are visual-demo inputs only;
they are never quantitative V1 data.

The container provides DejaVu Sans through Fontconfig because some unmodified curated sources
request it for Unicode copyright-markup glyphs. This preserves the source's requested typography
and avoids substituting a bundled LilyPond font with incomplete glyph coverage.
