# Curated score sources

This directory contains the rights-cleared starter corpus used to establish ScoreRestore's V1
provenance architecture. These are existing musical works and existing LilyPond typesettings; none
was procedurally composed by ScoreRestore.

Every bundled source must have a corresponding `manifest.yaml` entry that separately establishes:

- the rights status of the underlying composition or edition;
- the rights status of the specific LilyPond typesetting;
- the exact SHA-256 of the distributed source file.

The starter files were downloaded unchanged from the Mutopia Project. Their upstream catalog pages
and embedded headers identify both the score and typesetting as Public Domain. Mutopia describes its
Public Domain designation at <https://www.mutopiaproject.org/legal.html>.

Validate all bundled sources from the repository root:

```bash
uv run scorerestore inspect provenance
```

## Real-world demonstration PDFs

`real_world/` contains practical, unannotated score examples used only for visual inference
demonstrations. They are outside the generated training corpus and have no clean or semantic target
annotations, so they MUST NOT be used for V1 training, validation, test, challenge, or quantitative
evaluation. Use `scorerestore compare-real-world` to create a side-by-side visual comparison.

Do not edit or replace a source without reviewing its rights again and updating its hash, source
URL, verification date, and notes. Milestone 2 renders sources through a converted temporary copy;
the rights-cleared source bytes in this directory remain unchanged.
