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

Do not edit or replace a source without reviewing its rights again and updating its hash, source
URL, verification date, and notes. Rendering compatibility is intentionally addressed in Milestone
2; Milestone 1 validates provenance and source integrity only.

