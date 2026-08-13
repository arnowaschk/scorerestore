# Changelog

All notable changes to ScoreRestore are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Rights-cleared public-domain LilyPond starter corpus with strict manifest, rights, and SHA-256
  validation through `scorerestore inspect provenance`.
- Pinned LilyPond rendering with independent staff, notation, and text masks; derived background;
  strict QA; environment inspection; unknown-grob diagnostics; and visual QA panels.
- Reusable deterministic degradation API and CLI with exactly five pixel-aligned V1 corruption
  families, four composable presets, and complete JSON recipes.
- Deterministic source-level dataset splits, seeded engraving/layout variation, materialized
  inputs and five targets, strict JSONL manifests, stable sample IDs, dataset validation/loading,
  and exact or best-effort sample reproduction.
- Smoke, approximately 1,000-sample demo, and held-out challenge dataset configurations.
- Readable OpenCV classical cleaning comparison with shared illumination normalization, four fixed
  Otsu/adaptive variants with and without light morphology, foreground cleaning metrics, SSIM, and
  variant-labelled materialized results.
- Tiled bounded-memory inference for raster, multipage TIFF, and PDF inputs; binary cleaning,
  semantic masks/probabilities, overlays, per-page metadata, and a stable `scorerestore.clean()` API.
- Measured checkpoint evaluation and benchmark commands with machine-readable CSV/JSON metrics,
  separate validation/test/challenge summaries, deterministic five-panel visual comparison sheets,
  controlled U-Net/multitask and U-Net/ResNet-18 comparison provenance, and actual-only runtime
  records.
- Real-world PDF comparison command with automatic local custom-U-Net/ResNet-18 checkpoint
  selection, checkpoint-selection provenance, native-resolution cleaned PNG outputs, and YAML
  configurable, ordered original/classical/neural landscape panels per input page.

## [0.1.0] - 2026-08-11

### Added

- Initial Milestone 0 package, CLI, configuration, Docker Compose, test, and CI scaffold.
