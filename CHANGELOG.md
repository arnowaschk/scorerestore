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

## [0.1.0] - 2026-08-11

### Added

- Initial Milestone 0 package, CLI, configuration, Docker Compose, test, and CI scaffold.
