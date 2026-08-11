# Third-party notices

This file records direct software components and assets used to build, run, or test the current
public ScoreRestore V1 implementation. Transitive Python package versions are pinned in `uv.lock`.
ScoreRestore does not distribute a prebuilt container image at this milestone.

| Component | Version | License | Role | Source/project reference | Redistribution notes |
|---|---:|---|---|---|---|
| Python | 3.12.11 | Python-2.0 | Runtime and Docker base | https://www.python.org/ | Docker build uses the official `python:3.12.11-slim-bookworm` image. |
| Debian | bookworm-slim base | Multiple free-software licenses | Base operating-system packages | https://www.debian.org/ | Consult package-level copyright files if distributing a built image. |
| uv | 0.11.20 | Apache-2.0 OR MIT | Locked dependency installation | https://github.com/astral-sh/uv | Copied from the versioned official uv container image during build. |
| PyYAML | 6.0.3 | MIT | YAML configuration parsing | https://pyyaml.org/ | Runtime Python dependency; exact resolution is in `uv.lock`. |
| Hatchling | 1.27.0 | MIT | Python package build backend | https://github.com/pypa/hatch | Build-time Python dependency pinned in `pyproject.toml`. |
| pytest | 8.4.2 | MIT | Test runner | https://pytest.org/ | Development and CI dependency only; exact resolution is in `uv.lock`. |
| Ruff | 0.16.2 | MIT | Linting and formatting | https://github.com/astral-sh/ruff | Development and CI dependency only; exact resolution is in `uv.lock`. |

## Bundled score assets

The following unchanged LilyPond source files are distributed as the Milestone 1 starter corpus.
Their exact source URLs, SHA-256 hashes, composition rights, and source-file rights are recorded in
`assets/scores/manifest.yaml`.

| Component | Version/identifier | License | Role | Source/project reference | Redistribution notes |
|---|---|---|---|---|---|
| Bach, Invention No. 2 LilyPond source | BWV 773; Mutopia-2008/06/15-58 | Public Domain | Score-source starter asset | https://www.mutopiaproject.org/cgibin/piece-info.cgi?id=58 | Composition and contributor typesetting are identified as Public Domain. |
| Beethoven, Für Elise LilyPond source | WoO 59; Mutopia-2015/08/18-931 | Public Domain | Score-source starter asset | https://www.mutopiaproject.org/cgibin/piece-info.cgi?id=931 | Composition, 1888 source edition, and contributor typesetting are identified as Public Domain. |
| Foster, Hard Times Come Again No More LilyPond source | Mutopia-2014/03/24-371 | Public Domain | Voice/lyrics score-source starter asset | https://www.mutopiaproject.org/cgibin/piece-info.cgi?id=371 | Composition, source edition, and contributor typesetting are identified as Public Domain. |

No generated datasets or model weights are distributed in Milestone 1.
