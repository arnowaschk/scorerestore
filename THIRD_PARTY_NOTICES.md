# Third-party notices

This file records direct software components used to build, run, or test the public ScoreRestore V1
Milestone 0 scaffold. Transitive Python package versions are pinned in `uv.lock`. ScoreRestore does
not distribute a prebuilt container image at this milestone.

| Component | Version | License | Role | Source/project reference | Redistribution notes |
|---|---:|---|---|---|---|
| Python | 3.12.11 | Python-2.0 | Runtime and Docker base | https://www.python.org/ | Docker build uses the official `python:3.12.11-slim-bookworm` image. |
| Debian | bookworm-slim base | Multiple free-software licenses | Base operating-system packages | https://www.debian.org/ | Consult package-level copyright files if distributing a built image. |
| uv | 0.11.20 | Apache-2.0 OR MIT | Locked dependency installation | https://github.com/astral-sh/uv | Copied from the versioned official uv container image during build. |
| PyYAML | 6.0.3 | MIT | YAML configuration parsing | https://pyyaml.org/ | Runtime Python dependency; exact resolution is in `uv.lock`. |
| Hatchling | 1.27.0 | MIT | Python package build backend | https://github.com/pypa/hatch | Build-time Python dependency pinned in `pyproject.toml`. |
| pytest | 8.4.2 | MIT | Test runner | https://pytest.org/ | Development and CI dependency only; exact resolution is in `uv.lock`. |
| Ruff | 0.16.2 | MIT | Linting and formatting | https://github.com/astral-sh/ruff | Development and CI dependency only; exact resolution is in `uv.lock`. |

No LilyPond files, datasets, model weights, or score assets are distributed in Milestone 0.
