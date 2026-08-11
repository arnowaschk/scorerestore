"""Public package metadata for ScoreRestore.

The image convention used by later V1 milestones is grayscale intensity with
``0.0 = black`` and ``1.0 = white``. Restoration targets invert that convention into ink
coverage/probability with ``1.0 = desired ink`` and ``0.0 = desired background``. Keeping these
meanings explicit prevents silent target and metric inversion.
"""

from typing import Final

__version__: Final = "0.1.0"

from scorerestore.degradation import (
    DegradationConfig,
    DegradationPipeline,
    DegradationResult,
    degrade,
)

__all__ = [
    "DegradationConfig",
    "DegradationPipeline",
    "DegradationResult",
    "__version__",
    "degrade",
]
