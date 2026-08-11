"""Deterministic, pixel-aligned synthetic degradation for ScoreRestore V1.

The public subsystem intentionally contains exactly five photometric families: Gaussian blur,
Gaussian image noise, uneven illumination, JPEG artifacts, and procedural stains/speckles. It
never changes image dimensions or applies geometry. Inputs are normalized to V1 grayscale
intensity (``0 = black``, ``255 = white``), and every selected operation receives an independent
recorded seed so its output can be reproduced from the returned machine-readable recipe.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import numpy as np
import PIL
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from scorerestore import __version__
from scorerestore.config import ConfigError, load_config

DegradationFamily: TypeAlias = Literal[
    "blur", "gaussian_noise", "uneven_illumination", "jpeg", "stains"
]
PresetName: TypeAlias = Literal["light", "medium", "heavy", "random"]

DEGRADATION_FAMILIES: tuple[DegradationFamily, ...] = (
    "blur",
    "gaussian_noise",
    "uneven_illumination",
    "jpeg",
    "stains",
)
PRESET_NAMES: tuple[PresetName, ...] = ("light", "medium", "heavy", "random")


class DegradationConfigError(ValueError):
    """Raised when a degradation configuration is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class FloatRange:
    """Inclusive floating-point sampling range."""

    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class IntRange:
    """Inclusive integer sampling range."""

    minimum: int
    maximum: int


@dataclass(frozen=True, slots=True)
class BlurSettings:
    enabled: bool
    radius: FloatRange


@dataclass(frozen=True, slots=True)
class GaussianNoiseSettings:
    enabled: bool
    sigma: FloatRange


@dataclass(frozen=True, slots=True)
class UnevenIlluminationSettings:
    enabled: bool
    strength: FloatRange
    grid_size: IntRange


@dataclass(frozen=True, slots=True)
class JpegSettings:
    enabled: bool
    quality: IntRange


@dataclass(frozen=True, slots=True)
class StainSettings:
    enabled: bool
    opacity: FloatRange
    blob_count: IntRange
    blob_radius_fraction: FloatRange
    speckle_density: FloatRange


@dataclass(frozen=True, slots=True)
class DegradationConfig:
    """Resolved V1 degradation configuration."""

    preset: str
    operation_count: IntRange
    blur: BlurSettings
    gaussian_noise: GaussianNoiseSettings
    uneven_illumination: UnevenIlluminationSettings
    jpeg: JpegSettings
    stains: StainSettings

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> DegradationConfig:
        """Validate and resolve a YAML-compatible configuration mapping."""

        allowed = {"preset", "operation_count", *DEGRADATION_FAMILIES}
        _reject_unknown_keys(raw, allowed, "degradation config")
        preset = raw.get("preset", "custom")
        if not isinstance(preset, str) or not preset:
            raise DegradationConfigError("preset must be a nonempty string")

        base: dict[str, Any]
        if preset in _PRESET_DATA:
            base = _copy_mapping(_PRESET_DATA[preset])
        elif preset == "custom":
            base = _copy_mapping(_CUSTOM_BASE)
        else:
            choices = ", ".join(PRESET_NAMES)
            raise DegradationConfigError(f"unknown preset {preset!r}; expected one of: {choices}")
        _merge_config(base, raw)

        config = cls(
            preset=preset,
            operation_count=_positive_int_range(base["operation_count"], "operation_count"),
            blur=_blur_settings(base["blur"]),
            gaussian_noise=_noise_settings(base["gaussian_noise"]),
            uneven_illumination=_illumination_settings(base["uneven_illumination"]),
            jpeg=_jpeg_settings(base["jpeg"]),
            stains=_stain_settings(base["stains"]),
        )
        if not config.enabled_families:
            raise DegradationConfigError("at least one degradation family must be enabled")
        return config

    @property
    def enabled_families(self) -> tuple[DegradationFamily, ...]:
        """Return enabled families in the canonical V1 registry order."""

        return tuple(family for family in DEGRADATION_FAMILIES if getattr(self, family).enabled)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable resolved configuration."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DegradationResult:
    """Degraded grayscale image plus exact recipe and runtime metadata."""

    image: Image.Image
    recipe: dict[str, object]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class DegradationPipeline:
    """Small reusable V1 pipeline interface independent of dataset and training code."""

    config: DegradationConfig

    def apply(self, image: Image.Image, *, seed: int = 0) -> DegradationResult:
        """Apply this resolved configuration with a deterministic master seed."""

        return _degrade_with_config(image, self.config, seed)


_PRESET_DATA = MappingProxyType(
    {
        "light": {
            "operation_count": [1, 2],
            "blur": {"enabled": True, "radius": [0.3, 0.8]},
            "gaussian_noise": {"enabled": True, "sigma": [1.5, 4.0]},
            "uneven_illumination": {
                "enabled": True,
                "strength": [0.03, 0.10],
                "grid_size": [3, 4],
            },
            "jpeg": {"enabled": True, "quality": [82, 95]},
            "stains": {
                "enabled": True,
                "opacity": [0.02, 0.08],
                "blob_count": [1, 3],
                "blob_radius_fraction": [0.006, 0.018],
                "speckle_density": [0.00002, 0.00008],
            },
        },
        "medium": {
            "operation_count": [2, 3],
            "blur": {"enabled": True, "radius": [0.6, 1.4]},
            "gaussian_noise": {"enabled": True, "sigma": [3.0, 8.0]},
            "uneven_illumination": {
                "enabled": True,
                "strength": [0.08, 0.20],
                "grid_size": [3, 5],
            },
            "jpeg": {"enabled": True, "quality": [60, 82]},
            "stains": {
                "enabled": True,
                "opacity": [0.06, 0.18],
                "blob_count": [2, 7],
                "blob_radius_fraction": [0.008, 0.030],
                "speckle_density": [0.00008, 0.00030],
            },
        },
        "heavy": {
            "operation_count": [3, 5],
            "blur": {"enabled": True, "radius": [1.0, 2.4]},
            "gaussian_noise": {"enabled": True, "sigma": [7.0, 14.0]},
            "uneven_illumination": {
                "enabled": True,
                "strength": [0.15, 0.32],
                "grid_size": [3, 6],
            },
            "jpeg": {"enabled": True, "quality": [38, 65]},
            "stains": {
                "enabled": True,
                "opacity": [0.12, 0.30],
                "blob_count": [5, 14],
                "blob_radius_fraction": [0.010, 0.045],
                "speckle_density": [0.00025, 0.00080],
            },
        },
        "random": {
            "operation_count": [1, 5],
            "blur": {"enabled": True, "radius": [0.3, 2.4]},
            "gaussian_noise": {"enabled": True, "sigma": [1.5, 14.0]},
            "uneven_illumination": {
                "enabled": True,
                "strength": [0.03, 0.32],
                "grid_size": [3, 6],
            },
            "jpeg": {"enabled": True, "quality": [38, 95]},
            "stains": {
                "enabled": True,
                "opacity": [0.02, 0.30],
                "blob_count": [1, 14],
                "blob_radius_fraction": [0.006, 0.045],
                "speckle_density": [0.00002, 0.00080],
            },
        },
    }
)

_CUSTOM_BASE: Mapping[str, Any] = {
    "operation_count": [1, 1],
    "blur": {"enabled": False, "radius": [0.3, 0.8]},
    "gaussian_noise": {"enabled": False, "sigma": [1.5, 4.0]},
    "uneven_illumination": {
        "enabled": False,
        "strength": [0.03, 0.10],
        "grid_size": [3, 4],
    },
    "jpeg": {"enabled": False, "quality": [82, 95]},
    "stains": {
        "enabled": False,
        "opacity": [0.02, 0.08],
        "blob_count": [1, 3],
        "blob_radius_fraction": [0.006, 0.018],
        "speckle_density": [0.00002, 0.00008],
    },
}


def preset_config(name: PresetName) -> DegradationConfig:
    """Return one of the four built-in V1 presets."""

    if name not in PRESET_NAMES:
        choices = ", ".join(PRESET_NAMES)
        raise DegradationConfigError(f"unknown preset {name!r}; expected one of: {choices}")
    return DegradationConfig.from_mapping({"preset": name})


def resolve_degradation_config(
    config: DegradationConfig | Mapping[str, Any] | str | Path | None,
) -> DegradationConfig:
    """Resolve a typed config, preset name, mapping, or YAML path."""

    if config is None:
        return preset_config("medium")
    if isinstance(config, DegradationConfig):
        return config
    if isinstance(config, Mapping):
        return DegradationConfig.from_mapping(config)
    if isinstance(config, Path):
        return _config_from_path(config)
    if isinstance(config, str):
        if config in PRESET_NAMES:
            return preset_config(config)  # type: ignore[arg-type]
        return _config_from_path(Path(config))
    raise DegradationConfigError(
        "config must be a DegradationConfig, mapping, preset name, YAML path, or None"
    )


def degrade(
    image: Image.Image,
    *,
    config: DegradationConfig | Mapping[str, Any] | str | Path | None = None,
    seed: int = 0,
) -> DegradationResult:
    """Apply a reproducible composition of V1 degradations to one Pillow image."""

    return DegradationPipeline(resolve_degradation_config(config)).apply(image, seed=seed)


def _degrade_with_config(
    image: Image.Image,
    resolved: DegradationConfig,
    seed: int,
) -> DegradationResult:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Pillow Image")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    source = image.convert("L").copy()
    source_hash = _image_sha256(source)
    width, height = source.size
    master_rng = random.Random(seed)
    operation_count = master_rng.randint(
        resolved.operation_count.minimum, resolved.operation_count.maximum
    )
    enabled_families = list(resolved.enabled_families)
    if operation_count <= len(enabled_families):
        selected = master_rng.sample(enabled_families, operation_count)
    else:
        selected = master_rng.sample(enabled_families, len(enabled_families))
        selected.extend(
            master_rng.choices(enabled_families, k=operation_count - len(enabled_families))
        )
        master_rng.shuffle(selected)

    degraded = source
    operation_recipe: list[dict[str, object]] = []
    for order, family in enumerate(selected, start=1):
        operation_seed = master_rng.getrandbits(63)
        degraded, parameters = _apply_operation(
            degraded,
            family,
            resolved,
            operation_seed,
        )
        if degraded.size != source.size:
            raise RuntimeError(f"{family} unexpectedly changed image dimensions")
        operation_recipe.append(
            {
                "order": order,
                "family": family,
                "seed": operation_seed,
                "parameters": parameters,
            }
        )

    output_hash = _image_sha256(degraded)
    software_versions = {
        "scorerestore": __version__,
        "python": platform.python_version(),
        "pillow": PIL.__version__,
        "numpy": np.__version__,
    }
    metadata: dict[str, object] = {
        "seed": seed,
        "preset": resolved.preset,
        "input_mode": image.mode,
        "working_mode": "L",
        "dimensions": {"width": width, "height": height},
        "geometry_changed": False,
        "input_sha256": source_hash,
        "output_sha256": output_hash,
        "software_versions": software_versions,
    }
    recipe: dict[str, object] = {
        "schema_version": 1,
        "seed": seed,
        "degradation_preset": resolved.preset,
        "source": {
            "sha256": source_hash,
            "hash_kind": "normalized_grayscale_pixels_with_dimensions",
            "mode": "L",
            "dimensions": {"width": width, "height": height},
        },
        "resolved_config": resolved.to_dict(),
        "operations": operation_recipe,
        "output": {
            "sha256": output_hash,
            "hash_kind": "normalized_grayscale_pixels_with_dimensions",
            "mode": degraded.mode,
            "dimensions": {"width": degraded.width, "height": degraded.height},
        },
        "software_versions": software_versions,
    }
    return DegradationResult(image=degraded, recipe=recipe, metadata=metadata)


def recipe_json(result: DegradationResult) -> str:
    """Serialize a degradation result's recipe in stable human-readable JSON."""

    return json.dumps(result.recipe, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _apply_operation(
    image: Image.Image,
    family: DegradationFamily,
    config: DegradationConfig,
    seed: int,
) -> tuple[Image.Image, dict[str, object]]:
    parameter_rng = random.Random(seed)
    if family == "blur":
        radius = _sample_float(config.blur.radius, parameter_rng)
        return image.filter(ImageFilter.GaussianBlur(radius=radius)), {
            "kind": "gaussian",
            "radius": radius,
        }
    if family == "gaussian_noise":
        sigma = _sample_float(config.gaussian_noise.sigma, parameter_rng)
        return _gaussian_noise(image, sigma, seed), {"sigma": sigma}
    if family == "uneven_illumination":
        strength = _sample_float(config.uneven_illumination.strength, parameter_rng)
        grid_size = _sample_int(config.uneven_illumination.grid_size, parameter_rng)
        return _uneven_illumination(image, strength, grid_size, seed), {
            "strength": strength,
            "grid_size": grid_size,
            "kind": "smooth_shadow_field",
        }
    if family == "jpeg":
        quality = _sample_int(config.jpeg.quality, parameter_rng)
        return _jpeg_artifacts(image, quality), {
            "quality": quality,
            "optimize": False,
            "progressive": False,
        }
    if family == "stains":
        opacity = _sample_float(config.stains.opacity, parameter_rng)
        blob_count = _sample_int(config.stains.blob_count, parameter_rng)
        radius_fraction = _sample_float(config.stains.blob_radius_fraction, parameter_rng)
        speckle_density = _sample_float(config.stains.speckle_density, parameter_rng)
        stained, speckle_count, blur_radius = _procedural_stains(
            image,
            opacity=opacity,
            blob_count=blob_count,
            radius_fraction=radius_fraction,
            speckle_density=speckle_density,
            seed=seed,
        )
        return stained, {
            "opacity": opacity,
            "blob_count": blob_count,
            "blob_radius_fraction": radius_fraction,
            "speckle_density": speckle_density,
            "speckle_count": speckle_count,
            "edge_blur_radius": blur_radius,
        }
    raise AssertionError(f"unhandled degradation family: {family}")


def _gaussian_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    values = np.asarray(image, dtype=np.float32)
    noise = np.random.default_rng(seed).normal(0.0, sigma, values.shape)
    degraded = np.clip(np.rint(values + noise), 0, 255).astype(np.uint8)
    return Image.fromarray(degraded, mode="L")


def _uneven_illumination(
    image: Image.Image,
    strength: float,
    grid_size: int,
    seed: int,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    low_resolution = rng.uniform(1.0 - strength, 1.0, (grid_size, grid_size)).astype(np.float32)
    field = Image.fromarray(low_resolution, mode="F").resize(image.size, Image.Resampling.BICUBIC)
    values = np.asarray(image, dtype=np.float32)
    illumination = np.asarray(field, dtype=np.float32)
    degraded = np.clip(np.rint(values * illumination), 0, 255).astype(np.uint8)
    return Image.fromarray(degraded, mode="L")


def _jpeg_artifacts(image: Image.Image, quality: int) -> Image.Image:
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=quality, optimize=False, progressive=False)
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        return decoded.convert("L").copy()


def _procedural_stains(
    image: Image.Image,
    *,
    opacity: float,
    blob_count: int,
    radius_fraction: float,
    speckle_density: float,
    seed: int,
) -> tuple[Image.Image, int, float]:
    rng = np.random.default_rng(seed)
    width, height = image.size
    shortest_side = min(width, height)
    base_radius = max(1.0, shortest_side * radius_fraction)
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    maximum_ink = max(1, round(255 * opacity))
    for _ in range(blob_count):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        radius_x = max(1, round(base_radius * float(rng.uniform(0.7, 2.2))))
        radius_y = max(1, round(base_radius * float(rng.uniform(0.5, 1.8))))
        ink = max(1, round(maximum_ink * float(rng.uniform(0.55, 1.0))))
        draw.ellipse((x - radius_x, y - radius_y, x + radius_x, y + radius_y), fill=ink)

    speckle_count = round(width * height * speckle_density)
    for _ in range(speckle_count):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        radius = int(rng.integers(0, 2))
        ink = max(1, round(maximum_ink * float(rng.uniform(0.4, 1.0))))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=ink)

    blur_radius = round(max(0.5, base_radius * 0.18), 6)
    softened = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return ImageChops.subtract(image, softened), speckle_count, blur_radius


def _config_from_path(path: Path) -> DegradationConfig:
    try:
        raw = load_config(path)
    except ConfigError as error:
        raise DegradationConfigError(str(error)) from error
    return DegradationConfig.from_mapping(raw)


def _merge_config(base: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    if "operation_count" in overrides:
        base["operation_count"] = overrides["operation_count"]
    for family in DEGRADATION_FAMILIES:
        if family not in overrides:
            continue
        value = overrides[family]
        if not isinstance(value, Mapping):
            raise DegradationConfigError(f"{family} must be a mapping")
        current = base[family]
        current.update(value)


def _copy_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in raw.items():
        copied[key] = dict(value) if isinstance(value, Mapping) else list(value)
    return copied


def _blur_settings(raw: Any) -> BlurSettings:
    values = _family_mapping(raw, "blur", {"enabled", "radius"})
    return BlurSettings(
        enabled=_enabled(values, "blur"),
        radius=_float_range(values["radius"], "blur.radius", minimum=0.0, maximum=5.0),
    )


def _noise_settings(raw: Any) -> GaussianNoiseSettings:
    values = _family_mapping(raw, "gaussian_noise", {"enabled", "sigma"})
    return GaussianNoiseSettings(
        enabled=_enabled(values, "gaussian_noise"),
        sigma=_float_range(values["sigma"], "gaussian_noise.sigma", minimum=0.0, maximum=50.0),
    )


def _illumination_settings(raw: Any) -> UnevenIlluminationSettings:
    values = _family_mapping(raw, "uneven_illumination", {"enabled", "strength", "grid_size"})
    return UnevenIlluminationSettings(
        enabled=_enabled(values, "uneven_illumination"),
        strength=_float_range(
            values["strength"],
            "uneven_illumination.strength",
            minimum=0.0,
            maximum=0.5,
        ),
        grid_size=_int_range(
            values["grid_size"], "uneven_illumination.grid_size", minimum=2, maximum=12
        ),
    )


def _jpeg_settings(raw: Any) -> JpegSettings:
    values = _family_mapping(raw, "jpeg", {"enabled", "quality"})
    return JpegSettings(
        enabled=_enabled(values, "jpeg"),
        quality=_int_range(values["quality"], "jpeg.quality", minimum=20, maximum=100),
    )


def _stain_settings(raw: Any) -> StainSettings:
    values = _family_mapping(
        raw,
        "stains",
        {"enabled", "opacity", "blob_count", "blob_radius_fraction", "speckle_density"},
    )
    return StainSettings(
        enabled=_enabled(values, "stains"),
        opacity=_float_range(values["opacity"], "stains.opacity", minimum=0.0, maximum=0.5),
        blob_count=_int_range(values["blob_count"], "stains.blob_count", minimum=0, maximum=100),
        blob_radius_fraction=_float_range(
            values["blob_radius_fraction"],
            "stains.blob_radius_fraction",
            minimum=0.001,
            maximum=0.1,
        ),
        speckle_density=_float_range(
            values["speckle_density"],
            "stains.speckle_density",
            minimum=0.0,
            maximum=0.01,
        ),
    )


def _family_mapping(raw: Any, name: str, allowed: set[str]) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise DegradationConfigError(f"{name} must be a mapping")
    _reject_unknown_keys(raw, allowed, name)
    missing = allowed - set(raw)
    if missing:
        raise DegradationConfigError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    return raw


def _enabled(raw: Mapping[str, Any], name: str) -> bool:
    enabled = raw["enabled"]
    if not isinstance(enabled, bool):
        raise DegradationConfigError(f"{name}.enabled must be a boolean")
    return enabled


def _float_range(
    raw: Any,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> FloatRange:
    values = _range_values(raw, name)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise DegradationConfigError(f"{name} must contain numbers")
    low, high = (float(value) for value in values)
    if low < minimum or high > maximum or low > high:
        raise DegradationConfigError(
            f"{name} must be ordered within [{minimum}, {maximum}], got [{low}, {high}]"
        )
    return FloatRange(low, high)


def _int_range(
    raw: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> IntRange:
    values = _range_values(raw, name)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise DegradationConfigError(f"{name} must contain integers")
    low, high = values
    if low < minimum or high > maximum or low > high:
        raise DegradationConfigError(
            f"{name} must be ordered within [{minimum}, {maximum}], got [{low}, {high}]"
        )
    return IntRange(low, high)


def _positive_int_range(raw: Any, name: str) -> IntRange:
    values = _range_values(raw, name)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise DegradationConfigError(f"{name} must contain integers")
    low, high = values
    if low < 1 or low > high:
        raise DegradationConfigError(
            f"{name} must be an ordered positive range, got [{low}, {high}]"
        )
    return IntRange(low, high)


def _range_values(raw: Any, name: str) -> Sequence[Any]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != 2:
        raise DegradationConfigError(f"{name} must be a two-item range")
    return raw


def _reject_unknown_keys(raw: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise DegradationConfigError(f"unknown {context} fields: {', '.join(sorted(unknown))}")


def _sample_float(value_range: FloatRange, rng: random.Random) -> float:
    return round(rng.uniform(value_range.minimum, value_range.maximum), 6)


def _sample_int(value_range: IntRange, rng: random.Random) -> int:
    return rng.randint(value_range.minimum, value_range.maximum)


def _image_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("ascii"))
    digest.update(b"\0")
    digest.update(f"{image.width}x{image.height}".encode("ascii"))
    digest.update(b"\0")
    digest.update(image.tobytes())
    return digest.hexdigest()
