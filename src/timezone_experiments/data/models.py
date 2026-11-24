"""Core territory and sample data models."""

from __future__ import annotations

# Geographic coordinate bounds are the standard domain values.
import math
from array import array
from dataclasses import dataclass, replace
from itertools import batched
from typing import TYPE_CHECKING, Any

from timezone_experiments.constants import MAX_LATITUDE, MAX_LONGITUDE, MIN_LATITUDE, MIN_LONGITUDE

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

BASE_SAMPLE_COLUMNS = 4


@dataclass(frozen=True, slots=True)
class TerritorySample:
    """A location evaluated during a territory's solar analysis."""

    latitude: float
    longitude: float
    area_weight: float = 1.0
    population_weight: float | None = None
    longitude_weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate geographic coordinates and weights."""
        if not MIN_LATITUDE <= self.latitude <= MAX_LATITUDE:
            msg = f"Latitude must be between -90 and 90: {self.latitude}."
            raise ValueError(msg)
        if not MIN_LONGITUDE <= self.longitude <= MAX_LONGITUDE:
            msg = f"Longitude must be between -180 and 180: {self.longitude}."
            raise ValueError(msg)
        if not _is_non_negative(self.area_weight) or not _is_non_negative(self.longitude_weight):
            msg = f"Sample weights must be finite and non-negative: {self.area_weight}, {self.longitude_weight}."
            raise ValueError(msg)
        if self.population_weight is not None and not _is_non_negative(self.population_weight):
            msg = f"Population weight must be finite and non-negative: {self.population_weight}."
            raise ValueError(msg)

    def with_zero_population(self) -> TerritorySample:
        """Return this base sample with population weight 0; overlaid population lives in separate samples."""
        return replace(self, population_weight=0)


@dataclass(frozen=True, slots=True)
class Territory:
    """An administrative territory and the samples used to represent it."""

    territory_id: str
    country_code: str
    name: str
    admin_level: int
    samples: tuple[TerritorySample, ...]
    current_timezone: str | None = None
    geometry: dict[str, Any] | None = None

    @property
    def label(self) -> str:
        """Identifier and name, as shown in progress and error messages."""
        return f"{self.territory_id} ({self.name})"

    def __post_init__(self) -> None:
        """Ensure the territory has an identifier and evaluation samples."""
        if not self.territory_id:
            msg = "territory_id cannot be empty."
            raise ValueError(msg)
        if not self.country_code:
            msg = "country_code cannot be empty."
            raise ValueError(msg)
        if not self.samples:
            msg = f"Territory {self.territory_id!r} must contain at least one sample."
            raise ValueError(msg)


def _is_non_negative(value: float) -> bool:
    return math.isfinite(value) and value >= 0


def pack_base_samples(samples: Iterable[TerritorySample]) -> array[float]:
    """Flatten base samples to ``(latitude, longitude, area_weight, longitude_weight)`` doubles.

    The compact form is cheap to pickle between processes and to store in caches; population weights are
    not included (base samples have none).
    """
    return array(
        "d",
        (
            value
            for sample in samples
            for value in (sample.latitude, sample.longitude, sample.area_weight, sample.longitude_weight)
        ),
    )


def unpack_base_samples(values: Sequence[float]) -> tuple[TerritorySample, ...]:
    """Rebuild base samples from ``pack_base_samples`` output."""
    if len(values) % BASE_SAMPLE_COLUMNS:
        msg = "Packed base samples must contain whole (latitude, longitude, area, longitude weight) rows."
        raise ValueError(msg)
    return tuple(
        TerritorySample(latitude, longitude, area_weight=area, longitude_weight=longitude_weight)
        for latitude, longitude, area, longitude_weight in batched(values, BASE_SAMPLE_COLUMNS, strict=True)
    )
