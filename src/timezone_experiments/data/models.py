"""Core territory and sample data models."""

from __future__ import annotations

# Geographic coordinate bounds are the standard domain values.
# ruff: noqa: PLR2004
from dataclasses import dataclass
from typing import Any


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
        if not -90 <= self.latitude <= 90:
            msg = f"Latitude must be between -90 and 90: {self.latitude}."
            raise ValueError(msg)
        if not -180 <= self.longitude <= 180:
            msg = f"Longitude must be between -180 and 180: {self.longitude}."
            raise ValueError(msg)
        if self.area_weight < 0 or self.longitude_weight < 0:
            msg = "Sample weights must be non-negative."
            raise ValueError(msg)
        if self.population_weight is not None and self.population_weight < 0:
            msg = "Population weight must be non-negative."
            raise ValueError(msg)


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
