"""Core territory and sample data models."""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, replace
from itertools import batched
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from timezone_experiments.core.constants import (
    MAX_LATITUDE,
    MAX_LONGITUDE,
    MIN_LATITUDE,
    MIN_LONGITUDE,
    MINUTES_PER_DAY,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray

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


@dataclass(frozen=True, slots=True)
class NoonConstraint:
    """Inclusive minimum and maximum local solar-noon bounds."""

    minimum_minutes: int | None
    maximum_minutes: int | None

    @property
    def window_start(self) -> float:
        """Start of the 24-hour clock window that noon times are compared in.

        Clock times repeat every day, so 00:14 could mean 14 minutes past midnight or 11h46 before noon
        tomorrow's frame. Centering the window on the active bound (half a day either side) makes "earlier"
        and "later" unambiguous: a noon just after midnight counts as very early under a minimum bound,
        and as very late (24:14) under a maximum-only bound. The default 12:00 minimum gives 00:00-24:00.
        """
        anchor = self.minimum_minutes if self.minimum_minutes is not None else self.maximum_minutes
        return 0.0 if anchor is None else anchor - MINUTES_PER_DAY / 2

    @overload
    def normalize(self, solar_noon_minutes: float) -> float: ...
    @overload
    def normalize(self, solar_noon_minutes: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def normalize(self, solar_noon_minutes: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Map clock times (any day displacement) into this constraint's comparison window."""
        start = self.window_start
        return start + (solar_noon_minutes - start) % MINUTES_PER_DAY

    def compliant(self, solar_noon_minutes: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Return which local solar-noon clock times meet both bounds."""
        values = self.normalize(solar_noon_minutes)
        accepted = np.ones(values.shape, dtype=np.bool_)
        if self.minimum_minutes is not None:
            accepted &= values >= self.minimum_minutes
        if self.maximum_minutes is not None:
            accepted &= values <= self.maximum_minutes
        return accepted

    def accepts(self, solar_noon_minutes: float) -> bool:
        """Return whether one local solar-noon clock time meets both bounds."""
        return bool(self.compliant(np.array([solar_noon_minutes], dtype=np.float64))[0])
