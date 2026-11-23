"""NOAA equation-of-time-based solar-noon calculations."""

from __future__ import annotations

# Domain values are literal astronomical bounds and coefficients.
import calendar
import math
from datetime import date
from functools import lru_cache
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from timezone_experiments.constants import MAX_LATITUDE, MAX_LONGITUDE, MIN_LATITUDE, MIN_LONGITUDE

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from timezone_experiments.config import SolarMetric


@lru_cache(maxsize=32)
def _equation_of_time_series(year: int) -> tuple[float, ...]:
    days_in_year = 366 if calendar.isleap(year) else 365
    values: list[float] = []
    for day_number in range(1, days_in_year + 1):
        gamma = 2 * math.pi / days_in_year * (day_number - 1)
        values.append(
            229.18
            * (
                0.000075
                + 0.001868 * math.cos(gamma)
                - 0.032077 * math.sin(gamma)
                - 0.014615 * math.cos(2 * gamma)
                - 0.040849 * math.sin(2 * gamma)
            ),
        )
    return tuple(values)


@lru_cache(maxsize=16384)
def _solar_noon_base_series(longitude: float, year: int) -> tuple[float, ...]:
    return tuple(720 - 4 * longitude - equation for equation in _equation_of_time_series(year))


@lru_cache(maxsize=32)
def _ordered_equations_of_time(year: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ordered = sorted((value % 1440, value) for value in _equation_of_time_series(year))
    phases = np.array([phase for phase, _ in ordered])
    equations = np.array([value for _, value in ordered])
    phases.flags.writeable = equations.flags.writeable = False
    return phases, equations


def annual_solar_noon_minutes(
    longitudes: NDArray[np.float64], year: int, utc_offset: float, metric: SolarMetric | str
) -> NDArray[np.float64]:
    """Return each longitude's annual min/median/max solar-noon clock time (minutes in [0, 1440)).

    Equivalent to taking the statistic of the 365/366 daily values ``(720 - 4 * lon - EoT + 60 * offset) % 1440``
    without materializing them: EoT values sorted by phase make the daily clock times a rotation of a
    descending sequence, so a binary search finds the earliest one and the rest are fixed offsets from it.
    """
    phases, equations = _ordered_equations_of_time(year)
    count = len(equations)
    base = 720 - 4 * np.asarray(longitudes, dtype=np.float64)
    shift = 60 * utc_offset
    index = np.searchsorted(phases, (base + shift) % 1440, side="right") - 1
    # The neighbours absorb floating-point ties at the 0/1440 wrap, matching the daily-series arithmetic.
    neighbours = np.stack((index - 1, index, (index + 1) % count))
    clock = (base - equations[neighbours] + shift) % 1440
    if metric == "min":
        return clock.min(axis=0)
    if metric not in ("median", "max"):
        msg = f"Unsupported solar-noon metric: {metric!r}."
        raise ValueError(msg)
    earliest = np.take_along_axis(neighbours, clock.argmin(axis=0)[np.newaxis], axis=0)[0]
    if metric == "max":
        return (base - equations[(earliest + 1) % count] + shift) % 1440
    lower, upper = (count - 1) // 2, count // 2
    return (
        (base - equations[(earliest - lower) % count] + shift) % 1440
        + (base - equations[(earliest - upper) % count] + shift) % 1440
    ) / 2


def annual_solar_noon_series(
    latitude: float,
    longitude: float,
    year: int = 2025,
    utc_offset: float = 0.0,
) -> tuple[float, ...]:
    """Return local solar-noon clock times for every day of a year."""
    if not MIN_LATITUDE <= latitude <= MAX_LATITUDE:
        msg = f"Latitude must be between -90 and 90: {latitude}."
        raise ValueError(msg)
    if not MIN_LONGITUDE <= longitude <= MAX_LONGITUDE:
        msg = f"Longitude must be between -180 and 180: {longitude}."
        raise ValueError(msg)
    del latitude
    return tuple(value + 60 * utc_offset for value in _solar_noon_base_series(longitude, year))


def solar_noon_statistic(values: Iterable[float], metric: SolarMetric | str) -> float:
    """Calculate a supported annual solar-noon statistic."""
    data = tuple(values)
    if not data:
        msg = "Solar-noon statistics require at least one value."
        raise ValueError(msg)
    if metric == "min":
        return min(data)
    if metric == "max":
        return max(data)
    if metric == "median":
        return float(median(data))
    msg = f"Unsupported solar-noon metric: {metric!r}."
    raise ValueError(msg)


def winter_solstice_date(latitude: float, year: int) -> date:
    """Return the local hemisphere's winter solstice date."""
    return date(year, 12, 21) if latitude >= 0 else date(year, 6, 21)
