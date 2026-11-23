"""NOAA equation-of-time-based solar-noon calculations."""

from __future__ import annotations

# Domain values are literal astronomical bounds and coefficients.
import calendar
import math
from bisect import bisect_right
from datetime import date
from functools import lru_cache
from statistics import median
from typing import TYPE_CHECKING

from timezone_experiments.constants import MAX_LATITUDE, MAX_LONGITUDE, MIN_LATITUDE, MIN_LONGITUDE

if TYPE_CHECKING:
    from collections.abc import Iterable

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
def _ordered_equations_of_time(year: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    ordered = sorted((value % 1440, value) for value in _equation_of_time_series(year))
    return tuple(phase for phase, _ in ordered), tuple(value for _, value in ordered)


def _annual_solar_noon_minimum(longitude: float, year: int, utc_offset: float) -> float:
    phases, equations = _ordered_equations_of_time(year)
    base = 720 - 4 * longitude
    shift = 60 * utc_offset
    index = bisect_right(phases, (base + shift) % 1440) - 1
    # Check adjacent phases to preserve daily-series rounding at midnight.
    return min(
        (base - equations[neighbor] + shift) % 1440 for neighbor in (index - 1, index, (index + 1) % len(equations))
    )


def _annual_solar_noon_statistic(longitude: float, year: int, utc_offset: float, metric: SolarMetric | str) -> float:
    if metric == "min":
        return _annual_solar_noon_minimum(longitude, year, utc_offset)
    if metric not in ("median", "max"):
        msg = f"Unsupported solar-noon metric: {metric!r}."
        raise ValueError(msg)
    phases, equations = _ordered_equations_of_time(year)
    base = 720 - 4 * longitude
    shift = 60 * utc_offset
    phase = (base + shift) % 1440
    index = bisect_right(phases, phase) - 1
    # Clock order is a rotation of descending EOT phases; use original arithmetic at midnight.
    index = min(
        (index - 1, index, (index + 1) % len(equations)),
        key=lambda neighbor: (base - equations[neighbor] + shift) % 1440,
    )
    if metric == "median":
        middle = ((len(equations) - 1) // 2, len(equations) // 2)
        return sum((base - equations[(index - rank) % len(equations)] + shift) % 1440 for rank in middle) / 2
    return (base - equations[(index + 1) % len(equations)] + shift) % 1440


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
