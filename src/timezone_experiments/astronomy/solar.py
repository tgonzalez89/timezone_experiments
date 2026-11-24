"""NOAA equation-of-time-based solar-noon calculations."""

from __future__ import annotations

import calendar
import math
from datetime import date
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from timezone_experiments.constants import MINUTES_PER_DAY, MINUTES_PER_HOUR, mean_solar_noon_utc_minutes

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from timezone_experiments.config import SolarMetric


@lru_cache(maxsize=32)
def equation_of_time_minutes(year: int) -> tuple[float, ...]:
    """Return the NOAA equation of time (apparent minus mean solar time, minutes) for each day of ``year``."""
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


@lru_cache(maxsize=32)
def _ordered_equations_of_time(year: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ordered = sorted((value % MINUTES_PER_DAY, value) for value in equation_of_time_minutes(year))
    phases = np.array([phase for phase, _ in ordered])
    equations = np.array([value for _, value in ordered])
    phases.flags.writeable = equations.flags.writeable = False
    return phases, equations


def annual_solar_noon_minutes(
    longitudes: NDArray[np.float64], year: int, utc_offset_hours: float, metric: SolarMetric
) -> NDArray[np.float64]:
    """Return each longitude's annual min/median/max solar-noon clock time (minutes in [0, 1440)).

    Equivalent to taking the statistic of the 365/366 daily clock times
    ``(mean_solar_noon_utc_minutes(lon) - EoT + 60 * offset) % 1440`` without materializing them: EoT values
    sorted by phase make the daily clock times a rotation of a descending sequence, so a binary search finds
    the earliest one and the rest are fixed offsets from it.
    """
    phases, equations = _ordered_equations_of_time(year)
    count = len(equations)
    base = mean_solar_noon_utc_minutes(np.asarray(longitudes, dtype=np.float64))
    shift = MINUTES_PER_HOUR * utc_offset_hours
    index = np.searchsorted(phases, (base + shift) % MINUTES_PER_DAY, side="right") - 1
    # The neighbours absorb floating-point ties at the 0/1440 wrap, matching the daily-series arithmetic.
    neighbours = np.stack((index - 1, index, (index + 1) % count))
    clock = (base - equations[neighbours] + shift) % MINUTES_PER_DAY
    if metric == "min":
        return clock.min(axis=0)
    earliest = np.take_along_axis(neighbours, clock.argmin(axis=0)[np.newaxis], axis=0)[0]

    def clock_at(position: NDArray[np.intp]) -> NDArray[np.float64]:
        return (base - equations[position % count] + shift) % MINUTES_PER_DAY

    if metric == "max":
        return clock_at(earliest + 1)
    return (clock_at(earliest - (count - 1) // 2) + clock_at(earliest - count // 2)) / 2


def winter_solstice_date(latitude: float, year: int) -> date:
    """Return the local hemisphere's winter solstice date."""
    return date(year, 12, 21) if latitude >= 0 else date(year, 6, 21)
