"""NOAA equation-of-time-based solar-noon calculations."""

from __future__ import annotations

import calendar
import math
from datetime import date
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from timezone_experiments.core.constants import MINUTES_PER_DAY, MINUTES_PER_HOUR, mean_solar_noon_utc_minutes

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from timezone_experiments.core.config import SolarMetric


@lru_cache(maxsize=32)
def equation_of_time_minutes(year: int) -> tuple[float, ...]:
    """Return the NOAA equation of time (apparent minus mean solar time, minutes) for each day of ``year``.

    This is the Fourier series from NOAA's "General Solar Position Calculations" (after Spencer, 1971), accurate to
    about half a minute. ``gamma`` is the fractional year in radians at 12:00 of each day (NOAA's
    ``(hour - 12) / 24`` term is zero); 229.18 = 1440 / (2 pi) converts radians to minutes.
    """
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
    """Return the year's equations of time sorted by their phase (value mod 1440), with the phases."""
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
    # With t = (base + shift) mod 1440 and phases p sorted ascending, day i's clock time is (t - p_i) mod 1440:
    # t - p_i for p_i <= t, and t - p_i + 1440 for p_i > t. Both runs decrease with i, so the clock times in
    # phase order are a rotation of a descending sequence: the minimum sits at the last phase <= t (``index``,
    # found by binary search) and the maximum just after it. Walking backwards from the minimum visits the
    # clock times in ascending order, so the k-th smallest is at ``earliest - k`` (mod count).
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
    # The median is the middle rank, or the mean of the two middle ranks for an even number of days.
    return (clock_at(earliest - (count - 1) // 2) + clock_at(earliest - count // 2)) / 2


def winter_solstice_date(latitude: float, year: int) -> date:
    """Return the local hemisphere's winter solstice date.

    The fixed 21 December / 21 June dates are within a day of the true solstices, which is enough for the
    caller: it only needs a date when daylight saving time is off, so a zone's standard offset is read.
    """
    return date(year, 12, 21) if latitude >= 0 else date(year, 6, 21)
