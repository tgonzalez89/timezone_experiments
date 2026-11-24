"""Approximate annual sunrise/sunset extremes over a flat apparent horizon."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, timedelta
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from timezone_experiments.astronomy.solar import equation_of_time_minutes
from timezone_experiments.core.constants import (
    DEGREES_PER_HOUR,
    MAX_LATITUDE,
    MAX_LONGITUDE,
    MIN_LATITUDE,
    MIN_LONGITUDE,
    MINUTES_PER_DEGREE,
    MINUTES_PER_HOUR,
    SOLAR_NOON_UTC_MINUTES,
)
from timezone_experiments.core.geometry import unwrap_longitude

if TYPE_CHECKING:
    from numpy.typing import NDArray

SOLAR_EVENT_FIELDS = ("earliest_sunrise", "latest_sunrise", "earliest_sunset", "latest_sunset")


@dataclass(frozen=True, slots=True)
class SolarEvent:
    """Minutes relative to midnight on the associated solar-cycle date."""

    minutes: float
    date: date


@dataclass(frozen=True, slots=True)
class AnnualSolarEvents:
    """Extremes over days with events, plus days without horizon crossings."""

    earliest_sunrise: SolarEvent | None
    latest_sunrise: SolarEvent | None
    earliest_sunset: SolarEvent | None
    latest_sunset: SolarEvent | None
    polar_day_days: int
    polar_night_days: int


def annual_solar_events(latitude: float, longitude: float, year: int, utc_offset_hours: float) -> AnnualSolarEvents:
    """Return annual extrema without wrapping clock times across midnight.

    Use a solar zenith of 90.833 degrees (standard refraction and solar radius),
    ignoring terrain, elevation, and weather. Dates identify the solar cycle;
    minutes outside 0-1440 preserve the event's displacement from that date.
    """
    if (
        not MIN_LATITUDE <= latitude <= MAX_LATITUDE
        or not MIN_LONGITUDE <= longitude <= MAX_LONGITUDE
        or not math.isfinite(utc_offset_hours)
    ):
        msg = "Solar events require valid latitude/longitude and a finite UTC offset."
        raise ValueError(msg)
    # Rounding to ~0.1 m lets nearby sample points (e.g. representative points of map cells) share the profile.
    profile = _latitude_events(round(latitude, 6), year)
    # Measure longitude around the offset's meridian (e.g. +190 rather than -170 for UTC+13), so places across the
    # date line from it keep event times near the local date instead of jumping a whole day.
    shift = MINUTES_PER_HOUR * utc_offset_hours - MINUTES_PER_DEGREE * unwrap_longitude(
        longitude, utc_offset_hours * DEGREES_PER_HOUR
    )

    def shifted(event: SolarEvent | None) -> SolarEvent | None:
        return None if event is None else replace(event, minutes=event.minutes + shift)

    return replace(
        profile,
        earliest_sunrise=shifted(profile.earliest_sunrise),
        latest_sunrise=shifted(profile.latest_sunrise),
        earliest_sunset=shifted(profile.earliest_sunset),
        latest_sunset=shifted(profile.latest_sunset),
    )


@lru_cache(maxsize=32)
def _daily_series(year: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the year's daily solar declinations (radians) and equations of time (minutes)."""
    equations = np.array(equation_of_time_minutes(year))
    gamma = 2 * np.pi * np.arange(len(equations)) / len(equations)
    # NOAA "General Solar Position Calculations" declination series (radians), on the same fractional year as the
    # equation of time.
    declinations = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    declinations.flags.writeable = equations.flags.writeable = False
    return declinations, equations


@lru_cache(maxsize=8192)
def _latitude_events(latitude: float, year: int) -> AnnualSolarEvents:
    # Sunrise/sunset hour angle H: cos H = (cos z - sin(phi) sin(delta)) / (cos(phi) cos(delta)), with the zenith
    # z = 90.833 degrees (34' of refraction plus the 16' solar radius). cos H < -1 means the Sun never sets
    # (polar day) and cos H > 1 that it never rises (polar night). H in degrees times 4 minutes is half the day.
    declinations, equations = _daily_series(year)
    phi = math.radians(latitude)
    cosine = (math.cos(math.radians(90.833)) - math.sin(phi) * np.sin(declinations)) / (
        math.cos(phi) * np.cos(declinations)
    )
    polar_days = {"polar_day_days": int(np.sum(cosine < -1)), "polar_night_days": int(np.sum(cosine > 1))}
    days = np.flatnonzero(np.abs(cosine) <= 1)
    if not days.size:
        return AnnualSolarEvents(
            earliest_sunrise=None, latest_sunrise=None, earliest_sunset=None, latest_sunset=None, **polar_days
        )
    half_days = MINUTES_PER_DEGREE * np.degrees(np.arccos(cosine[days]))
    rises = SOLAR_NOON_UTC_MINUTES - equations[days] - half_days
    sets = SOLAR_NOON_UTC_MINUTES - equations[days] + half_days
    start = date(year, 1, 1)

    def event(minutes: NDArray[np.float64], position: int) -> SolarEvent:
        return SolarEvent(float(minutes[position]), start + timedelta(days=int(days[position])))

    return AnnualSolarEvents(
        earliest_sunrise=event(rises, int(np.argmin(rises))),
        latest_sunrise=event(rises, int(np.argmax(rises))),
        earliest_sunset=event(sets, int(np.argmin(sets))),
        latest_sunset=event(sets, int(np.argmax(sets))),
        **polar_days,
    )
