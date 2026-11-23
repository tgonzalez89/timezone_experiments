"""Approximate annual sunrise/sunset extremes over a flat apparent horizon."""

from __future__ import annotations

# NOAA solar-declination coefficients and the conventional 90.833-degree zenith.
import math
from dataclasses import dataclass, replace
from datetime import date, timedelta
from functools import lru_cache

from timezone_experiments.astronomy.solar import _equation_of_time_series
from timezone_experiments.constants import MAX_LATITUDE, MAX_LONGITUDE, MIN_LATITUDE, MIN_LONGITUDE
from timezone_experiments.data.geometry import _unwrap_longitude

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


def annual_solar_events(latitude: float, longitude: float, year: int, utc_offset: float) -> AnnualSolarEvents:
    """Return annual extrema without wrapping clock times across midnight.

    Use a solar zenith of 90.833 degrees (standard refraction and solar radius),
    ignoring terrain, elevation, and weather. Dates identify the solar cycle;
    minutes outside 0-1440 preserve the event's displacement from that date.
    """
    if (
        not MIN_LATITUDE <= latitude <= MAX_LATITUDE
        or not MIN_LONGITUDE <= longitude <= MAX_LONGITUDE
        or not math.isfinite(utc_offset)
    ):
        msg = "Solar events require valid latitude/longitude and a finite UTC offset."
        raise ValueError(msg)
    profile = _latitude_events(latitude, year)
    shift = 60 * utc_offset - 4 * _unwrap_longitude(longitude, utc_offset * 15)

    def shifted(event: SolarEvent | None) -> SolarEvent | None:
        return None if event is None else replace(event, minutes=event.minutes + shift)

    return AnnualSolarEvents(
        shifted(profile.earliest_sunrise),
        shifted(profile.latest_sunrise),
        shifted(profile.earliest_sunset),
        shifted(profile.latest_sunset),
        profile.polar_day_days,
        profile.polar_night_days,
    )


@lru_cache(maxsize=32)
def _declination_series(year: int) -> tuple[float, ...]:
    days = len(_equation_of_time_series(year))
    values = []
    for day in range(days):
        gamma = 2 * math.pi * day / days
        values.append(
            0.006918
            - 0.399912 * math.cos(gamma)
            + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma)
            + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma)
            + 0.00148 * math.sin(3 * gamma)
        )
    return tuple(values)


@lru_cache(maxsize=2048)
def _latitude_events(latitude: float, year: int) -> AnnualSolarEvents:
    start = date(year, 1, 1)
    phi = math.radians(latitude)
    horizon = math.cos(math.radians(90.833))
    rises: list[SolarEvent] = []
    sets: list[SolarEvent] = []
    polar_day = polar_night = 0
    for day, (declination, equation) in enumerate(
        zip(_declination_series(year), _equation_of_time_series(year), strict=True)
    ):
        cosine = (horizon - math.sin(phi) * math.sin(declination)) / (math.cos(phi) * math.cos(declination))
        if cosine < -1:
            polar_day += 1
        elif cosine > 1:
            polar_night += 1
        else:
            half_day = 4 * math.degrees(math.acos(cosine))
            solar_date = start + timedelta(days=day)
            rises.append(SolarEvent(720 - equation - half_day, solar_date))
            sets.append(SolarEvent(720 - equation + half_day, solar_date))
    return AnnualSolarEvents(
        min(rises, key=lambda event: event.minutes, default=None),
        max(rises, key=lambda event: event.minutes, default=None),
        min(sets, key=lambda event: event.minutes, default=None),
        max(sets, key=lambda event: event.minutes, default=None),
        polar_day,
        polar_night,
    )
