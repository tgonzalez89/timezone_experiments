"""Persistable annual noon statistics for a complete territory."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

from timezone_experiments.astronomy.solar import equation_of_time_minutes
from timezone_experiments.constants import DEGREES_PER_HOUR, MINUTES_PER_HOUR, mean_solar_noon_utc_minutes
from timezone_experiments.data.geometry import iter_polygons, ring_winding, unwrap_longitude, unwrap_ring
from timezone_experiments.data.timezones import current_noon_ranges, describe_timezone

if TYPE_CHECKING:
    from timezone_experiments.data.models import Territory


@dataclass(frozen=True, slots=True)
class NoonStatistics:
    """Unrounded minutes relative to midnight, including signed day displacement."""

    year: int
    proposed_earliest_minutes: float
    proposed_latest_minutes: float
    current_timezone_description: str
    current_ranges: dict[str, tuple[float, float]]
    median_equation_of_time_minutes: float


def calculate_noon_statistics(territory: Territory, offset: float, *, year: int) -> NoonStatistics:
    """Calculate full-boundary extrema once, independent of optimization coverage."""
    if territory.geometry is None:
        longitudes = (sample.longitude for sample in territory.samples)
    else:
        longitudes = (
            longitude
            for polygon in iter_polygons(territory.geometry)
            if ring_winding(unwrap_ring(polygon[0])) != 0
            for longitude, _ in polygon[0]
        )
    west, east = float("inf"), float("-inf")
    for longitude in longitudes:
        unwrapped = unwrap_longitude(longitude, offset * DEGREES_PER_HOUR)
        west, east = min(west, unwrapped), max(east, unwrapped)
    if west > east:
        msg = f"No nonzero-area boundary or sample points for noon statistics: {territory.territory_id!r}."
        raise ValueError(msg)
    equations = equation_of_time_minutes(year)
    return NoonStatistics(
        year=year,
        proposed_earliest_minutes=mean_solar_noon_utc_minutes(east) - max(equations) + MINUTES_PER_HOUR * offset,
        proposed_latest_minutes=mean_solar_noon_utc_minutes(west) - min(equations) + MINUTES_PER_HOUR * offset,
        current_timezone_description=describe_timezone(territory.current_timezone, year=year),
        current_ranges=(
            current_noon_ranges(territory.current_timezone, year=year, west=west, east=east)
            if territory.current_timezone is not None
            else {}
        ),
        median_equation_of_time_minutes=float(median(equations)),
    )
