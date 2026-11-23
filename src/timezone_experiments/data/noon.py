"""Persistable annual noon statistics for a complete territory."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

from timezone_experiments.astronomy.solar import _equation_of_time_series
from timezone_experiments.data.geometry import _ring_winding, _unwrap_longitude, _unwrap_ring, iter_polygons
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
            if _ring_winding(_unwrap_ring(polygon[0])) != 0
            for longitude, _ in polygon[0]
        )
    west, east = float("inf"), float("-inf")
    for longitude in longitudes:
        unwrapped = _unwrap_longitude(longitude, offset * 15)
        west, east = min(west, unwrapped), max(east, unwrapped)
    if west > east:
        msg = f"No nonzero-area boundary or sample points for noon statistics: {territory.territory_id!r}."
        raise ValueError(msg)
    equations = _equation_of_time_series(year)
    return NoonStatistics(
        year=year,
        proposed_earliest_minutes=720 - 4 * east - max(equations) + 60 * offset,
        proposed_latest_minutes=720 - 4 * west - min(equations) + 60 * offset,
        current_timezone_description=describe_timezone(territory.current_timezone, year=year),
        current_ranges=(
            current_noon_ranges(territory.current_timezone, year=year, west=west, east=east)
            if territory.current_timezone is not None
            else {}
        ),
        median_equation_of_time_minutes=float(median(equations)),
    )
