"""Judging a territory under a candidate offset: the noon rule, weighted coverage, and saved noon statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from timezone_experiments.astronomy.solar import equation_of_time_minutes
from timezone_experiments.core.constants import (
    DEGREES_PER_HOUR,
    MINUTES_PER_HOUR,
    SECONDS_PER_HOUR,
    mean_solar_noon_utc_minutes,
)
from timezone_experiments.core.geometry import iter_polygons, ring_winding, unwrap_longitude, unwrap_ring
from timezone_experiments.core.timezones import (
    describe_timezone,
    format_utc_offset,
    is_fixed_offset,
    parse_utc_offset,
    timezone_info,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from timezone_experiments.core.config import CoverageType
    from timezone_experiments.core.models import Territory


# --- Weighted coverage -----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """Fraction of territory/population satisfying a solar constraint."""

    compliance_rate: float
    population_covered: float | None


@dataclass(frozen=True, slots=True)
class CoverageWeights:
    """Per-sample weights extracted once per territory, so each candidate offset is just a masked sum."""

    weights: NDArray[np.float64]
    total: float
    population: NDArray[np.float64] | None
    population_total: float

    @classmethod
    def from_territory(cls, territory: Territory, coverage_type: CoverageType) -> CoverageWeights:
        """Collect the weights ``coverage_type`` measures, plus population weights when every sample has one."""
        samples = territory.samples
        if coverage_type == "longitude":
            weights = np.fromiter((sample.longitude_weight for sample in samples), np.float64, len(samples))
        elif coverage_type == "area":
            weights = np.fromiter((sample.area_weight for sample in samples), np.float64, len(samples))
        elif coverage_type == "population":
            if any(sample.population_weight is None for sample in samples):
                msg = (
                    f"Population coverage requested but territory {territory.territory_id!r} "
                    "has missing population weights."
                )
                raise ValueError(msg)
            weights = np.fromiter((sample.population_weight or 0.0 for sample in samples), np.float64, len(samples))
        else:
            msg = f"Unsupported coverage type: {coverage_type!r}."
            raise ValueError(msg)
        total = float(weights.sum())
        if total <= 0:
            msg = f"Territory {territory.territory_id!r} has no positive {coverage_type} weights."
            raise ValueError(msg)
        population = None
        if all(sample.population_weight is not None for sample in samples):
            population = np.fromiter((sample.population_weight or 0.0 for sample in samples), np.float64, len(samples))
        population_total = 0.0 if population is None else float(population.sum())
        return cls(weights, total, population, population_total)

    def evaluate(self, compliant: NDArray[np.bool_]) -> CoverageSummary:
        """Summarize coverage for a boolean mask aligned with the territory's samples."""
        if compliant.shape != self.weights.shape:
            msg = "Compliance values must correspond one-to-one with territory samples."
            raise ValueError(msg)
        population_covered = None
        if self.population is not None and self.population_total:
            population_covered = float(self.population[compliant].sum()) / self.population_total
        return CoverageSummary(float(self.weights[compliant].sum()) / self.total, population_covered)


def evaluate_coverage(
    territory: Territory,
    compliant: Sequence[bool],
    coverage_type: CoverageType,
) -> CoverageSummary:
    """Compute weighted compliance and optional population coverage."""
    if len(compliant) != len(territory.samples):
        msg = "Compliance values must correspond one-to-one with territory samples."
        raise ValueError(msg)
    return CoverageWeights.from_territory(territory, coverage_type).evaluate(np.asarray(compliant, dtype=np.bool_))


# --- Saved noon statistics -------------------------------------------------------------------------------------------


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
    # Measure longitudes around the proposed offset's meridian, so a territory across the date line from it gets
    # one continuous west-east span and noon times near the local day instead of a whole day off.
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


def current_noon_ranges(value: str, *, year: int, west: float, east: float) -> dict[str, tuple[float, float]]:
    """Return annual endpoint noon ranges, separated by observed offset and DST state."""
    equations = equation_of_time_minutes(year)
    if is_fixed_offset(value):
        offset = parse_utc_offset(value)
        return {
            f"Fixed offset {format_utc_offset(offset)} (DST information unavailable)": (
                mean_solar_noon_utc_minutes(east) - max(equations) + MINUTES_PER_HOUR * offset,
                mean_solar_noon_utc_minutes(west) - min(equations) + MINUTES_PER_HOUR * offset,
            )
        }
    zone = timezone_info(value)
    start = datetime(year, 1, 1, tzinfo=UTC)
    ranges: dict[str, tuple[float, float]] = {}
    # Include adjacent UTC dates, then restrict to the requested local civil year.
    for day in range(-2, len(equations) + 2):
        date = start + timedelta(days=day)
        equation = equation_of_time_minutes(date.year)[date.timetuple().tm_yday - 1]
        for longitude in (east, west):
            utc_minutes = mean_solar_noon_utc_minutes(longitude) - equation
            instant = (date + timedelta(minutes=utc_minutes)).astimezone(zone)
            if instant.year != year:
                continue
            offset, daylight = instant.utcoffset(), instant.dst()
            if offset is None or daylight is None:
                msg = f"Timezone {value!r} did not provide UTC/DST offsets."
                raise ValueError(msg)
            label = (
                "Standard time" if daylight == timedelta(0) else "DST" if daylight > timedelta(0) else "Negative DST"
            )
            offset_hours = offset.total_seconds() / SECONDS_PER_HOUR
            label = f"{label} ({format_utc_offset(offset_hours)})"
            noon = utc_minutes + MINUTES_PER_HOUR * offset_hours
            earliest, latest = ranges.get(label, (noon, noon))
            ranges[label] = min(earliest, noon), max(latest, noon)
    return ranges
