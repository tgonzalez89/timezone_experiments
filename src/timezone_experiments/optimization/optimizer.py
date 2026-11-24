"""Search UTC offsets that satisfy configurable solar-noon constraints."""

from __future__ import annotations

# Longitude and solar-offset constants are standard geographic domain values.
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from timezone_experiments.astronomy.solar import annual_solar_noon_minutes, winter_solstice_date
from timezone_experiments.core.constants import (
    DEGREES_PER_HOUR,
    HALF_LONGITUDE_TURN,
    MINUTES_PER_HOUR,
    SOLAR_NOON_UTC_MINUTES,
    mean_solar_noon_utc_minutes,
)
from timezone_experiments.core.geometry import unwrap_longitudes
from timezone_experiments.core.timezones import format_utc_offset, is_known_timezone, parse_utc_offset
from timezone_experiments.optimization.assessment import (
    CoverageSummary,
    CoverageWeights,
    NoonStatistics,
    calculate_noon_statistics,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from timezone_experiments.core.config import CoverageType, ExperimentConfig
    from timezone_experiments.core.models import NoonConstraint, Territory

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AssignmentResult:
    """Optimized timezone assignment and comparison statistics."""

    territory_id: str
    country_code: str
    territory_name: str
    admin_level: int
    current_timezone: str | None
    proposed_timezone: str
    proposed_utc_offset_hours: float
    timezone_shift_hours: float | None
    area_weighted_noon_minutes: float
    compliance_rate: float
    population_covered: float | None
    coverage_type_used: str
    fallback_used: bool
    solar_year: int
    noon: NoonStatistics


@dataclass(frozen=True, slots=True)
class OffsetEvaluation:
    """Coverage achieved by one candidate offset, with noon times for each distinct sample longitude."""

    offset: float
    noon_minutes: NDArray[np.float64]
    coverage: CoverageSummary


def optimize_territory(territory: Territory, config: ExperimentConfig) -> AssignmentResult:
    """Choose the offset whose solar noon hugs the configured bounds while meeting the coverage threshold.

    With a minimum bound the earliest feasible noon wins; with only a maximum bound the latest feasible
    noon wins; with no bounds the offset nearest the territory's solar meridian wins. When no offset meets
    the threshold, the one with the highest coverage is used and ``fallback_used`` is set.
    """
    territory = _without_unknown_timezone(territory)
    coverage_type = _effective_coverage_type(territory, config.coverage_type)
    analyzed = _analysis_samples(territory, coverage_type)
    current_offset = _current_offset(territory, config.year)
    reference_longitude = _solar_longitude_reference(territory, current_offset)
    longitudes = _sample_values(analyzed, "longitude")
    calendar_reference = reference_longitude if reference_longitude is not None else float(longitudes[0])
    unwrapped = unwrap_longitudes(longitudes, calendar_reference)
    constraint = config.noon_constraint
    offsets = _candidate_order(
        config.utc_offsets,
        constraint,
        west=float(unwrapped.min()),
        east=float(unwrapped.max()),
        center=calendar_reference,
    )
    # Solar noon depends only on longitude, and grid samples share longitudes, so evaluate each distinct one once.
    distinct_longitudes, sample_longitude = np.unique(longitudes, return_inverse=True)
    weights = CoverageWeights.from_territory(analyzed, coverage_type)
    evaluations: list[OffsetEvaluation] = []
    for offset in offsets:
        noon = constraint.normalize(
            annual_solar_noon_minutes(distinct_longitudes, config.year, offset, config.solar_metric)
        )
        evaluation = OffsetEvaluation(offset, noon, weights.evaluate(constraint.compliant(noon)[sample_longitude]))
        if evaluation.coverage.compliance_rate + 1e-12 >= config.coverage_threshold:
            chosen, fallback_used = evaluation, False
            break
        evaluations.append(evaluation)
    else:
        # max() keeps the first (most preferred) offset among equally covering ones.
        chosen, fallback_used = max(evaluations, key=lambda item: item.coverage.compliance_rate), True
        logger.warning(
            "No UTC offset meets %.1f%% %s coverage for %s; using %s with %.1f%% maximum coverage.",
            config.coverage_threshold * 100,
            coverage_type,
            territory.territory_id,
            format_utc_offset(chosen.offset),
            chosen.coverage.compliance_rate * 100,
        )
    return AssignmentResult(
        territory_id=territory.territory_id,
        country_code=territory.country_code,
        territory_name=territory.name,
        admin_level=territory.admin_level,
        current_timezone=territory.current_timezone,
        proposed_timezone=format_utc_offset(chosen.offset),
        proposed_utc_offset_hours=chosen.offset,
        timezone_shift_hours=None if current_offset is None else chosen.offset - current_offset,
        area_weighted_noon_minutes=_weighted_mean(
            chosen.noon_minutes[sample_longitude], _sample_values(analyzed, "area_weight")
        ),
        compliance_rate=chosen.coverage.compliance_rate,
        population_covered=chosen.coverage.population_covered,
        coverage_type_used=coverage_type,
        fallback_used=fallback_used,
        solar_year=config.year,
        noon=calculate_noon_statistics(territory, chosen.offset, year=config.year),
    )


def _without_unknown_timezone(territory: Territory) -> Territory:
    if territory.current_timezone is None or is_known_timezone(territory.current_timezone):
        return territory
    logger.warning(
        "Unknown timezone %r for %s (newer than the installed tzdata?); ignoring it.",
        territory.current_timezone,
        territory.territory_id,
    )
    return replace(territory, current_timezone=None)


def _effective_coverage_type(territory: Territory, coverage_type: CoverageType) -> CoverageType:
    if coverage_type == "population" and (
        any(sample.population_weight is None for sample in territory.samples)
        or sum(sample.population_weight or 0 for sample in territory.samples) <= 0
    ):
        logger.warning("Population weights are unavailable for %s; using area coverage.", territory.territory_id)
        return "area"
    return coverage_type


def _analysis_samples(territory: Territory, coverage_type: CoverageType) -> Territory:
    """Drop samples that carry no weight for the analysis (e.g. zero-weight padding).

    Territories with mixed population metadata (some samples None, some not) or without any area samples
    are kept intact, since dropping samples there would change what the weights mean.
    """
    population_states = {sample.population_weight is None for sample in territory.samples}
    if len(population_states) != 1 or not any(sample.area_weight > 0 for sample in territory.samples):
        return territory
    return replace(
        territory,
        samples=tuple(
            sample
            for sample in territory.samples
            if sample.area_weight > 0
            or (coverage_type == "longitude" and sample.longitude_weight > 0)
            or (sample.population_weight or 0) > 0
        ),
    )


def _current_offset(territory: Territory, year: int) -> float | None:
    """Standard-time UTC offset of the territory's current zone, read at the local winter solstice."""
    if territory.current_timezone is None:
        return None
    reference_latitude = _weighted_mean(_sample_values(territory, "latitude"), _sample_values(territory, "area_weight"))
    winter_date = winter_solstice_date(reference_latitude, year)
    return parse_utc_offset(
        territory.current_timezone,
        at=datetime(winter_date.year, winter_date.month, winter_date.day, tzinfo=UTC),
    )


def _candidate_order(
    offsets: Sequence[float],
    constraint: NoonConstraint,
    *,
    west: float,
    east: float,
    center: float,
) -> list[float]:
    """Order offsets so the first one meeting the coverage threshold is the preferred answer."""

    def noon(offset: float, longitude: float) -> float:
        return constraint.normalize(mean_solar_noon_utc_minutes(longitude) + MINUTES_PER_HOUR * offset)

    def primary(offset: float) -> float:
        if constraint.minimum_minutes is not None:
            return noon(offset, east)  # The eastern edge sees the earliest noon.
        if constraint.maximum_minutes is not None:
            return -noon(offset, west)  # The western edge sees the latest noon.
        return abs(noon(offset, center) - SOLAR_NOON_UTC_MINUTES)

    # Offsets 24 h apart (e.g. -10 and +14) share clock times; prefer the one nearest the territory's meridian.
    return sorted(offsets, key=lambda offset: (primary(offset), abs(offset - center / DEGREES_PER_HOUR), offset))


def _solar_longitude_reference(territory: Territory, current_offset: float | None) -> float | None:
    longitudes = _sample_values(territory, "longitude")
    if longitudes.max() - longitudes.min() <= HALF_LONGITUDE_TURN and current_offset is None:
        return None
    longitude = _weighted_mean(
        unwrap_longitudes(longitudes, float(longitudes[0])), _sample_values(territory, "area_weight")
    )
    # Keep +180 for a territory centered exactly on the date line.
    longitude = 180 - (180 - longitude) % 360
    if current_offset is not None:
        longitude += 360 * round((current_offset * DEGREES_PER_HOUR - longitude) / 360)
    return longitude


def _sample_values(territory: Territory, field: str) -> NDArray[np.float64]:
    return np.fromiter((getattr(sample, field) for sample in territory.samples), np.float64, len(territory.samples))


def _weighted_mean(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    """Weighted mean, falling back to the plain mean when no weight is positive."""
    total = float(weights.sum())
    if total <= 0:
        return float(values.mean())
    return float(np.dot(values, weights)) / total
