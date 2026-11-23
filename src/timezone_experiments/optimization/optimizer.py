"""Search UTC offsets that satisfy configurable solar-noon constraints."""

from __future__ import annotations

# Longitude and solar-offset constants are standard geographic domain values.
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from timezone_experiments.astronomy.solar import (
    _annual_solar_noon_statistic,
    winter_solstice_date,
)
from timezone_experiments.constants import HALF_LONGITUDE_TURN
from timezone_experiments.data.geometry import _unwrap_longitude
from timezone_experiments.data.noon import NoonStatistics, calculate_noon_statistics
from timezone_experiments.data.timezones import format_utc_offset, parse_utc_offset
from timezone_experiments.optimization.constraints import NoonConstraint
from timezone_experiments.territory.coverage import CoverageSummary, evaluate_coverage

if TYPE_CHECKING:
    from timezone_experiments.config import ExperimentConfig
    from timezone_experiments.data.models import Territory

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AssignmentResult:
    """Optimized timezone assignment and comparison statistics."""

    territory_id: str
    country: str
    region: str
    admin_level: int
    current_timezone: str | None
    proposed_timezone: str
    proposed_utc_offset: float
    timezone_shift_hours: float | None
    solar_noon_statistic: float
    compliance_rate: float
    population_covered: float | None
    coverage_type_used: str
    fallback_used: bool
    solar_year: int
    noon: NoonStatistics


def optimize_territory(territory: Territory, config: ExperimentConfig) -> AssignmentResult:
    """Choose the earliest solar-noon offset satisfying the configured constraints."""
    coverage_type = config.coverage_type
    if coverage_type == "population" and (
        any(sample.population_weight is None for sample in territory.samples)
        or sum(sample.population_weight or 0 for sample in territory.samples) <= 0
    ):
        logger.warning(
            "Population weights are unavailable for %s; using area coverage.",
            territory.territory_id,
        )
        coverage_type = "area"
    analyzed = territory
    population_states = {sample.population_weight is None for sample in territory.samples}
    # Keep mixed/missing population metadata and sample-only territories intact.
    if len(population_states) == 1 and any(sample.area_weight > 0 for sample in territory.samples):
        analyzed = replace(
            territory,
            samples=tuple(
                sample
                for sample in territory.samples
                if sample.area_weight > 0
                or (coverage_type == "longitude" and sample.longitude_weight > 0)
                or (sample.population_weight or 0) > 0
            ),
        )
    current_offset = None
    if territory.current_timezone is not None:
        reference_latitude = _weighted_mean_latitude(territory)
        winter_date = winter_solstice_date(reference_latitude, config.year)
        current_offset = parse_utc_offset(
            territory.current_timezone,
            at=datetime(winter_date.year, winter_date.month, winter_date.day, tzinfo=UTC),
        )
    reference_longitude = _solar_longitude_reference(territory, current_offset)
    calendar_reference = reference_longitude if reference_longitude is not None else analyzed.samples[0].longitude
    easternmost = max(_unwrap_longitude(sample.longitude, calendar_reference) for sample in analyzed.samples)
    reference_offset = calendar_reference / 15
    # Order solar clock times first; use the calendar frame only to resolve whole-day aliases.
    offsets = sorted(
        config.utc_offsets,
        key=lambda offset: ((offset % 24 - easternmost / 15 + 12) % 24, abs(offset - reference_offset), offset),
    )
    constraint = NoonConstraint.from_clocks(config.minimum_noon, config.maximum_noon)
    best: tuple[float, list[float], CoverageSummary] | None = None
    for chosen_offset in offsets:
        noon_values = [
            _annual_solar_noon_statistic(sample.longitude, config.year, chosen_offset, config.solar_metric)
            for sample in analyzed.samples
        ]
        compliant = [constraint.accepts(value) for value in noon_values]
        coverage = evaluate_coverage(analyzed, compliant, coverage_type)
        if best is None or coverage.compliance_rate > best[2].compliance_rate:
            best = chosen_offset, noon_values, coverage
        if coverage.compliance_rate + 1e-12 >= config.coverage_threshold:
            fallback_used = False
            break
    else:
        if best is None:
            msg = f"No UTC offsets were evaluated for {territory.territory_id!r}."
            raise RuntimeError(msg)
        chosen_offset, noon_values, coverage = best
        fallback_used = True
        logger.warning(
            "No UTC offset meets %.1f%% %s coverage for %s; using %s with %.1f%% maximum coverage.",
            config.coverage_threshold * 100,
            coverage_type,
            territory.territory_id,
            format_utc_offset(chosen_offset),
            coverage.compliance_rate * 100,
        )
    weights = [sample.area_weight for sample in analyzed.samples]
    if sum(weights) <= 0:
        weights = [1.0] * len(analyzed.samples)
    noon_statistic = sum(value * weight for value, weight in zip(noon_values, weights, strict=True)) / sum(weights)
    return AssignmentResult(
        territory_id=territory.territory_id,
        country=territory.country_code,
        region=territory.name,
        admin_level=territory.admin_level,
        current_timezone=territory.current_timezone,
        proposed_timezone=format_utc_offset(chosen_offset),
        proposed_utc_offset=chosen_offset,
        timezone_shift_hours=None if current_offset is None else chosen_offset - current_offset,
        solar_noon_statistic=noon_statistic,
        compliance_rate=coverage.compliance_rate,
        population_covered=coverage.population_covered,
        coverage_type_used=coverage_type,
        fallback_used=fallback_used,
        solar_year=config.year,
        noon=calculate_noon_statistics(territory, chosen_offset, year=datetime.now(UTC).year),
    )


def _solar_longitude_reference(territory: Territory, current_offset: float | None) -> float | None:
    longitudes = [sample.longitude for sample in territory.samples]
    if max(longitudes) - min(longitudes) <= HALF_LONGITUDE_TURN and current_offset is None:
        return None
    weights = [sample.area_weight for sample in territory.samples]
    if sum(weights) <= 0:
        weights = [1.0] * len(territory.samples)
    longitude = sum(
        _unwrap_longitude(value, longitudes[0]) * weight for value, weight in zip(longitudes, weights, strict=True)
    ) / sum(weights)
    # Keep +180 for a territory centered exactly on the date line.
    longitude = 180 - (180 - longitude) % 360
    if current_offset is not None:
        longitude += 360 * round((current_offset * 15 - longitude) / 360)
    return longitude


def _weighted_mean_latitude(territory: Territory) -> float:
    weights = [sample.area_weight for sample in territory.samples]
    if sum(weights) <= 0:
        weights = [1.0] * len(territory.samples)
    return sum(sample.latitude * weight for sample, weight in zip(territory.samples, weights, strict=True)) / sum(
        weights
    )
