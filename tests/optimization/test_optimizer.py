"""Offset optimization: candidate order, coverage, fallbacks, and the dateline."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from factories import (
    make_territory,
)

from timezone_experiments.core.config import (
    CoverageType,
    ExperimentConfig,
    SolarMetric,
)
from timezone_experiments.core.models import Territory, TerritorySample
from timezone_experiments.optimization import optimizer as optimizer_module
from timezone_experiments.optimization.optimizer import optimize_territory

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


# Rule the dateline tests' expected offsets and noon values were derived with.
LONGITUDE_95_IN_2025 = ExperimentConfig(coverage_type="longitude", coverage_threshold=0.95, year=2025)


def _point_territory(longitude: float, *, current_timezone: str | None = None) -> Territory:
    return Territory(
        territory_id="TST:1",
        country_code="TST",
        name="Test",
        admin_level=0,
        samples=(TerritorySample(latitude=0, longitude=longitude),),
        current_timezone=current_timezone,
    )


@pytest.mark.parametrize("coverage", ["area", "longitude", "population"])
def test_optimizer_skips_only_weightless_samples(coverage: CoverageType, monkeypatch: pytest.MonkeyPatch) -> None:
    active = TerritorySample(0, 0, area_weight=2, longitude_weight=3, population_weight=10)
    padding = TerritorySample(1, 1, area_weight=0, longitude_weight=0, population_weight=0)
    minimal = make_territory(samples=(active,))
    padded = replace(minimal, samples=(active, padding))
    config = ExperimentConfig(coverage_type=coverage, minimum_noon=None, workers=1)
    expected = optimize_territory(minimal, config)
    original = optimizer_module.annual_solar_noon_minutes
    evaluated: list[int] = []

    def count(longitudes: NDArray[np.float64], year: int, offset: float, metric: SolarMetric) -> object:
        evaluated.append(len(longitudes))
        return original(longitudes, year, offset, metric)

    monkeypatch.setattr(optimizer_module, "annual_solar_noon_minutes", count)
    result = optimize_territory(padded, config)
    assert replace(result, noon=expected.noon) == expected
    assert evaluated == [1]
    assert padded.samples == (active, padding)


def test_optimizer_keeps_incomplete_population_metadata_and_zero_area_fallback() -> None:
    config = ExperimentConfig(coverage_type="area", minimum_noon=None, workers=1)
    incomplete = make_territory(
        samples=(
            TerritorySample(0, 0, population_weight=10),
            TerritorySample(1, 1, area_weight=0, longitude_weight=0),
        )
    )
    assert optimize_territory(incomplete, config).population_covered is None
    zero_area = replace(
        incomplete,
        samples=(
            TerritorySample(0, 0, area_weight=0, longitude_weight=1),
            TerritorySample(0, 15, area_weight=0, longitude_weight=1),
        ),
    )
    uniform_area = replace(zero_area, samples=tuple(replace(sample, area_weight=1) for sample in zero_area.samples))
    longitude_config = replace(config, coverage_type="longitude")
    assert optimize_territory(zero_area, longitude_config) == optimize_territory(uniform_area, longitude_config)


@pytest.mark.parametrize("coverage", ["longitude", "population"])
def test_optimizer_retains_positive_non_area_weights(coverage: CoverageType) -> None:
    territory = make_territory(
        samples=(
            TerritorySample(0, 0, population_weight=1),
            TerritorySample(0, 15, area_weight=0, longitude_weight=1, population_weight=1),
        )
    )
    result = optimize_territory(
        territory,
        ExperimentConfig(
            coverage_type=coverage, coverage_threshold=0.5, minimum_noon="11:01", utc_offsets=(0,), workers=1
        ),
    )
    assert result.compliance_rate == 0.5
    assert result.population_covered == 0.5


@pytest.mark.parametrize("population", [None, 0.0], ids=["population-missing", "population-all-zero"])
def test_population_coverage_without_population_falls_back_to_area(
    caplog: pytest.LogCaptureFixture, population: float | None
) -> None:
    territory = make_territory(
        samples=(
            TerritorySample(0, 0, population_weight=population),
            TerritorySample(1, 1, population_weight=population),
        )
    )
    result = optimize_territory(territory, ExperimentConfig(coverage_type="population", utc_offsets=(1, 2)))
    assert result.coverage_type_used == "area"
    assert "using area coverage" in caplog.text


def test_optimizer_selects_earliest_feasible_offset_and_compares_current_zone() -> None:
    config = ExperimentConfig(utc_offsets=(0, 1, 2), minimum_noon="12:00")
    result = optimize_territory(make_territory(), config)
    assert result.proposed_utc_offset_hours == 0
    assert result.timezone_shift_hours == 0
    assert result.compliance_rate == 1


@pytest.mark.parametrize(("coverage_type", "expected"), [("population", 1), ("longitude", 1), ("area", 2)])
@pytest.mark.parametrize("reverse", [False, True])
def test_earliest_feasible_offset_uses_constraints_not_area_alignment(
    coverage_type: CoverageType, expected: float, *, reverse: bool
) -> None:
    samples = (
        TerritorySample(0, 0, area_weight=1, longitude_weight=9, population_weight=90),
        TerritorySample(0, 25, area_weight=100, longitude_weight=1, population_weight=10),
    )
    territory = make_territory(samples=tuple(reversed(samples)) if reverse else samples)
    config = ExperimentConfig(
        solar_metric="median", coverage_type=coverage_type, coverage_threshold=0.9, utc_offsets=(4, 2, 1.5, 1)
    )
    result = optimize_territory(territory, config)
    assert result.proposed_utc_offset_hours == expected
    assert result.compliance_rate >= 0.9
    if coverage_type == "population":
        assert result.population_covered == pytest.approx(0.9)
        changed_areas = replace(
            territory, samples=tuple(replace(sample, area_weight=101 - sample.area_weight) for sample in samples)
        )
        assert optimize_territory(changed_areas, config).proposed_utc_offset_hours == expected


@pytest.mark.parametrize("maximum", [None, "12:00"])
def test_earliest_feasible_offset_handles_fractional_candidates_without_a_minimum(maximum: str | None) -> None:
    config = ExperimentConfig(solar_metric="median", minimum_noon=None, maximum_noon=maximum, utc_offsets=(1.5, 0.5))
    result = optimize_territory(make_territory(longitude=15), config)
    assert result.proposed_utc_offset_hours == 0.5


@pytest.mark.parametrize("reverse", [False, True])
def test_earliest_feasible_noon_does_not_wrap_late_population_noon_to_midnight(*, reverse: bool) -> None:
    samples = (
        TerritorySample(0, 110, area_weight=100, population_weight=0),
        TerritorySample(0, 145, area_weight=1, population_weight=100),
    )
    territory = make_territory(samples=tuple(reversed(samples)) if reverse else samples, current_timezone="UTC+10")
    config = ExperimentConfig(solar_metric="median", coverage_type="population", coverage_threshold=0.9)
    result = optimize_territory(territory, config)
    assert result.proposed_utc_offset_hours == 10
    assert result.population_covered == 1


def test_default_area_coverage_accepts_exactly_90_percent_and_preserves_overrides() -> None:
    territory = make_territory(
        samples=(
            TerritorySample(0, 0, area_weight=9),
            TerritorySample(0, 30, area_weight=1),
        ),
    )
    config = ExperimentConfig(utc_offsets=(1, 3))
    result = optimize_territory(territory, config)
    assert result.proposed_utc_offset_hours == 1
    assert result.compliance_rate == pytest.approx(0.9)
    assert result.coverage_type_used == "area"
    assert optimize_territory(territory, replace(config, coverage_threshold=1)).proposed_utc_offset_hours == 3


@pytest.mark.parametrize(
    ("longitude", "expected_offset"),
    [(-82.563574, -5), (-77.195996, -5), (-74, -4), (139.7, 10), (-157, -10), (179, 12), (-179, -11)],
)
def test_default_offsets_follow_geography_and_keep_median_noon_near_midday(
    longitude: float, expected_offset: float
) -> None:
    config = ExperimentConfig()
    result = optimize_territory(make_territory(longitude=longitude), config)
    assert result.proposed_utc_offset_hours == expected_offset
    assert 720 <= result.area_weighted_noon_minutes < 780
    assert result.compliance_rate == 1


def test_earliest_feasible_offset_unwraps_the_dateline() -> None:
    config = ExperimentConfig(coverage_type="area")
    dateline = make_territory(samples=(TerritorySample(0, 179), TerritorySample(0, -179)))
    assert optimize_territory(dateline, config).proposed_utc_offset_hours == 13


@pytest.mark.parametrize("current_timezone", [None, "UTC-09:00"])
@pytest.mark.parametrize("reverse", [False, True])
def test_dateline_outlier_uses_full_territory_calendar_date(current_timezone: str | None, *, reverse: bool) -> None:
    samples = (
        TerritorySample(52, 179.7734, area_weight=0.01),
        TerritorySample(64, -152, area_weight=10),
        TerritorySample(55, -129.9942, area_weight=0),
    )
    territory = make_territory(
        samples=tuple(reversed(samples)) if reverse else samples, current_timezone=current_timezone
    )
    result = optimize_territory(territory, LONGITUDE_95_IN_2025)
    assert result.proposed_utc_offset_hours == -8
    assert result.area_weighted_noon_minutes == pytest.approx(848.7334727188363)
    assert result.compliance_rate == 1
    if current_timezone is not None:
        assert result.timezone_shift_hours == 1


@pytest.mark.parametrize(
    ("samples", "current_timezone", "expected_offset"),
    [
        ((TerritorySample(0, 178, area_weight=10), TerritorySample(0, -178)), None, 13),
        ((TerritorySample(0, 178, area_weight=10), TerritorySample(0, -178)), "UTC+12", 13),
        ((TerritorySample(0, 174), TerritorySample(0, -157, area_weight=10)), "UTC+14", 14),
        ((TerritorySample(0, 179, area_weight=0), TerritorySample(0, -150, area_weight=0)), None, -10),
    ],
)
def test_dateline_calendar_reference_preserves_pacific_dates_and_handles_zero_area(
    samples: tuple[TerritorySample, ...], current_timezone: str | None, expected_offset: float
) -> None:
    result = optimize_territory(
        make_territory(samples=samples, current_timezone=current_timezone), LONGITUDE_95_IN_2025
    )
    assert result.proposed_utc_offset_hours == expected_offset
    assert result.compliance_rate == 1


@pytest.mark.parametrize(
    ("longitude", "current_timezone", "expected_offset"),
    [
        (-175, "UTC+13", 13),
        (-178, "UTC+12", 13),
        (-172, "UTC+13", 13),
        (-157, "UTC+14", 14),
        (179, "UTC-11", -12),
    ],
)
def test_calendar_reference_preserves_civil_date_without_crossing_geometry(
    longitude: float, current_timezone: str, expected_offset: float
) -> None:
    config = ExperimentConfig()
    territory = make_territory(longitude=longitude, current_timezone=current_timezone)
    unanchored = optimize_territory(replace(territory, current_timezone=None), config)
    result = optimize_territory(territory, config)
    assert result.proposed_utc_offset_hours == expected_offset
    assert result.timezone_shift_hours is not None
    assert abs(result.timezone_shift_hours) <= 1
    assert result.area_weighted_noon_minutes == pytest.approx(unanchored.area_weighted_noon_minutes)
    assert result.compliance_rate == unanchored.compliance_rate == 1


def test_optimizer_falls_back_to_maximum_coverage_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    territory = make_territory(
        samples=(TerritorySample(0, 0), TerritorySample(0, 4)),
        longitude=0,
        current_timezone="UTC+00:00",
    )
    config = ExperimentConfig(
        minimum_noon="12:00",
        maximum_noon="12:30",
        coverage_threshold=1,
        utc_offsets=(1, 0),
    )
    result = optimize_territory(territory, config)
    assert result.proposed_utc_offset_hours == 0
    assert result.compliance_rate == 0.5
    assert result.fallback_used
    assert "No UTC offset meets 100.0% area coverage" in caplog.text
    assert "using UTC+00:00 with 50.0% maximum coverage" in caplog.text


def test_optimizer_fallback_ties_use_existing_earliest_noon_order(caplog: pytest.LogCaptureFixture) -> None:
    result = optimize_territory(
        make_territory(),
        ExperimentConfig(minimum_noon="23:00", coverage_threshold=1, utc_offsets=(1, 0)),
    )
    assert result.proposed_utc_offset_hours == 0
    assert result.compliance_rate == 0
    assert result.fallback_used


@pytest.mark.parametrize(("longitude", "expected"), [(0, 0), (-75, -5), (100, 7), (179, 12)])
@pytest.mark.parametrize(("minimum", "maximum"), [(None, None), (None, "13:00"), ("12:00", None)])
def test_offsets_stay_near_the_solar_meridian_without_a_minimum_bound(
    longitude: float, expected: float, minimum: str | None, maximum: str | None
) -> None:
    config = ExperimentConfig(minimum_noon=minimum, maximum_noon=maximum)
    result = optimize_territory(_point_territory(longitude), config)
    assert result.proposed_utc_offset_hours == expected
    assert 660 <= result.area_weighted_noon_minutes <= 780
    assert not result.fallback_used


def test_maximum_only_bound_picks_the_latest_feasible_noon() -> None:
    config = ExperimentConfig(minimum_noon=None, maximum_noon="13:01")
    result = optimize_territory(_point_territory(0), config)
    assert result.proposed_utc_offset_hours == 1
    assert 720 < result.area_weighted_noon_minutes <= 781


def test_minimum_bound_picks_the_earliest_feasible_noon() -> None:
    config = ExperimentConfig(minimum_noon="11:00", maximum_noon=None)
    result = optimize_territory(_point_territory(0), config)
    assert result.proposed_utc_offset_hours == -1
    assert 660 <= result.area_weighted_noon_minutes < 720


def test_noon_statistics_use_the_configured_year() -> None:
    config = ExperimentConfig(year=2031)
    result = optimize_territory(_point_territory(0, current_timezone="Europe/London"), config)
    assert result.solar_year == 2031
    assert result.noon.year == 2031


def test_unknown_current_timezone_is_ignored_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    result = optimize_territory(_point_territory(0, current_timezone="Mars/Olympus_Mons"), ExperimentConfig())
    assert result.current_timezone is None
    assert result.timezone_shift_hours is None
    assert "Unknown timezone 'Mars/Olympus_Mons' for TST:1" in caplog.text
