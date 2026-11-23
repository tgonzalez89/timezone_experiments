"""Verify UTC-offset selection under each combination of solar-noon bounds."""

from __future__ import annotations

import pytest

from timezone_experiments.config import ExperimentConfig
from timezone_experiments.data.models import Territory, TerritorySample
from timezone_experiments.optimization.constraints import NoonConstraint
from timezone_experiments.optimization.optimizer import optimize_territory


def _point_territory(longitude: float, *, current_timezone: str | None = None) -> Territory:
    return Territory(
        territory_id="TST:1",
        country_code="TST",
        name="Test",
        admin_level=0,
        samples=(TerritorySample(latitude=0, longitude=longitude),),
        current_timezone=current_timezone,
    )


@pytest.mark.parametrize(("longitude", "expected"), [(0, 0), (-75, -5), (100, 7), (179, 12)])
@pytest.mark.parametrize(("minimum", "maximum"), [(None, None), (None, "13:00"), ("12:00", None)])
def test_offsets_stay_near_the_solar_meridian_without_a_minimum_bound(
    longitude: float, expected: float, minimum: str | None, maximum: str | None
) -> None:
    config = ExperimentConfig(minimum_noon=minimum, maximum_noon=maximum)
    result = optimize_territory(_point_territory(longitude), config)
    assert result.proposed_utc_offset == expected
    assert 660 <= result.solar_noon_statistic <= 780
    assert not result.fallback_used


def test_maximum_only_bound_picks_the_latest_feasible_noon() -> None:
    config = ExperimentConfig(minimum_noon=None, maximum_noon="13:01")
    result = optimize_territory(_point_territory(0), config)
    assert result.proposed_utc_offset == 1
    assert 720 < result.solar_noon_statistic <= 781


def test_minimum_bound_picks_the_earliest_feasible_noon() -> None:
    config = ExperimentConfig(minimum_noon="11:00", maximum_noon=None)
    result = optimize_territory(_point_territory(0), config)
    assert result.proposed_utc_offset == -1
    assert 660 <= result.solar_noon_statistic < 720


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


@pytest.mark.parametrize(
    ("constraint", "minutes", "accepted"),
    [
        # Just after midnight is far too early under a minimum bound ...
        (NoonConstraint(720, None), 14, False),
        (NoonConstraint(720, None), 1430, True),
        # ... and far too late (24:14) under a maximum-only bound.
        (NoonConstraint(None, 780), 14, False),
        (NoonConstraint(None, 780), 779, True),
        (NoonConstraint(720, 780), 750, True),
        (NoonConstraint(None, None), 14, True),
    ],
)
def test_constraint_compares_clock_times_in_a_window_centered_on_its_bounds(
    constraint: NoonConstraint, minutes: float, *, accepted: bool
) -> None:
    assert constraint.accepts(minutes) is accepted
    assert constraint.accepts(minutes + 1440) is accepted
