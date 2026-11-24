"""Noon statistics and current-timezone noon ranges."""

from __future__ import annotations

from typing import TYPE_CHECKING

from factories import make_territory

from timezone_experiments.core.models import TerritorySample
from timezone_experiments.optimization.assessment import evaluate_coverage

if TYPE_CHECKING:
    from timezone_experiments.core.config import CoverageType

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from factories import (
    daily_noon,
)

from timezone_experiments.core.timezones import format_utc_offset
from timezone_experiments.optimization.assessment import (
    current_noon_ranges,
)


@pytest.mark.parametrize(
    ("timezone", "west", "east"),
    [
        ("America/New_York", -80, -70),
        ("Australia/Lord_Howe", 159, 160),
        ("Europe/Dublin", -10, -6),
        ("Pacific/Apia", -172, -171),
    ],
)
def test_current_noon_ranges_match_daily_events_in_each_season(timezone: str, west: float, east: float) -> None:
    year = 2024
    zone = ZoneInfo(timezone)
    expected: dict[str, list[float]] = {}
    start = datetime(year, 1, 1, tzinfo=UTC)
    for day in range(-2, 368):
        midnight = start + timedelta(days=day)
        index = midnight.timetuple().tm_yday - 1
        for longitude in (west, east):
            noon = daily_noon(longitude, midnight.year)[index]
            local = (midnight + timedelta(minutes=noon)).astimezone(zone)
            if local.year != year:
                continue
            daylight, offset = local.dst(), local.utcoffset()
            assert daylight is not None
            assert offset is not None
            state = (
                "Standard time" if daylight == timedelta(0) else "DST" if daylight > timedelta(0) else "Negative DST"
            )
            key = f"{state} ({format_utc_offset(offset.total_seconds() / 3600)})"
            expected.setdefault(key, []).append(noon + offset.total_seconds() / 60)
    actual = current_noon_ranges(timezone, year=year, west=west, east=east)
    assert actual.keys() == expected.keys()
    for key, values in expected.items():
        assert actual[key] == pytest.approx((min(values), max(values)), abs=1e-10)
    if timezone == "America/New_York":
        fixed = next(iter(current_noon_ranges("UTC-04:00", year=year, west=west, east=east).values()))
        assert actual["DST (UTC-04:00)"][1] < fixed[1]


@pytest.mark.parametrize(
    ("samples", "compliant", "coverage_type", "message"),
    [
        ((TerritorySample(0, 0),), [True], "population", "missing population"),
        ((TerritorySample(0, 0),), [], "area", "correspond one-to-one"),
        ((TerritorySample(0, 0, area_weight=0, longitude_weight=0),), [False], "longitude", "positive"),
    ],
    ids=["population-missing", "length-mismatch", "zero-total-weight"],
)
def test_weighted_coverage_rejects_unusable_inputs(
    samples: tuple[TerritorySample, ...], compliant: list[bool], coverage_type: CoverageType, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_coverage(make_territory(samples=samples), compliant, coverage_type)
