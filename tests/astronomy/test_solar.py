"""Equation of time, solstices, and the fast annual solar-noon statistics."""

from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pytest
from factories import (
    daily_noon,
    statistic,
)

from timezone_experiments.astronomy import solar as solar_module
from timezone_experiments.astronomy.solar import winter_solstice_date

if TYPE_CHECKING:
    from timezone_experiments.core.config import SolarMetric


def test_winter_solstice_follows_the_hemisphere() -> None:
    assert winter_solstice_date(1, 2025) == date(2025, 12, 21)
    assert winter_solstice_date(-1, 2025) == date(2025, 6, 21)


@pytest.mark.parametrize("year", [2024, 2025])
def test_fast_annual_minimum_matches_daily_series_including_midnight_rounding(year: int) -> None:
    equations = solar_module.equation_of_time_minutes(year)
    midnight_longitude = (720 - max(equations)) / 4
    longitudes = [
        -180,
        -157,
        0,
        45,
        179.999,
        midnight_longitude,
        math.nextafter(midnight_longitude, -math.inf),
        math.nextafter(midnight_longitude, math.inf),
    ]
    for offset in (-12, -11.5, 0, 5.75, 14):
        expected = [min(value % 1440 for value in daily_noon(longitude, year, offset)) for longitude in longitudes]
        actual = solar_module.annual_solar_noon_minutes(np.array(longitudes), year, offset, "min")
        assert actual.tolist() == expected


@pytest.mark.parametrize("year", [2024, 2025])
@pytest.mark.parametrize(
    "metric",
    [
        "median",
        "max",
    ],
)
def test_fast_annual_statistics_match_daily_series_at_clock_wraps(year: int, metric: SolarMetric) -> None:
    equations = solar_module.equation_of_time_minutes(year)
    longitudes = [-180, -157, 0, 45, 179.999]
    for equation in equations:
        longitude = (720 - equation) / 4
        if -180 <= longitude <= 180:
            longitudes.extend((longitude, math.nextafter(longitude, -math.inf), math.nextafter(longitude, math.inf)))
    for offset in (-12, -11.5, 0, 5.75, 14):
        expected = [
            statistic((value % 1440 for value in daily_noon(longitude, year, offset)), metric)
            for longitude in longitudes
        ]
        actual = solar_module.annual_solar_noon_minutes(np.array(longitudes), year, offset, metric)
        assert actual.tolist() == expected
