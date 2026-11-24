"""Annual sunrise and sunset extremes."""

from __future__ import annotations

import pytest

from timezone_experiments.astronomy.events import SOLAR_EVENT_FIELDS, annual_solar_events
from timezone_experiments.maps.scales import clock


def test_annual_solar_events_capture_london_seasonal_extremes_not_averages() -> None:
    events = annual_solar_events(51.5, 0, 2025, 0)
    # Approximate London times with permanent UTC, not summer DST.
    for field, expected, months in zip(
        SOLAR_EVENT_FIELDS, (223, 486, 951, 1221), ((6,), (1, 12), (12,), (6,)), strict=True
    ):
        event = getattr(events, field)
        assert event is not None
        assert event.minutes == pytest.approx(expected, abs=3)
        assert event.date.month in months
    assert events.polar_day_days == events.polar_night_days == 0


def test_solar_events_preserve_fractional_offsets_day_displacement_and_dateline_aliases() -> None:
    baseline = annual_solar_events(70, 0, 2024, 0)
    shifted = annual_solar_events(70, 15, 2024, 5.75)
    for field in SOLAR_EVENT_FIELDS:
        original, actual = getattr(baseline, field), getattr(shifted, field)
        assert original is not None
        assert actual is not None
        assert actual.minutes == pytest.approx(original.minutes + 285)
        assert actual.date == original.date
    latest = shifted.latest_sunset
    assert latest is not None
    assert latest.minutes > 1440
    assert "(+1 day)" in clock(latest.minutes)
    assert annual_solar_events(10, -179, 2024, 14) == annual_solar_events(10, -179, 2024, -10)


@pytest.mark.parametrize("latitude", [-80, 80])
def test_polar_extremes_use_only_real_event_days_and_support_leap_years(latitude: float) -> None:
    events = annual_solar_events(latitude, 0, 2024, 0)
    assert 0 < events.polar_day_days < 366
    assert 0 < events.polar_night_days < 366
    assert events.polar_day_days + events.polar_night_days < 366
    minutes = []
    for field in SOLAR_EVENT_FIELDS:
        event = getattr(events, field)
        assert event is not None
        assert event.date.year == 2024
        minutes.append(event.minutes)
    assert minutes[0] < minutes[1]
    assert minutes[2] < minutes[3]


@pytest.mark.parametrize(("latitude", "longitude", "offset"), [(91, 0, 0), (0, 181, 0), (0, 0, float("inf"))])
def test_solar_events_reject_invalid_coordinates_and_offsets(latitude: float, longitude: float, offset: float) -> None:
    with pytest.raises(ValueError, match="valid latitude/longitude"):
        annual_solar_events(latitude, longitude, 2025, offset)


@pytest.mark.parametrize(("year", "days"), [(2024, 366), (2025, 365)])
@pytest.mark.parametrize("latitude", [90, -90])
def test_at_the_poles_there_are_no_events_and_polar_day_and_night_fill_the_year(
    latitude: float, year: int, days: int
) -> None:
    events = annual_solar_events(latitude, 0, year, 0)
    assert all(getattr(events, field) is None for field in SOLAR_EVENT_FIELDS)
    assert events.polar_day_days + events.polar_night_days == days
    assert events.polar_day_days > 0
    assert events.polar_night_days > 0


@pytest.mark.parametrize("latitude", [66.5, -66.5])
def test_at_the_polar_circles_refraction_gives_midnight_sun_but_no_polar_night(latitude: float) -> None:
    events = annual_solar_events(latitude, 0, 2024, 0)
    assert all(getattr(events, field) is not None for field in SOLAR_EVENT_FIELDS)
    assert events.polar_night_days == 0
    assert 0 < events.polar_day_days < 60
