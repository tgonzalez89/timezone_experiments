"""Timezone parsing and descriptions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest
import tzdata

from timezone_experiments.core import timezones
from timezone_experiments.core.timezones import (
    describe_timezone,
    format_utc_offset,
    is_known_timezone,
    parse_utc_offset,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("timezone", "expected"),
    [
        ("America/New_York", "America/New_York (standard: UTC-05:00; DST: UTC-04:00)"),
        ("Australia/Lord_Howe", "Australia/Lord_Howe (standard: UTC+10:30; DST: UTC+11:00)"),
        ("Africa/Casablanca", "Africa/Casablanca (standard: UTC+01:00; negative DST: UTC+00:00)"),
        ("America/Costa_Rica", "America/Costa_Rica (UTC-06:00; no DST)"),
        ("Etc/UTC", "Etc/UTC (UTC+00:00; no DST)"),
        ("UTC+05:45", "UTC+05:45 (DST information unavailable)"),
        (None, "Not available"),
    ],
)
def test_timezone_descriptions_include_actual_seasonal_offsets(timezone: str | None, expected: str) -> None:
    assert describe_timezone(timezone, year=2025) == expected


@pytest.mark.parametrize("timezone", ["Not/A_Zone", "UTC+14:30"])
def test_timezone_descriptions_report_invalid_timezone_data(timezone: str) -> None:
    with pytest.raises(ValueError, match=r"Unknown IANA|Invalid UTC offset"):
        describe_timezone(timezone, year=2026)


def test_known_timezone_detection() -> None:
    assert is_known_timezone("Europe/Madrid")
    assert is_known_timezone("UTC+05:30")
    assert not is_known_timezone("UTC+15")
    assert not is_known_timezone("Mars/Olympus_Mons")


def test_aware_instants_are_converted_not_relabelled() -> None:
    # 05:30 at UTC+05:00 is 00:30 UTC, before Madrid's 01:00 UTC switch to summer time on 2025-03-30.
    instant = datetime(2025, 3, 30, 5, 30, tzinfo=timezone(timedelta(hours=5)))
    assert parse_utc_offset("Europe/Madrid", at=instant) == 1
    assert parse_utc_offset("Europe/Madrid", at=instant.astimezone(UTC).replace(tzinfo=None)) == 1


@pytest.mark.parametrize(("text", "hours"), [("UTC-05:30", -5.5), ("UTC+14", 14), ("UTC+00:00", 0)])
def test_fixed_offsets_parse(text: str, hours: float) -> None:
    assert parse_utc_offset(text) == hours


@pytest.mark.parametrize(("hours", "text"), [(-5.5, "UTC-05:30"), (5.75, "UTC+05:45"), (0, "UTC+00:00")])
def test_offsets_format_as_utc_hours_and_minutes(hours: float, text: str) -> None:
    assert format_utc_offset(hours) == text


@pytest.mark.parametrize(
    ("text", "message"), [("UTC+14:30", "Invalid UTC offset"), ("Not/AZone", "Unknown IANA")], ids=["range", "unknown"]
)
def test_invalid_offsets_and_unknown_zones_are_rejected(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_utc_offset(text)


def test_tz_database_version_prefers_the_system_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unversioned = tmp_path / "unversioned"
    unversioned.mkdir()
    (unversioned / "tzdata.zi").write_text("# no version line\n", encoding="utf-8")
    system = tmp_path / "system"
    system.mkdir()
    (system / "tzdata.zi").write_text("# version 2031z\nZ Etc/UTC 0 - UTC\n", encoding="utf-8")
    monkeypatch.setattr(timezones.zoneinfo, "TZPATH", (str(tmp_path / "missing"), str(unversioned), str(system)))
    assert timezones.tz_database_version() == "2031z"
    monkeypatch.setattr(timezones.zoneinfo, "TZPATH", ())
    assert timezones.tz_database_version() == tzdata.IANA_VERSION
