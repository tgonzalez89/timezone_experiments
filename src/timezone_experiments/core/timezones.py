"""Helpers for timezone labels, UTC offsets, and HH:MM clock values."""

from __future__ import annotations

# Offset limits and minute conversions are standard timezone domain values.
import re
import zoneinfo
from calendar import isleap
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import tzdata

from timezone_experiments.core.constants import (
    MAX_CLOCK_HOUR,
    MAX_CLOCK_MINUTE,
    MAX_UTC_OFFSET_HOURS,
    MIN_CLOCK_HOUR,
    MIN_CLOCK_MINUTE,
    MIN_UTC_OFFSET_HOURS,
    MINUTES_PER_HOUR,
    SECONDS_PER_HOUR,
)

_OFFSET_PATTERN = re.compile(r"^(?:UTC)?(?P<sign>[+-])(?P<hours>\d{1,2})(?::?(?P<minutes>\d{2}))?$", re.IGNORECASE)


def tz_database_version() -> str:
    """Return the IANA tz database release that zoneinfo reads: the system TZPATH first, then the tzdata package."""
    for directory in zoneinfo.TZPATH:
        try:
            with (Path(directory) / "tzdata.zi").open(encoding="utf-8") as file:
                header = file.readline()
        except OSError:
            continue
        if header.startswith("# version "):
            return header.removeprefix("# version ").strip()
    return str(tzdata.IANA_VERSION)


def parse_utc_offset(value: str | float, *, at: datetime | None = None) -> float:
    """Parse a UTC offset or IANA timezone label into decimal hours."""
    if isinstance(value, (float, int)):
        return float(value)
    match = _OFFSET_PATTERN.fullmatch(value.strip())
    if match is not None:
        hours = int(match.group("hours"))
        minutes = int(match.group("minutes") or 0)
        if minutes >= MINUTES_PER_HOUR or hours > MAX_UTC_OFFSET_HOURS or (hours == MAX_UTC_OFFSET_HOURS and minutes):
            msg = f"Invalid UTC offset: {value!r}."
            raise ValueError(msg)
        sign = 1 if match.group("sign") == "+" else -1
        return sign * (hours + minutes / MINUTES_PER_HOUR)
    zone = timezone_info(value)
    if at is None:
        instant = datetime.now(UTC)
    elif at.tzinfo is None:
        instant = at.replace(tzinfo=UTC)
    else:
        instant = at.astimezone(UTC)
    offset = instant.astimezone(zone).utcoffset()
    if offset is None:
        msg = f"Timezone {value!r} did not provide a UTC offset."
        raise ValueError(msg)
    return offset.total_seconds() / SECONDS_PER_HOUR


def is_fixed_offset(value: str) -> bool:
    """Return whether a label is a fixed UTC offset such as ``UTC+05:30`` rather than an IANA identifier."""
    return _OFFSET_PATTERN.fullmatch(value.strip()) is not None


def is_known_timezone(value: str) -> bool:
    """Return whether a label is a fixed UTC offset or an IANA zone in the installed tzdata."""
    try:
        parse_utc_offset(value)
    except ValueError:
        return False
    return True


def format_utc_offset(offset: float) -> str:
    """Format a decimal-hour UTC offset as UTC±HH:MM."""
    if not MIN_UTC_OFFSET_HOURS <= offset <= MAX_UTC_OFFSET_HOURS:
        msg = f"UTC offset must be between -14 and +14 hours: {offset}."
        raise ValueError(msg)
    total_minutes = round(abs(offset) * MINUTES_PER_HOUR)
    hours, minutes = divmod(total_minutes, MINUTES_PER_HOUR)
    sign = "+" if offset >= 0 else "-"
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


@cache
def describe_timezone(value: str | None, *, year: int) -> str:
    """Describe standard and daylight offsets observed under a year's IANA rules."""
    if value is None:
        return "Not available"
    if _OFFSET_PATTERN.fullmatch(value.strip()) is not None:
        return f"{format_utc_offset(parse_utc_offset(value))} (DST information unavailable)"
    zone = timezone_info(value)
    start = datetime(year, 1, 1, 12, tzinfo=zone)
    offsets: dict[str, set[float]] = {"standard": set()}
    for day in range(366 if isleap(year) else 365):
        instant = start + timedelta(days=day)
        offset, daylight = instant.utcoffset(), instant.dst()
        if offset is None or daylight is None:
            msg = f"Timezone {value!r} did not provide UTC/DST offsets."
            raise ValueError(msg)
        offsets["standard"].add((offset - daylight).total_seconds() / SECONDS_PER_HOUR)
        if daylight != timedelta(0):
            label = "negative DST" if daylight < timedelta(0) else "DST"
            offsets.setdefault(label, set()).add(offset.total_seconds() / SECONDS_PER_HOUR)
    if len(offsets) == 1:
        standard = " / ".join(format_utc_offset(offset) for offset in sorted(offsets["standard"]))
        return f"{value} ({standard}; no DST)"
    details = [
        f"{label}: {' / '.join(format_utc_offset(offset) for offset in sorted(offsets[label]))}"
        for label in ("standard", "DST", "negative DST")
        if label in offsets
    ]
    return f"{value} ({'; '.join(details)})"


def timezone_info(value: str) -> ZoneInfo:
    """Return the IANA zone for an identifier, with a clear error for unknown names."""
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as error:
        msg = f"Unknown IANA timezone or UTC offset: {value!r}."
        raise ValueError(msg) from error


def parse_clock_minutes(value: str | None) -> int | None:
    """Convert an optional HH:MM clock value to minutes after midnight."""
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"Clock time must be a string or null: {value!r}."
        raise TypeError(msg)
    try:
        hour_text, minute_text = value.split(":")
        hour, minute = int(hour_text), int(minute_text)
    except ValueError as error:
        msg = f"Clock time must use HH:MM format: {value!r}."
        raise ValueError(msg) from error
    if not MIN_CLOCK_HOUR <= hour <= MAX_CLOCK_HOUR or not MIN_CLOCK_MINUTE <= minute <= MAX_CLOCK_MINUTE:
        msg = f"Clock time must be between 00:00 and 23:59: {value!r}."
        raise ValueError(msg)
    return hour * MINUTES_PER_HOUR + minute
