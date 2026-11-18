"""Helpers for current timezone labels and UTC offsets."""

from __future__ import annotations

# Offset limits and minute conversions are standard timezone domain values.
# ruff: noqa: PLR2004
import re
from calendar import isleap
from datetime import UTC, datetime, timedelta
from functools import cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from timezone_experiments.astronomy.solar import _equation_of_time_series

_OFFSET_PATTERN = re.compile(r"^(?:UTC)?(?P<sign>[+-])(?P<hours>\d{1,2})(?::?(?P<minutes>\d{2}))?$", re.IGNORECASE)


def parse_utc_offset(value: str | float, *, at: datetime | None = None) -> float:
    """Parse a UTC offset or IANA timezone label into decimal hours."""
    if isinstance(value, (float, int)):
        return float(value)
    match = _OFFSET_PATTERN.fullmatch(value.strip())
    if match is not None:
        hours = int(match.group("hours"))
        minutes = int(match.group("minutes") or 0)
        if minutes >= 60 or hours > 14 or (hours == 14 and minutes):
            msg = f"Invalid UTC offset: {value!r}."
            raise ValueError(msg)
        sign = 1 if match.group("sign") == "+" else -1
        return sign * (hours + minutes / 60)
    zone = _timezone(value)
    instant = at.replace(tzinfo=UTC) if at is not None else datetime.now(UTC)
    offset = instant.astimezone(zone).utcoffset()
    if offset is None:
        msg = f"Timezone {value!r} did not provide a UTC offset."
        raise ValueError(msg)
    return offset.total_seconds() / 3600


def format_utc_offset(offset: float) -> str:
    """Format a decimal-hour UTC offset as UTC±HH:MM."""
    if not -14 <= offset <= 14:
        msg = f"UTC offset must be between -14 and +14 hours: {offset}."
        raise ValueError(msg)
    total_minutes = round(abs(offset) * 60)
    hours, minutes = divmod(total_minutes, 60)
    sign = "+" if offset >= 0 else "-"
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


@cache
def describe_timezone(value: str | None, *, year: int) -> str:
    """Describe standard and daylight offsets observed under a year's IANA rules."""
    if value is None:
        return "Not available"
    if _OFFSET_PATTERN.fullmatch(value.strip()) is not None:
        return f"{format_utc_offset(parse_utc_offset(value))} (DST information unavailable)"
    zone = _timezone(value)
    start = datetime(year, 1, 1, 12, tzinfo=zone)
    offsets: dict[str, set[float]] = {"standard": set()}
    for day in range(366 if isleap(year) else 365):
        instant = start + timedelta(days=day)
        offset, daylight = instant.utcoffset(), instant.dst()
        if offset is None or daylight is None:
            msg = f"Timezone {value!r} did not provide UTC/DST offsets."
            raise ValueError(msg)
        offsets["standard"].add((offset - daylight).total_seconds() / 3600)
        if daylight != timedelta(0):
            label = "negative DST" if daylight < timedelta(0) else "DST"
            offsets.setdefault(label, set()).add(offset.total_seconds() / 3600)
    if len(offsets) == 1:
        standard = " / ".join(format_utc_offset(offset) for offset in sorted(offsets["standard"]))
        return f"{value} ({standard}; no DST)"
    details = [
        f"{label}: {' / '.join(format_utc_offset(offset) for offset in sorted(offsets[label]))}"
        for label in ("standard", "DST", "negative DST")
        if label in offsets
    ]
    return f"{value} ({'; '.join(details)})"


def current_noon_ranges(value: str, *, year: int, west: float, east: float) -> dict[str, tuple[float, float]]:
    """Return annual endpoint noon ranges, separated by observed offset and DST state."""
    equations = _equation_of_time_series(year)
    if _OFFSET_PATTERN.fullmatch(value.strip()) is not None:
        offset = parse_utc_offset(value)
        return {
            f"Fixed offset {format_utc_offset(offset)} (DST information unavailable)": (
                720 - 4 * east - max(equations) + 60 * offset,
                720 - 4 * west - min(equations) + 60 * offset,
            )
        }
    zone = _timezone(value)
    start = datetime(year, 1, 1, tzinfo=UTC)
    ranges: dict[str, tuple[float, float]] = {}
    # Include adjacent UTC dates, then restrict to the requested local civil year.
    for day in range(-2, len(equations) + 2):
        date = start + timedelta(days=day)
        equation = _equation_of_time_series(date.year)[date.timetuple().tm_yday - 1]
        for longitude in (east, west):
            utc_minutes = 720 - 4 * longitude - equation
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
            offset_hours = offset.total_seconds() / 3600
            label = f"{label} ({format_utc_offset(offset_hours)})"
            noon = utc_minutes + 60 * offset_hours
            earliest, latest = ranges.get(label, (noon, noon))
            ranges[label] = min(earliest, noon), max(latest, noon)
    return ranges


def _timezone(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as error:
        msg = f"Unknown IANA timezone or UTC offset: {value!r}."
        raise ValueError(msg) from error
