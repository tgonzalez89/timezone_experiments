"""Shared domain constants for geographic coordinates and clock values."""

import re

MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0
HALF_LONGITUDE_TURN = 180.0
FULL_LONGITUDE_TURN = 360.0

MIN_UTC_OFFSET_HOURS = -14.0
MAX_UTC_OFFSET_HOURS = 14.0
MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = 3600
MIN_CLOCK_HOUR = 0
MAX_CLOCK_HOUR = 23
MIN_CLOCK_MINUTE = 0
MAX_CLOCK_MINUTE = 59

MIN_POLYGON_RING_POINTS = 3
POSITION_DIMENSIONS = 2
GEOMETRY_AREA_EPSILON = 1e-12

MIN_SOLAR_YEAR = 1900
MAX_SOLAR_YEAR = 2200

# ISO 3166 alpha-3 plus GADM's alphanumeric codes for disputed areas (e.g. Z01).
_COUNTRY_CODE_PATTERN = re.compile(r"[A-Z][A-Z0-9]{2}")


def normalize_country_code(value: str) -> str:
    """Upper-case a three-character country code, rejecting anything unsafe for URLs and file names."""
    code = value.strip().upper()
    if _COUNTRY_CODE_PATTERN.fullmatch(code) is None:
        msg = f"Country codes must be three letters or digits, such as 'USA': {value!r}."
        raise ValueError(msg)
    return code
