"""Shared domain constants for geographic coordinates and clock values."""

import re
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0
HALF_LONGITUDE_TURN = 180.0
FULL_LONGITUDE_TURN = 360.0

MIN_UTC_OFFSET_HOURS = -14.0
MAX_UTC_OFFSET_HOURS = 14.0
MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 1440
SECONDS_PER_HOUR = 3600
# The Earth turns 15 degrees of longitude per hour, so mean solar noon moves 4 minutes per degree.
DEGREES_PER_HOUR = 15.0
MINUTES_PER_DEGREE = 4.0
# Mean solar noon at Greenwich, in UTC minutes after midnight.
SOLAR_NOON_UTC_MINUTES = 720.0
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


# Allowed values for resolution settings shared by the configuration, CLI, dashboard, and map/raster code.
POPULATION_CELL_DEGREES = (0.1, 0.25, 0.5, 1.0)
EVENT_MAP_CELL_DEGREES = (0.25, 0.5, 1.0, 2.0)
MAP_SIMPLIFICATION_DEGREES = (0.0, 0.005, 0.01, 0.02)
DEFAULT_MAP_SIMPLIFICATION_DEGREES = 0.01


def validate_choice[T](name: str, value: T, choices: tuple[T, ...], error: type[ValueError] = ValueError) -> T:
    """Return ``value`` if it is one of ``choices`` (booleans never match numbers), else raise ``error``."""
    if isinstance(value, bool) or value not in choices:
        msg = f"{name} must be one of {', '.join(map(str, choices))}: {value!r}."
        raise error(msg)
    return value


@overload
def mean_solar_noon_utc_minutes(longitude: float) -> float: ...
@overload
def mean_solar_noon_utc_minutes(longitude: NDArray[np.float64]) -> NDArray[np.float64]: ...
def mean_solar_noon_utc_minutes(longitude: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Return mean solar noon (ignoring the equation of time) in UTC minutes after midnight.

    Mean noon is 12:00 UTC on the prime meridian, and the Sun moves west 1 degree every 4 minutes
    (1440 minutes / 360 degrees), so it is 4 minutes earlier for each degree east.
    """
    return SOLAR_NOON_UTC_MINUTES - MINUTES_PER_DEGREE * longitude


def require_positive_workers(workers: int, stage: str) -> None:
    """Reject a non-positive worker count for a parallel stage (e.g. "Timezone overlay")."""
    if workers < 1:
        msg = f"{stage} workers must be positive."
        raise ValueError(msg)
