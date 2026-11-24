"""Runtime data layout: one data directory holding downloads/ and cache/."""

from pathlib import Path

DEFAULT_DATA_DIRECTORY = Path("data")
TIMEZONES_FILENAME = "world_timezones.geojson"


def downloads_directory(data_directory: Path) -> Path:
    """Return where downloaded inputs are stored inside a data directory."""
    return data_directory / "downloads"


def cache_directory(data_directory: Path) -> Path:
    """Return where prepared-input and map caches are stored inside a data directory."""
    return data_directory / "cache"


def boundary_filename(source: str, country_code: str, level: int) -> str:
    """Return the download file name for one provider's country boundaries at an administrative level."""
    return f"boundaries_{source}_{country_code}_{level}.geojson"


def world_filename(*, include_antarctica: bool) -> str:
    """Return the download file name for the Natural Earth world country layer."""
    return "world_countries_with_antarctica.geojson" if include_antarctica else "world_countries.geojson"
