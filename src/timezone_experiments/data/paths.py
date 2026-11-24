"""Runtime data layout: one data directory holding downloads/ and cache/.

Download file names include the pinned dataset version (see data/download/sources.json), so changing a pin
fetches the new release instead of reusing a file downloaded for the old one.
"""

from pathlib import Path

from timezone_experiments.data.download.catalog import SOURCES

DEFAULT_DATA_DIRECTORY = Path("data")


def downloads_directory(data_directory: Path) -> Path:
    """Return where downloaded inputs are stored inside a data directory."""
    return data_directory / "downloads"


def cache_directory(data_directory: Path) -> Path:
    """Return where prepared-input and map caches are stored inside a data directory."""
    return data_directory / "cache"


def boundary_filename(source: str, country_code: str, level: int) -> str:
    """Return the download file name for one provider's country boundaries at an administrative level."""
    version = SOURCES[source].version
    label = source if version is None else f"{source}-{version}"
    return f"boundaries_{label}_{country_code}_{level}.geojson"


def world_filename(*, include_antarctica: bool) -> str:
    """Return the download file name for the Natural Earth world country layer."""
    version = SOURCES["natural_earth"].version
    return f"world_countries{'_with_antarctica' if include_antarctica else ''}_{version}.geojson"


def timezones_filename() -> str:
    """Return the download file name for the timezone-boundary-builder land polygons."""
    return f"world_timezones_{SOURCES['timezone_boundary_builder'].version}.geojson"


def population_filename() -> str:
    """Return the file name of the population CSV aggregated from the GHSL raster."""
    return f"population_{SOURCES['ghsl_population'].version}_E2025.csv"


def population_archive_filename() -> str:
    """Return the file name of the downloaded GHSL population archive."""
    return f"ghsl_population_{SOURCES['ghsl_population'].version}_E2025.zip"
