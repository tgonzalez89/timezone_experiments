"""Download one administrative boundary dataset for an experiment."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from timezone_experiments.core.constants import normalize_country_code
from timezone_experiments.core.files import write_bytes_atomic
from timezone_experiments.data.download.catalog import SOURCES, write_source_metadata
from timezone_experiments.data.download.http import download_bytes
from timezone_experiments.data.paths import boundary_filename

if TYPE_CHECKING:
    from pathlib import Path

BOUNDARY_SOURCES = ("gadm", "geoboundaries")
# Fields of a GeoBoundaries API answer that identify the build that was downloaded.
_GEOBOUNDARIES_BUILD_FIELDS = (
    "boundaryID",
    "boundaryYearRepresented",
    "boundarySource",
    "boundaryLicense",
    "sourceDataUpdateDate",
    "buildDate",
)


def download(source: str, country: str, level: int, output_directory: Path) -> Path:
    """Fetch a GeoJSON boundary dataset and return its local path."""
    country = normalize_country_code(country)
    if source not in BOUNDARY_SOURCES:
        msg = f"Unsupported boundary source: {source!r}."
        raise ValueError(msg)
    if level < 0:
        msg = "Administrative level must be non-negative."
        raise ValueError(msg)
    dataset = SOURCES[source]
    details: dict[str, object] = {"country": country, "level": level}
    if source == "gadm":
        url = dataset.resolve(country=country, level=level)
        data = download_bytes(url)
    else:
        # GeoBoundaries only serves its current build; record the build the API describes instead of a pin.
        api_url = dataset.resolve(country=country, level=level)
        metadata = json.loads(download_bytes(api_url))
        url = metadata.get("gjDownloadURL") if isinstance(metadata, dict) else None
        if not isinstance(metadata, dict) or not isinstance(url, str) or not url:
            msg = f"GeoBoundaries did not return a GeoJSON download URL for {country} ADM{level}."
            raise ValueError(msg)
        data = download_bytes(url)
        details |= {"api_url": api_url} | {key: metadata.get(key) for key in _GEOBOUNDARIES_BUILD_FIELDS}
    path = output_directory / boundary_filename(source, country, level)
    write_bytes_atomic(path, data)
    write_source_metadata(path, dataset, url, **details)
    return path
