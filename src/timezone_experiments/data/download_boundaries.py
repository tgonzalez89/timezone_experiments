"""Download one administrative boundary dataset for an experiment."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from timezone_experiments.constants import normalize_country_code
from timezone_experiments.data.download.http import download_bytes
from timezone_experiments.data.paths import boundary_filename
from timezone_experiments.files import write_bytes_atomic

if TYPE_CHECKING:
    from pathlib import Path

BOUNDARY_SOURCES = ("gadm", "geoboundaries")


def download(source: str, country: str, level: int, output_directory: Path) -> Path:
    """Fetch a GeoJSON boundary dataset and return its local path."""
    country = normalize_country_code(country)
    if source not in BOUNDARY_SOURCES:
        msg = f"Unsupported boundary source: {source!r}."
        raise ValueError(msg)
    if level < 0:
        msg = "Administrative level must be non-negative."
        raise ValueError(msg)
    if source == "gadm":
        url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{country}_{level}.json"
        data = download_bytes(url)
    else:
        metadata_url = f"https://www.geoboundaries.org/api/current/gbOpen/{country}/ADM{level}/"
        metadata = json.loads(download_bytes(metadata_url))
        download_url = metadata.get("gjDownloadURL") if isinstance(metadata, dict) else None
        if not isinstance(download_url, str) or not download_url:
            msg = f"GeoBoundaries did not return a GeoJSON download URL for {country} ADM{level}."
            raise ValueError(msg)
        data = download_bytes(download_url)
    path = output_directory / boundary_filename(source, country, level)
    write_bytes_atomic(path, data)
    return path
