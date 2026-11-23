"""Download one administrative boundary dataset for an experiment."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from timezone_experiments.constants import normalize_country_code
from timezone_experiments.data.download.http import download_bytes
from timezone_experiments.files import write_bytes_atomic

if TYPE_CHECKING:
    from pathlib import Path

NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"
)


def download(source: str, country: str, level: int, output_directory: Path) -> Path:
    """Fetch a GeoJSON boundary dataset and return its local path."""
    country = normalize_country_code(country)
    if source not in {"gadm", "geoboundaries", "naturalearth"}:
        msg = f"Unsupported boundary source: {source!r}."
        raise ValueError(msg)
    if level < 0:
        msg = "Administrative level must be non-negative."
        raise ValueError(msg)
    if source == "naturalearth" and level != 0:
        msg = "Natural Earth's configured country dataset only supports administrative level 0."
        raise ValueError(msg)
    if source == "gadm":
        url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{country}_{level}.json"
        data = download_bytes(url)
    elif source == "geoboundaries":
        metadata_url = f"https://www.geoboundaries.org/api/current/gbOpen/{country}/ADM{level}/"
        metadata = json.loads(download_bytes(metadata_url))
        download_url = metadata.get("gjDownloadURL") if isinstance(metadata, dict) else None
        if not isinstance(download_url, str) or not download_url:
            msg = f"GeoBoundaries did not return a GeoJSON download URL for {country} ADM{level}."
            raise ValueError(msg)
        data = download_bytes(download_url)
    else:
        data = download_bytes(NATURAL_EARTH_URL)
    path = output_directory / f"boundaries_{source}_{country}_{level}.geojson"
    write_bytes_atomic(path, data)
    return path
