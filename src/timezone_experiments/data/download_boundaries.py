"""Download one administrative boundary dataset for an experiment."""

from __future__ import annotations

import io
import json
import logging
import urllib.request
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from timezone_experiments.progress import copy_with_progress

if TYPE_CHECKING:
    from pathlib import Path

NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"
)
ALLOWED_HOSTS = {
    "geodata.ucdavis.edu",
    "www.geoboundaries.org",
    "raw.githubusercontent.com",
    "github.com",
    "jeodpp.jrc.ec.europa.eu",
}
HTTP_CLIENT_ERROR = 400
logger = logging.getLogger(__name__)


def download(source: str, country: str, level: int, output_directory: Path) -> Path:
    """Fetch a GeoJSON boundary dataset and return its local path."""
    country = country.upper()
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
        data = _download_bytes(url)
    elif source == "geoboundaries":
        metadata_url = f"https://www.geoboundaries.org/api/current/gbOpen/{country}/ADM{level}/"
        metadata = json.loads(_download_bytes(metadata_url))
        download_url = metadata.get("gjDownloadURL")
        if not isinstance(download_url, str) or not download_url:
            msg = f"GeoBoundaries did not return a GeoJSON download URL for {country} ADM{level}."
            raise ValueError(msg)
        data = _download_bytes(download_url)
    else:
        data = _download_bytes(NATURAL_EARTH_URL)
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / f"boundaries_{source}_{country}_{level}.geojson"
    path.write_bytes(data)
    return path


def _dataset_request(url: str) -> urllib.request.Request:
    parsed_url = urlsplit(url)
    if parsed_url.scheme != "https" or parsed_url.hostname not in ALLOWED_HOSTS:
        msg = f"Dataset URL must use HTTPS and a supported host: {url!r}."
        raise ValueError(msg)
    return urllib.request.Request(  # noqa: S310
        url,
        headers={"User-Agent": "timezone-experiments/0.1"},
    )


def download_file(url: str, path: Path) -> None:
    """Stream a supported dataset to a file without buffering it in memory."""
    request = _dataset_request(url)
    logger.info("Downloading %s -> %s", url, path)
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as target:  # noqa: S310
        length = response.headers.get("Content-Length")
        total = int(length) if length is not None else None
        copy_with_progress(response, target, label=f"Downloading {path.name}", total=total)


def _download_bytes(url: str) -> bytes:
    request = _dataset_request(url)
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        if response.status >= HTTP_CLIENT_ERROR:
            msg = f"Dataset server returned HTTP {response.status} for {url}."
            raise OSError(msg)
        length = response.headers.get("Content-Length")
        total = int(length) if length is not None else None
        with io.BytesIO() as target:
            copy_with_progress(response, target, label="Downloading dataset", total=total)
            return target.getvalue()
