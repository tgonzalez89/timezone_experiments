"""Download global inputs for the default country-level experiment."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from zipfile import ZipFile

from timezone_experiments.data.download_boundaries import _download_bytes, download_file
from timezone_experiments.progress import copy_with_progress

WORLD_BOUNDARIES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"
)
TIMEZONES_URL = (
    "https://github.com/evansiroky/timezone-boundary-builder/releases/latest/download/timezones-now.geojson.zip"
)
logger = logging.getLogger(__name__)


def download_world_boundaries(output_directory: Path, *, include_antarctica: bool = False) -> Path:
    """Download Natural Earth countries, excluding Antarctica unless requested."""
    collection = json.loads(_download_bytes(WORLD_BOUNDARIES_URL))
    if collection.get("type") != "FeatureCollection" or not collection.get("features"):
        msg = "Natural Earth did not return a non-empty GeoJSON FeatureCollection."
        raise ValueError(msg)
    if not include_antarctica:
        collection["features"] = [
            feature
            for feature in collection["features"]
            if feature["properties"].get("ADM0_A3") != "ATA" and feature["properties"].get("ISO_A3") != "ATA"
        ]
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = "world_countries_with_antarctica.geojson" if include_antarctica else "world_countries.geojson"
    path = output_directory / filename
    path.write_text(json.dumps(collection), encoding="utf-8")
    return path


def download_timezones(output_directory: Path) -> Path:
    """Download and unpack the current timezone-boundary-builder land polygons."""
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "world_timezones.geojson"
    with tempfile.TemporaryDirectory(dir=output_directory) as temporary_directory:
        archive_path = Path(temporary_directory) / "timezones.zip"
        download_file(TIMEZONES_URL, archive_path)
        with ZipFile(archive_path) as archive:
            members = [member for member in archive.infolist() if Path(member.filename).suffix in {".json", ".geojson"}]
            if len(members) != 1:
                msg = "Expected exactly one GeoJSON file in the timezone-boundary-builder archive."
                raise ValueError(msg)
            extracted_path = Path(temporary_directory) / "timezones.geojson"
            logger.info("Extracting timezone polygons: %s (%s bytes)", members[0].filename, f"{members[0].file_size:,}")
            with archive.open(members[0]) as source, extracted_path.open("wb") as target:
                copy_with_progress(source, target, label="Extracting timezones", total=members[0].file_size)
        extracted_path.replace(path)
    return path


def prepare_population(output_directory: Path, cell_degrees: float, *, refresh: bool = False, workers: int = 1) -> Path:
    """Convert population inputs with an actionable optional-dependency error."""
    try:
        from timezone_experiments.data.download_population import download_population  # noqa: PLC0415
    except ImportError as error:
        msg = "Population conversion requires numpy and rasterio; install with `uv sync --extra population`."
        raise RuntimeError(msg) from error
    return download_population(output_directory, cell_degrees=cell_degrees, refresh=refresh, workers=workers)
