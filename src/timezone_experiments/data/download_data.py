"""Download global inputs for the default country-level experiment."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from timezone_experiments.data.download.http import download_bytes, download_file, extract_single_member
from timezone_experiments.data.download_population import download_population
from timezone_experiments.data.paths import TIMEZONES_FILENAME, world_filename
from timezone_experiments.files import write_text_atomic

WORLD_BOUNDARIES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"
)
TIMEZONES_URL = (
    "https://github.com/evansiroky/timezone-boundary-builder/releases/latest/download/timezones-now.geojson.zip"
)
# timezones-now.geojson is a few hundred MiB uncompressed; anything far larger is not the expected file.
_MAX_TIMEZONES_BYTES = 4 * 1024**3


def download_world_boundaries(output_directory: Path, *, include_antarctica: bool = False) -> Path:
    """Download Natural Earth countries, excluding Antarctica unless requested."""
    collection = json.loads(download_bytes(WORLD_BOUNDARIES_URL))
    if (
        not isinstance(collection, dict)
        or collection.get("type") != "FeatureCollection"
        or not collection.get("features")
    ):
        msg = "Natural Earth did not return a non-empty GeoJSON FeatureCollection."
        raise ValueError(msg)
    if not include_antarctica:
        collection["features"] = [
            feature
            for feature in collection["features"]
            if feature["properties"].get("ADM0_A3") != "ATA" and feature["properties"].get("ISO_A3") != "ATA"
        ]
    path = output_directory / world_filename(include_antarctica=include_antarctica)
    write_text_atomic(path, json.dumps(collection))
    return path


def download_timezones(output_directory: Path) -> Path:
    """Download and unpack the current timezone-boundary-builder land polygons."""
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / TIMEZONES_FILENAME
    with tempfile.TemporaryDirectory(dir=output_directory) as temporary_directory:
        archive_path = Path(temporary_directory) / "timezones.zip"
        download_file(TIMEZONES_URL, archive_path)
        extract_single_member(
            archive_path,
            frozenset({".json", ".geojson"}),
            path,
            label="Extracting timezones",
            max_bytes=_MAX_TIMEZONES_BYTES,
        )
    return path


def prepare_population(output_directory: Path, cell_degrees: float, *, refresh: bool = False, workers: int = 1) -> Path:
    """Convert population inputs."""
    return download_population(output_directory, cell_degrees=cell_degrees, refresh=refresh, workers=workers)
