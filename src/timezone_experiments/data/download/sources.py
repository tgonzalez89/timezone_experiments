"""Download global inputs for the default country-level experiment."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from timezone_experiments.core.files import write_text_atomic
from timezone_experiments.data.download.catalog import SOURCES, write_source_metadata
from timezone_experiments.data.download.http import download_bytes, download_file, extract_single_member
from timezone_experiments.data.download.population import download_population
from timezone_experiments.data.download.versions import warn_if_outdated
from timezone_experiments.data.paths import timezones_filename, world_filename

# timezones-now.geojson is a few hundred MiB uncompressed; anything far larger is not the expected file.
_MAX_TIMEZONES_BYTES = 4 * 1024**3


def download_world_boundaries(output_directory: Path, *, include_antarctica: bool = False) -> Path:
    """Download the pinned Natural Earth countries, excluding Antarctica unless requested."""
    source = SOURCES["natural_earth"]
    warn_if_outdated(source.key, output_directory)
    url = source.resolve()
    collection = json.loads(download_bytes(url))
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
    write_source_metadata(path, source, url, antarctica_included=include_antarctica)
    return path


def download_timezones(output_directory: Path) -> Path:
    """Download and unpack the pinned timezone-boundary-builder land polygons (timezones-now)."""
    output_directory.mkdir(parents=True, exist_ok=True)
    source = SOURCES["timezone_boundary_builder"]
    warn_if_outdated(source.key, output_directory)
    url = source.resolve()
    path = output_directory / timezones_filename()
    with tempfile.TemporaryDirectory(dir=output_directory) as temporary_directory:
        archive_path = Path(temporary_directory) / "timezones.zip"
        download_file(url, archive_path)
        extract_single_member(
            archive_path,
            frozenset({".json", ".geojson"}),
            path,
            label="Extracting timezones",
            max_bytes=_MAX_TIMEZONES_BYTES,
        )
    write_source_metadata(path, source, url)
    return path


def prepare_population(output_directory: Path, cell_degrees: float, *, refresh: bool = False, workers: int = 1) -> Path:
    """Convert population inputs."""
    return download_population(output_directory, cell_degrees=cell_degrees, refresh=refresh, workers=workers)
