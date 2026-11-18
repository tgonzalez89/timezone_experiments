"""Download GHSL population counts and aggregate them into a point-grid CSV."""

from __future__ import annotations

import csv
import json
import logging
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

import numpy as np
import rasterio

from timezone_experiments.data.download_boundaries import download_file
from timezone_experiments.progress import Progress, copy_with_progress

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray
    from rasterio.io import DatasetReader

POPULATION_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
    "GHS_POP_E2025_GLOBE_R2023A_4326_30ss/V1-0/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.zip"
)
POPULATION_CITATION = "https://doi.org/10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE"
logger = logging.getLogger(__name__)


def download_population(
    output_directory: Path, *, cell_degrees: float = 1.0, refresh: bool = False, workers: int = 1
) -> Path:
    """Cache the GHSL 2025 ZIP and produce a count-preserving coarse CSV."""
    _validate_cell_degrees(cell_degrees)
    if workers < 1:
        msg = "Population conversion workers must be positive."
        raise ValueError(msg)
    output_directory.mkdir(parents=True, exist_ok=True)
    archive_path = output_directory / "ghsl_population_2025.zip"
    csv_path = output_directory / "population_2025.csv"
    with tempfile.TemporaryDirectory(dir=output_directory) as temporary_directory:
        temporary_path = Path(temporary_directory)
        if refresh or not archive_path.exists():
            logger.info(
                "Downloading GHSL 2025 population (~461 MiB); conversion needs several GiB of temporary disk space..."
            )
            downloaded_path = temporary_path / "population.zip"
            download_file(POPULATION_URL, downloaded_path)
            downloaded_path.replace(archive_path)
        else:
            logger.info("Reusing cached population archive: %s", archive_path)
        with ZipFile(archive_path) as archive:
            members = [member for member in archive.infolist() if Path(member.filename).suffix.lower() == ".tif"]
            if len(members) != 1:
                msg = "Expected exactly one GeoTIFF in the GHSL population archive."
                raise ValueError(msg)
            raster_path = temporary_path / "population.tif"
            logger.info("Extracting %s (%s bytes).", members[0].filename, f"{members[0].file_size:,}")
            with archive.open(members[0]) as source, raster_path.open("wb") as target:
                copy_with_progress(source, target, label="Extracting population raster", total=members[0].file_size)
        logger.info("Summing population into %g-degree cells.", cell_degrees)
        temporary_csv = temporary_path / "population.csv"
        rows, total = population_raster_to_csv(raster_path, temporary_csv, cell_degrees=cell_degrees, workers=workers)
        temporary_csv.replace(csv_path)
    metadata = {
        "dataset": "GHS-POP R2023A, epoch 2025 (projection), WGS84 30 arcseconds",
        "source_url": POPULATION_URL,
        "dataset_citation": POPULATION_CITATION,
        "methodology_citation": "https://doi.org/10.1080/17538947.2024.2390454",
        "license": "CC BY 4.0; European Union",
        "changes": "Summed native population-count cells; omitted no-data and zero-population cells.",
        "cell_degrees": cell_degrees,
        "populated_cells": rows,
        "population_total": total,
    }
    csv_path.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s populated cells; total population %s.", f"{rows:,}", f"{total:,.2f}")
    return csv_path


def population_raster_to_csv(
    raster_path: Path,
    csv_path: Path,
    *,
    cell_degrees: float = 1.0,
    workers: int = 1,
) -> tuple[int, float]:
    """Sum aligned WGS84 raster blocks, writing their centers and counts."""
    _validate_cell_degrees(cell_degrees)
    if workers < 1:
        msg = "Population conversion workers must be positive."
        raise ValueError(msg)
    rows_written = 0
    totals: list[float] = []
    with rasterio.open(raster_path) as source:
        if (
            source.crs != rasterio.CRS.from_epsg(4326)
            or source.transform.b != 0
            or source.transform.d != 0
            or source.transform.a <= 0
            or source.transform.e >= 0
        ):
            msg = "Population raster must be a north-up WGS84 (EPSG:4326) grid."
            raise ValueError(msg)
        column_factor = _aggregation_factor(cell_degrees, source.transform.a)
        row_factor = _aggregation_factor(cell_degrees, -source.transform.e)
        column_starts = np.arange(0, source.width, column_factor)
        workers = min(workers, max(1, math.ceil(source.height / row_factor)))
        logger.info(
            "Reading raster %s: %s columns x %s rows with %d workers.",
            raster_path,
            f"{source.width:,}",
            f"{source.height:,}",
            workers,
        )
        with (
            csv_path.open("w", encoding="utf-8", newline="") as file,
            Progress("Aggregating population raster rows", source.height, unit="rows") as progress,
        ):
            writer = csv.writer(file)
            writer.writerow(("latitude", "longitude", "population"))
            for row_start, row_count, counts in _population_blocks(source, row_factor, column_factor, workers):
                for column_start, count in zip(column_starts, counts, strict=True):
                    if count > 0:
                        column_count = min(column_factor, source.width - int(column_start))
                        longitude = source.transform.c + (int(column_start) + column_count / 2) * source.transform.a
                        longitude = (longitude + 180) % 360 - 180
                        latitude = source.transform.f + (row_start + row_count / 2) * source.transform.e
                        writer.writerow((latitude, longitude, float(count)))
                        rows_written += 1
                totals.append(float(counts.sum()))
                progress.update(row_start + row_count, f"{rows_written:,} populated CSV cells")
    if not rows_written:
        msg = "Population raster contains no positive population counts."
        raise ValueError(msg)
    return rows_written, math.fsum(totals)


def _population_counts(
    source: DatasetReader, row_start: int, row_count: int, column_factor: int
) -> NDArray[np.float64]:
    data = source.read(
        1, window=((row_start, row_start + row_count), (0, source.width)), masked=True, out_dtype="float64"
    ).filled(0)
    if not np.isfinite(data).all() or np.any(data < 0):
        msg = "Population raster contains non-finite or negative counts outside its no-data mask."
        raise ValueError(msg)
    return np.add.reduceat(data.sum(axis=0), np.arange(0, source.width, column_factor))


def _population_block(
    raster_path: Path, row_factor: int, column_factor: int, row_start: int
) -> tuple[int, int, NDArray[np.float64]]:
    # Each thread owns its GDAL reader; readers must not be shared across concurrent read calls.
    with rasterio.open(raster_path) as source:
        row_count = min(row_factor, source.height - row_start)
        return row_start, row_count, _population_counts(source, row_start, row_count, column_factor)


def _population_blocks(
    source: DatasetReader, row_factor: int, column_factor: int, workers: int
) -> Iterator[tuple[int, int, NDArray[np.float64]]]:
    starts = range(0, source.height, row_factor)
    if workers == 1:
        for row_start in starts:
            row_count = min(row_factor, source.height - row_start)
            yield row_start, row_count, _population_counts(source, row_start, row_count, column_factor)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(
            partial(_population_block, Path(source.name), row_factor, column_factor), starts, buffersize=workers
        )


def _validate_cell_degrees(value: float) -> None:
    if value not in {0.1, 0.25, 0.5, 1.0}:
        msg = "Population cell size must be 0.1, 0.25, 0.5, or 1.0 degrees."
        raise ValueError(msg)


def _aggregation_factor(cell_degrees: float, native_degrees: float) -> int:
    factor = round(cell_degrees / native_degrees)
    # GHSL's nominal 30-arcsecond pixel size has small GeoTIFF rounding error.
    if factor < 1 or not math.isclose(factor * native_degrees, cell_degrees, rel_tol=1e-7):
        msg = "Population cell size must be an integer multiple of the native raster resolution."
        raise ValueError(msg)
    return factor
