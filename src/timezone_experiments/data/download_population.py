"""Download GHSL population counts and aggregate them into a point-grid CSV."""

from __future__ import annotations

import csv
import json
import logging
import math
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio

from timezone_experiments.constants import POPULATION_CELL_DEGREES, require_positive_workers, validate_choice
from timezone_experiments.data.download.http import download_file, extract_single_member
from timezone_experiments.files import write_text_atomic
from timezone_experiments.progress import Progress

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray
    from rasterio.io import DatasetReader

POPULATION_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
    "GHS_POP_E2025_GLOBE_R2023A_4326_30ss/V1-0/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.zip"
)
# The global 30-arcsecond GeoTIFF is a few GiB uncompressed; anything far larger is not the expected file.
_MAX_RASTER_BYTES = 32 * 1024**3
POPULATION_CITATION = "https://doi.org/10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE"
logger = logging.getLogger(__name__)


def download_population(
    output_directory: Path, *, cell_degrees: float = 1.0, refresh: bool = False, workers: int = 1
) -> Path:
    """Cache the GHSL 2025 ZIP and produce a count-preserving coarse CSV."""
    validate_choice("Population cell size (degrees)", cell_degrees, POPULATION_CELL_DEGREES)
    require_positive_workers(workers, "Population conversion")
    output_directory.mkdir(parents=True, exist_ok=True)
    archive_path = output_directory / "ghsl_population_2025.zip"
    csv_path = output_directory / "population_2025.csv"
    metadata_path = csv_path.with_suffix(".source.json")
    with tempfile.TemporaryDirectory(dir=output_directory) as temporary_directory:
        temporary_path = Path(temporary_directory)
        if refresh or not archive_path.exists():
            logger.info(
                "Downloading GHSL 2025 population (~461 MiB); conversion needs several GiB of temporary disk space..."
            )
            download_file(POPULATION_URL, archive_path)
        else:
            logger.info("Reusing cached population archive: %s", archive_path)
        raster_path = temporary_path / "population.tif"
        extract_single_member(
            archive_path,
            frozenset({".tif"}),
            raster_path,
            label="Extracting population raster",
            max_bytes=_MAX_RASTER_BYTES,
        )
        logger.info("Summing population into %g-degree cells.", cell_degrees)
        temporary_csv = temporary_path / "population.csv"
        rows, total = population_raster_to_csv(raster_path, temporary_csv, cell_degrees=cell_degrees, workers=workers)
        # Drop the old metadata first: a crash before it is rewritten must not label the new CSV.
        metadata_path.unlink(missing_ok=True)
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
    write_text_atomic(metadata_path, json.dumps(metadata, indent=2) + "\n")
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
    validate_choice("Population cell size (degrees)", cell_degrees, POPULATION_CELL_DEGREES)
    require_positive_workers(workers, "Population conversion")
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


def _population_blocks(
    source: DatasetReader, row_factor: int, column_factor: int, workers: int
) -> Iterator[tuple[int, int, NDArray[np.float64]]]:
    starts = range(0, source.height, row_factor)

    def block(reader: DatasetReader, row_start: int) -> tuple[int, int, NDArray[np.float64]]:
        row_count = min(row_factor, source.height - row_start)
        return row_start, row_count, _population_counts(reader, row_start, row_count, column_factor)

    if workers == 1:
        for row_start in starts:
            yield block(source, row_start)
        return
    # GDAL readers must not be shared across concurrent reads, so each thread keeps one open reader. Keeping it
    # open (rather than reopening per row band) preserves GDAL's block cache for tiles that straddle bands.
    local = threading.local()
    readers: list[DatasetReader] = []

    def threaded_block(row_start: int) -> tuple[int, int, NDArray[np.float64]]:
        reader = getattr(local, "reader", None)
        if reader is None:
            reader = local.reader = rasterio.open(source.name)
            readers.append(reader)
        return block(reader, row_start)

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            yield from executor.map(threaded_block, starts, buffersize=workers)
    finally:
        for reader in readers:
            reader.close()


def _aggregation_factor(cell_degrees: float, native_degrees: float) -> int:
    factor = round(cell_degrees / native_degrees)
    # GHSL's nominal 30-arcsecond pixel size has small GeoTIFF rounding error.
    if factor < 1 or not math.isclose(factor * native_degrees, cell_degrees, rel_tol=1e-7):
        msg = "Population cell size must be an integer multiple of the native raster resolution."
        raise ValueError(msg)
    return factor
