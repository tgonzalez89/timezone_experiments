"""Point lookup against timezone-boundary-builder GeoJSON."""

from __future__ import annotations

import logging
import math
import time
from array import array
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import batched, chain, islice
from typing import TYPE_CHECKING

from timezone_experiments.data.geometry import PreparedPolygon, _unwrap_longitude, iter_polygons, unwrap_polygon
from timezone_experiments.progress import Progress

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from timezone_experiments.data.models import Territory

logger = logging.getLogger(__name__)
CELL_DEGREES = 5
LONGITUDE_CELLS = 360 // CELL_DEGREES
LOOKUP_BATCH_SIZE = 25_000
_worker_index: TimezoneBoundaryIndex | None = None


class TimezoneBoundaryIndex:
    """Spatially indexed, prepared timezone polygons with stable source precedence."""

    def __init__(self, boundaries: Sequence[tuple[str, dict[str, object]]]) -> None:
        """Parse polygons once and index their bounding boxes in geographic cells."""
        if not boundaries:
            msg = "At least one timezone boundary is required."
            raise ValueError(msg)
        self._cells: dict[tuple[int, int], list[tuple[str, PreparedPolygon]]] = {}
        logger.info("Preparing spatial index for %d timezone features.", len(boundaries))
        for timezone, geometry in boundaries:
            for polygon in iter_polygons(geometry):
                prepared = PreparedPolygon(unwrap_polygon(polygon))
                min_lon, min_lat, max_lon, max_lat = prepared.bounds
                longitude_cells = {
                    cell % LONGITUDE_CELLS
                    for cell in range(math.floor(min_lon / CELL_DEGREES), math.floor(max_lon / CELL_DEGREES) + 1)
                }
                for longitude_cell in longitude_cells:
                    for latitude_cell in range(
                        math.floor(min_lat / CELL_DEGREES), math.floor(max_lat / CELL_DEGREES) + 1
                    ):
                        self._cells.setdefault((longitude_cell, latitude_cell), []).append((timezone, prepared))
        logger.info("Timezone spatial index ready.")

    def timezone_at(self, latitude: float, longitude: float) -> str | None:
        """Return the timezone polygon containing a coordinate, if any."""
        cell = (math.floor(longitude / CELL_DEGREES) % LONGITUDE_CELLS, math.floor(latitude / CELL_DEGREES))
        for timezone, polygon in self._cells.get(cell, ()):
            adjusted = _unwrap_longitude(longitude, polygon.reference_longitude)
            if polygon.contains((adjusted, latitude)):
                return timezone
        return None


def annotate_current_timezones(
    territories: Sequence[Territory],
    index: TimezoneBoundaryIndex,
    *,
    workers: int = 1,
) -> list[Territory]:
    """Assign the area-majority current timezone to each territory."""
    if workers < 1:
        msg = "Timezone overlay workers must be positive."
        raise ValueError(msg)
    started = time.perf_counter()
    counts = [sum(sample.area_weight > 0 for sample in territory.samples) for territory in territories]
    total = sum(counts)
    workers = min(workers, max(1, math.ceil(total / LOOKUP_BATCH_SIZE)))
    logger.info(
        "Looking up current timezones for %d territories (%s positive-area samples) with %d workers.",
        len(territories),
        f"{total:,}",
        workers,
    )
    if workers > 1:
        zones = _parallel_timezones(territories, index, workers)
    else:
        zones = (
            (
                position,
                [
                    index.timezone_at(sample.latitude, sample.longitude)
                    for sample in territory.samples
                    if sample.area_weight > 0
                ],
            )
            for position, territory in enumerate(territories)
        )
    weights: list[dict[str, float]] = [{} for _ in territories]
    samples = [iter(sample for sample in territory.samples if sample.area_weight > 0) for territory in territories]
    with Progress("Timezone overlay", len(territories)) as progress:
        completed = 0
        for position, timezones in zones:
            for sample, timezone in zip(islice(samples[position], len(timezones)), timezones, strict=True):
                if timezone is not None:
                    weights[position][timezone] = weights[position].get(timezone, 0) + sample.area_weight
            counts[position] -= len(timezones)
            if counts[position] == 0:
                territory = territories[position]
                completed += 1
                progress.update(completed, f"finished {territory.territory_id} ({territory.name})")
    annotated = [
        replace(territory, current_timezone=max(area, key=area.__getitem__) if area else territory.current_timezone)
        for territory, area in zip(territories, weights, strict=True)
    ]
    logger.info(
        "Current timezone overlay complete for %d territories in %.2f seconds.",
        len(annotated),
        time.perf_counter() - started,
    )
    return annotated


def _initialize_timezone_worker(index: TimezoneBoundaryIndex) -> None:
    global _worker_index  # noqa: PLW0603
    _worker_index = index


def _lookup_chunk(task: tuple[int, array[float]]) -> tuple[int, list[str | None]]:
    if _worker_index is None:
        msg = "Timezone worker has not been initialized."
        raise RuntimeError(msg)
    position, coordinates = task
    return position, [_worker_index.timezone_at(lat, lon) for lat, lon in batched(coordinates, 2, strict=True)]


def _coordinate_chunks(territories: Sequence[Territory]) -> Iterator[tuple[int, array[float]]]:
    for position, territory in enumerate(territories):
        points = ((sample.latitude, sample.longitude) for sample in territory.samples if sample.area_weight > 0)
        has_points = False
        for batch in batched(points, LOOKUP_BATCH_SIZE, strict=False):
            has_points = True
            yield position, array("d", chain.from_iterable(batch))
        if not has_points:
            yield position, array("d")


def _parallel_timezones(
    territories: Sequence[Territory], index: TimezoneBoundaryIndex, workers: int
) -> Iterator[tuple[int, list[str | None]]]:
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_initialize_timezone_worker, initargs=(index,)
    ) as executor:
        yield from executor.map(_lookup_chunk, _coordinate_chunks(territories), buffersize=2 * workers)
