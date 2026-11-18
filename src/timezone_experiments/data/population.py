"""Population point-grid loading and territory weighting."""

from __future__ import annotations

import csv
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from timezone_experiments.data.geometry import PreparedPolygon, _unwrap_longitude, _unwrap_ring, iter_polygons
from timezone_experiments.data.models import Territory, TerritorySample
from timezone_experiments.progress import Progress

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)
_POPULATION_CELL_DEGREES = 5
_POPULATION_LONGITUDE_CELLS = 360 // _POPULATION_CELL_DEGREES
type PopulationIndex = dict[tuple[int, int], list[tuple[int, PopulationPoint]]]
_worker_population_index: PopulationIndex | None = None


@dataclass(frozen=True, slots=True)
class PopulationPoint:
    """Population count located at a grid-cell coordinate."""

    latitude: float
    longitude: float
    population: float


def load_population_csv(path: str | Path) -> list[PopulationPoint]:
    """Load a point grid CSV with latitude, longitude, and population columns."""
    points: list[PopulationPoint] = []
    logger.info("Reading population CSV: %s", path)
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"latitude", "longitude", "population"}
        if reader.fieldnames is None or not required.issubset({name.lower() for name in reader.fieldnames}):
            msg = "Population CSV must contain latitude, longitude, and population columns."
            raise ValueError(msg)
        fields = {name.lower(): name for name in reader.fieldnames}
        for row in reader:
            population = float(row[fields["population"]])
            if population < 0:
                msg = "Population counts cannot be negative."
                raise ValueError(msg)
            if population:
                points.append(
                    PopulationPoint(
                        latitude=float(row[fields["latitude"]]),
                        longitude=float(row[fields["longitude"]]),
                        population=population,
                    ),
                )
    logger.info("Loaded %s populated grid cells from %s.", f"{len(points):,}", path)
    return points


def attach_population_weights(
    territories: Iterable[Territory],
    population_points: Iterable[PopulationPoint],
    *,
    workers: int = 1,
) -> list[Territory]:
    """Attach gridded population cells inside each GeoJSON territory."""
    if workers < 1:
        msg = "Population overlay workers must be positive."
        raise ValueError(msg)
    points = tuple(population_points)
    units = tuple(territories)
    workers = min(workers, max(1, len(units)))
    logger.info(
        "Overlaying %s population cells on %d territories with %d workers.", f"{len(points):,}", len(units), workers
    )
    results: dict[int, Territory] = {}
    with Progress("Population overlay", len(units)) as progress:
        for completed, (position, territory) in enumerate(_population_overlays(units, points, workers), start=1):
            results[position] = territory
            progress.update(completed, f"finished {territory.territory_id} ({territory.name})")
    result = [results[position] for position in range(len(units))]
    logger.info("Population overlay complete for %d territories.", len(result))
    return result


def _population_overlays(
    territories: tuple[Territory, ...], points: tuple[PopulationPoint, ...], workers: int
) -> Iterator[tuple[int, Territory]]:
    if workers == 1:
        index = _index_population_points(points)
        for position, territory in enumerate(territories):
            yield position, _population_overlay(territory, index)
        return
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_initialize_population_worker, initargs=(points,)
    ) as executor:
        futures = {
            executor.submit(_population_overlay_worker, territory): position
            for position, territory in enumerate(territories)
        }
        for future in as_completed(futures):
            yield futures.pop(future), future.result()


def _initialize_population_worker(points: tuple[PopulationPoint, ...]) -> None:
    global _worker_population_index  # noqa: PLW0603
    _worker_population_index = _index_population_points(points)


def _population_overlay_worker(territory: Territory) -> Territory:
    if _worker_population_index is None:
        msg = "Population worker has not been initialized."
        raise RuntimeError(msg)
    return _population_overlay(territory, _worker_population_index)


def _index_population_points(points: tuple[PopulationPoint, ...]) -> PopulationIndex:
    index: PopulationIndex = {}
    for position, point in enumerate(points):
        cell = (
            math.floor(point.longitude / _POPULATION_CELL_DEGREES) % _POPULATION_LONGITUDE_CELLS,
            math.floor(point.latitude / _POPULATION_CELL_DEGREES),
        )
        index.setdefault(cell, []).append((position, point))
    return index


def _population_overlay(territory: Territory, index: PopulationIndex) -> Territory:
    if territory.geometry is None:
        msg = f"Population overlay requires geometry for {territory.territory_id!r}."
        raise ValueError(msg)
    samples = [
        TerritorySample(
            latitude=sample.latitude,
            longitude=sample.longitude,
            area_weight=sample.area_weight,
            population_weight=0,
            longitude_weight=sample.longitude_weight,
        )
        for sample in territory.samples
    ]
    matched: dict[int, PopulationPoint] = {}
    for polygon in iter_polygons(territory.geometry):
        prepared = PreparedPolygon(tuple(_unwrap_ring(ring) for ring in polygon))
        min_lon, min_lat, max_lon, max_lat = prepared.bounds
        longitude_cells = {
            cell % _POPULATION_LONGITUDE_CELLS
            for cell in range(
                math.floor(min_lon / _POPULATION_CELL_DEGREES), math.floor(max_lon / _POPULATION_CELL_DEGREES) + 1
            )
        }
        for longitude_cell in longitude_cells:
            for latitude_cell in range(
                math.floor(min_lat / _POPULATION_CELL_DEGREES), math.floor(max_lat / _POPULATION_CELL_DEGREES) + 1
            ):
                for position, point in index.get((longitude_cell, latitude_cell), ()):
                    if position not in matched and prepared.contains(
                        (_unwrap_longitude(point.longitude, prepared.reference_longitude), point.latitude)
                    ):
                        matched[position] = point
    samples.extend(
        TerritorySample(
            latitude=point.latitude,
            longitude=point.longitude,
            area_weight=0,
            population_weight=point.population,
            longitude_weight=0,
        )
        for _, point in sorted(matched.items())
    )
    return Territory(
        territory_id=territory.territory_id,
        country_code=territory.country_code,
        name=territory.name,
        admin_level=territory.admin_level,
        samples=tuple(samples),
        current_timezone=territory.current_timezone,
        geometry=territory.geometry,
    )
