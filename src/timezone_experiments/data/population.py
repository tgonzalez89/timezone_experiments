"""Population point-grid loading and territory weighting."""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from timezone_experiments.constants import MAX_LATITUDE, MAX_LONGITUDE, MIN_LATITUDE, MIN_LONGITUDE
from timezone_experiments.data.geometry import PolygonIndex
from timezone_experiments.data.models import Territory, TerritorySample

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


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
            try:
                point = _population_point(row[fields["latitude"]], row[fields["longitude"]], row[fields["population"]])
            except (TypeError, ValueError) as error:
                msg = f"Invalid population row at {path}:{reader.line_num}: {error}"
                raise ValueError(msg) from error
            if point.population:
                points.append(point)
    logger.info("Loaded %s populated grid cells from %s.", f"{len(points):,}", path)
    return points


def _population_point(latitude: str | None, longitude: str | None, population: str | None) -> PopulationPoint:
    if latitude is None or longitude is None or population is None:
        msg = "row has missing columns."
        raise ValueError(msg)
    point = PopulationPoint(latitude=float(latitude), longitude=float(longitude), population=float(population))
    if not MIN_LATITUDE <= point.latitude <= MAX_LATITUDE:
        msg = f"latitude must be between -90 and 90: {latitude!r}."
        raise ValueError(msg)
    if not MIN_LONGITUDE <= point.longitude <= MAX_LONGITUDE:
        msg = f"longitude must be between -180 and 180: {longitude!r}."
        raise ValueError(msg)
    if not math.isfinite(point.population) or point.population < 0:
        msg = f"population must be finite and non-negative: {population!r}."
        raise ValueError(msg)
    return point


def attach_population_weights(
    territories: Iterable[Territory],
    population_points: Iterable[PopulationPoint],
    *,
    workers: int = 1,
) -> list[Territory]:
    """Attach the gridded population cells lying inside each territory's geometry.

    Base samples keep their area/longitude weights with ``population_weight=0``; each matched cell is appended
    as a population-only sample (zero area and longitude weight), in population-file order. A cell inside
    several (overlapping) territories counts for each of them.
    """
    if workers < 1:
        msg = "Population overlay workers must be positive."
        raise ValueError(msg)
    points = tuple(population_points)
    units = tuple(territories)
    for territory in units:
        if territory.geometry is None:
            msg = f"Population overlay requires geometry for {territory.territory_id!r}."
            raise ValueError(msg)
    logger.info(
        "Overlaying %s population cells on %d territories with %d workers.", f"{len(points):,}", len(units), workers
    )
    index = PolygonIndex([territory.geometry for territory in units if territory.geometry is not None])
    point_indices, owners = index.matches(
        np.fromiter((point.longitude for point in points), np.float64, len(points)),
        np.fromiter((point.latitude for point in points), np.float64, len(points)),
        workers=workers,
        progress_label="Population overlay",
    )
    # Pairs are sorted by territory, so each territory's matches are one contiguous, point-ordered slice.
    bounds = np.searchsorted(owners, np.arange(len(units) + 1))
    result = [
        _with_population(territory, [points[index] for index in point_indices[start:stop].tolist()])
        for territory, start, stop in zip(units, bounds[:-1].tolist(), bounds[1:].tolist(), strict=True)
    ]
    logger.info("Population overlay complete for %d territories.", len(result))
    return result


def _with_population(territory: Territory, matched: list[PopulationPoint]) -> Territory:
    base = (
        TerritorySample(
            latitude=sample.latitude,
            longitude=sample.longitude,
            area_weight=sample.area_weight,
            population_weight=0,
            longitude_weight=sample.longitude_weight,
        )
        for sample in territory.samples
    )
    population = (
        TerritorySample(
            latitude=point.latitude,
            longitude=point.longitude,
            area_weight=0,
            population_weight=point.population,
            longitude_weight=0,
        )
        for point in matched
    )
    return replace(territory, samples=(*base, *population))
