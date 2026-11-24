"""Lightweight GeoJSON polygon operations used by boundary loaders."""

from __future__ import annotations

# Spherical and coordinate-domain constants are intrinsic to geometry calculations.
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import shapely

from timezone_experiments.core.constants import (
    FULL_LONGITUDE_TURN,
    GEOMETRY_AREA_EPSILON,
    HALF_LONGITUDE_TURN,
    MAX_LATITUDE,
    MIN_LATITUDE,
    MIN_POLYGON_RING_POINTS,
    POSITION_DIMENSIONS,
)
from timezone_experiments.core.models import TerritorySample
from timezone_experiments.core.progress import Progress

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Any

    from numpy.typing import NDArray

type Position = tuple[float, float]
type Ring = tuple[Position, ...]

logger = logging.getLogger(__name__)


class GeoJSONError(ValueError):
    """Malformed GeoJSON structure (wrong types or shapes), reported to users like any invalid input."""


class PolygonIndex:
    """Prepared polygons from many GeoJSON geometries, answering "which geometries contain these points?".

    Polygons are unwrapped across the antimeridian once and prepared for fast point-in-polygon tests. Points
    are matched against each polygon's bounding box through a longitude-sorted array, then tested with
    ``shapely.contains_xy`` (interior only, so points exactly on an edge do not match). shapely releases the
    GIL during those tests, so ``workers`` threads run them in parallel without copying data to processes.
    """

    def __init__(self, geometries: Sequence[dict[str, Any]]) -> None:
        """Prepare every non-degenerate polygon; its owner is the index of the geometry it came from."""
        self._polygons: list[shapely.Polygon] = []
        owners: list[int] = []
        for owner, geometry in enumerate(geometries):
            for polygon in iter_polygons(geometry):
                rings = unwrap_polygon(polygon)
                if rings and _ring_has_area(rings[0]):
                    self._polygons.append(_planar_polygon(rings))
                    owners.append(owner)
        self._owners = np.array(owners, dtype=np.intp)

    def __len__(self) -> int:
        """Return the number of prepared polygons."""
        return len(self._polygons)

    def matches(
        self,
        longitudes: NDArray[np.float64],
        latitudes: NDArray[np.float64],
        *,
        workers: int = 1,
        progress_label: str | None = None,
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Return unique ``(point, owner)`` index pairs, sorted by owner then point, for every containment."""
        # Unwrapped polygons may extend past +/-180 degrees, so compare wrapped points shifted by whole turns.
        wrapped = (np.asarray(longitudes, dtype=np.float64) + HALF_LONGITUDE_TURN) % FULL_LONGITUDE_TURN
        wrapped -= HALF_LONGITUDE_TURN
        order = np.argsort(wrapped, kind="stable")
        sorted_longitudes = wrapped[order]
        sorted_latitudes = np.asarray(latitudes, dtype=np.float64)[order]

        def contained(polygon: shapely.Polygon) -> NDArray[np.intp]:
            min_lon, min_lat, max_lon, max_lat = polygon.bounds
            found = []
            for shift in (-FULL_LONGITUDE_TURN, 0.0, FULL_LONGITUDE_TURN):
                start = np.searchsorted(sorted_longitudes, min_lon - shift, side="left")
                stop = np.searchsorted(sorted_longitudes, max_lon - shift, side="right")
                if start == stop:
                    continue
                band = sorted_latitudes[start:stop]
                candidates = np.flatnonzero((band >= min_lat) & (band <= max_lat)) + start
                if candidates.size:
                    inside = shapely.contains_xy(
                        polygon, sorted_longitudes[candidates] + shift, sorted_latitudes[candidates]
                    )
                    found.append(order[candidates[inside]])
            return np.concatenate(found) if found else np.empty(0, dtype=np.intp)

        with (
            ThreadPoolExecutor(max_workers=max(1, workers)) as executor,
            Progress(
                progress_label or "Point-in-polygon", len(self._polygons), enabled=progress_label is not None
            ) as progress,
        ):
            per_polygon = []
            for completed, points in enumerate(executor.map(contained, self._polygons), start=1):
                per_polygon.append(points)
                progress.update(completed)
        if not per_polygon:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
        points = np.concatenate(per_polygon)
        owners = np.repeat(self._owners, [len(item) for item in per_polygon])
        # Several polygons of one geometry (or its copies across the antimeridian) may contain the same point.
        pairs = np.unique(np.stack((owners, points)), axis=1)
        return pairs[1], pairs[0]

    def first_owner(
        self,
        longitudes: NDArray[np.float64],
        latitudes: NDArray[np.float64],
        *,
        workers: int = 1,
        progress_label: str | None = None,
    ) -> NDArray[np.intp]:
        """Return, for each point, the lowest owner index containing it (-1 when none does)."""
        points, owners = self.matches(longitudes, latitudes, workers=workers, progress_label=progress_label)
        first = np.full(len(longitudes), np.iinfo(np.intp).max, dtype=np.intp)
        np.minimum.at(first, points, owners)
        first[first == np.iinfo(np.intp).max] = -1
        return first


def iter_polygons(geometry: dict[str, Any]) -> Iterator[tuple[Ring, ...]]:
    """Yield polygons from GeoJSON Polygon or MultiPolygon geometry."""
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        yield _polygon_rings(coordinates)
    elif geometry_type == "MultiPolygon":
        if not isinstance(coordinates, list):
            msg = "MultiPolygon coordinates must be an array."
            raise GeoJSONError(msg)
        for polygon in coordinates:
            yield _polygon_rings(polygon)
    else:
        msg = f"Expected Polygon or MultiPolygon geometry, got {geometry_type!r}."
        raise GeoJSONError(msg)


def sample_geometry(geometry: dict[str, Any], grid_size: int) -> tuple[TerritorySample, ...]:
    """Create approximate area-weighted samples for polygonal GeoJSON."""
    if grid_size < 1:
        msg = "grid_size must be positive."
        raise ValueError(msg)
    samples: list[TerritorySample] = []
    for polygon in iter_polygons(geometry):
        rings = unwrap_polygon(polygon)
        if not rings or not _ring_has_area(rings[0]):
            logger.warning("Skipping degenerate polygon component with zero area.")
            continue
        outer = rings[0]
        min_lon = min(point[0] for point in outer)
        max_lon = max(point[0] for point in outer)
        min_lat = min(point[1] for point in outer)
        max_lat = max(point[1] for point in outer)
        grid_cell_weight = (max_lon - min_lon) * (max_lat - min_lat) / (grid_size * grid_size)
        longitude_cell_weight = (max_lon - min_lon) / grid_size
        # Cell centres of a grid_size x grid_size grid over the polygon's (unwrapped) bounding box.
        latitudes = min_lat + (np.arange(grid_size) + 0.5) * (max_lat - min_lat) / grid_size
        longitudes = min_lon + (np.arange(grid_size) + 0.5) * (max_lon - min_lon) / grid_size
        longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
        inside = shapely.contains_xy(_planar_polygon(rings), longitude_grid, latitude_grid)
        if inside.any():
            # Each grid column's width is shared evenly by its inside cells, so longitude coverage measures the
            # share of the territory's longitude span; area weights scale the equal-angle cell by cos(latitude).
            column_counts = inside.sum(axis=0)
            rows, columns = np.nonzero(inside)
            samples.extend(
                TerritorySample(
                    latitude=float(latitudes[row]),
                    longitude=_wrap_longitude(float(longitudes[column])),
                    area_weight=grid_cell_weight * max(math.cos(math.radians(latitudes[row])), 1e-8),
                    longitude_weight=longitude_cell_weight / int(column_counts[column]),
                )
                for row, column in zip(rows.tolist(), columns.tolist(), strict=True)
            )
        else:
            candidate = _polygon_centroid(rings)
            if not _contains(rings, candidate):
                candidate = _interior_fallback(rings)
            samples.append(
                TerritorySample(
                    latitude=candidate[1],
                    longitude=_wrap_longitude(candidate[0]),
                    area_weight=max(grid_cell_weight, 1e-8) * max(math.cos(math.radians(candidate[1])), 1e-8),
                    longitude_weight=max(longitude_cell_weight, 1e-8),
                ),
            )
    if not samples:
        msg = "GeoJSON geometry contains no non-zero-area polygonal samples."
        raise ValueError(msg)
    return tuple(samples)


def _planar_polygon(rings: tuple[Ring, ...]) -> shapely.Polygon:
    """Build a prepared shapely polygon from unwrapped rings, dropping holes too small to enclose area."""
    polygon = shapely.Polygon(rings[0], [hole for hole in rings[1:] if len(hole) >= MIN_POLYGON_RING_POINTS])
    shapely.prepare(polygon)
    return polygon


def _ring_has_area(ring: Ring) -> bool:
    return ring_winding(ring) != 0


def ring_winding(ring: Ring) -> float:
    """Return twice the signed planar area of a ring (positive when counter-clockwise, 0 when degenerate)."""
    if len(ring) < MIN_POLYGON_RING_POINTS:
        return 0.0
    # Translation avoids cancelling tiny areas at large longitude/latitude coordinates.
    origin_lon, origin_lat = ring[0]
    return math.fsum(
        (first[0] - origin_lon) * (second[1] - origin_lat) - (second[0] - origin_lon) * (first[1] - origin_lat)
        for first, second in zip(ring, (*ring[1:], ring[0]), strict=True)
    )


def point_in_geometry(longitude: float, latitude: float, geometry: dict[str, Any]) -> bool:
    """Return whether a coordinate lies strictly inside a polygon or multipolygon."""
    index = PolygonIndex([geometry])
    return bool(index.first_owner(np.array([longitude]), np.array([latitude]))[0] == 0)


def aggregate_samples(samples: list[TerritorySample]) -> tuple[TerritorySample, ...]:
    """Combine point weights while retaining distinct sample coordinates."""
    combined: dict[tuple[float, float], TerritorySample] = {}
    for sample in samples:
        key = (sample.latitude, sample.longitude)
        previous = combined.get(key)
        if previous is None:
            combined[key] = sample
            continue
        combined[key] = TerritorySample(
            latitude=sample.latitude,
            longitude=sample.longitude,
            area_weight=previous.area_weight + sample.area_weight,
            longitude_weight=previous.longitude_weight + sample.longitude_weight,
            population_weight=(
                (previous.population_weight or 0) + (sample.population_weight or 0)
                if previous.population_weight is not None or sample.population_weight is not None
                else None
            ),
        )
    return tuple(combined.values())


def _polygon_rings(coordinates: object) -> tuple[Ring, ...]:
    if not isinstance(coordinates, list):
        msg = "Polygon coordinates must be an array."
        raise GeoJSONError(msg)
    rings: list[Ring] = []
    for ring in coordinates:
        if not isinstance(ring, list):
            msg = "Polygon ring must be an array."
            raise GeoJSONError(msg)
        positions: list[Position] = []
        for point in ring:
            if (
                not isinstance(point, list)
                or len(point) < POSITION_DIMENSIONS
                or not isinstance(point[0], (float, int))
                or not isinstance(point[1], (float, int))
            ):
                msg = "Polygon positions must contain numeric longitude and latitude."
                raise GeoJSONError(msg)
            positions.append((float(point[0]), float(point[1])))
        rings.append(tuple(positions))
    return tuple(rings)


def unwrap_polygon(rings: tuple[Ring, ...]) -> tuple[Ring, ...]:
    """Unwrap a polygon's rings into one planar frame, keeping holes beside their outer ring."""
    if not rings:
        return rings
    outer = unwrap_ring(rings[0])
    if not outer:
        return (outer,)
    center = (min(point[0] for point in outer) + max(point[0] for point in outer)) / 2
    holes = []
    for hole in rings[1:]:
        unwrapped = unwrap_ring(hole)
        if unwrapped:
            shift = unwrap_longitude(unwrapped[0][0], center) - unwrapped[0][0]
            unwrapped = tuple((longitude + shift, latitude) for longitude, latitude in unwrapped)
        holes.append(unwrapped)
    return (outer, *holes)


def unwrap_ring(ring: Ring) -> Ring:
    """Make consecutive vertices continuous across the antimeridian (longitudes may leave [-180, 180]).

    A ring that winds once around a pole (Antarctica) ends 360 degrees away from where it started. Such a
    ring is closed through the pole so it becomes a planar polygon spanning the full longitude range.
    """
    if not ring:
        return ring
    unwrapped: list[Position] = [ring[0]]
    for longitude, latitude in ring[1:]:
        unwrapped.append((unwrap_longitude(longitude, unwrapped[-1][0]), latitude))
    first_longitude, first_latitude = unwrapped[0]
    closing_longitude = unwrap_longitude(first_longitude, unwrapped[-1][0])
    if abs(closing_longitude - first_longitude) < HALF_LONGITUDE_TURN:
        return tuple(unwrapped)
    pole = MIN_LATITUDE if sum(point[1] for point in unwrapped) < 0 else MAX_LATITUDE
    if unwrapped[-1] != (closing_longitude, first_latitude):
        unwrapped.append((closing_longitude, first_latitude))
    unwrapped.extend(((closing_longitude, pole), (first_longitude, pole), (first_longitude, first_latitude)))
    return tuple(unwrapped)


def unwrap_longitude(longitude: float, reference: float) -> float:
    """Shift a longitude by whole turns to within 180 degrees of ``reference``."""
    while longitude - reference > HALF_LONGITUDE_TURN:
        longitude -= FULL_LONGITUDE_TURN
    while longitude - reference < -HALF_LONGITUDE_TURN:
        longitude += FULL_LONGITUDE_TURN
    return longitude


def unwrap_longitudes(longitudes: NDArray[np.float64], reference: float) -> NDArray[np.float64]:
    """Vectorized ``unwrap_longitude``: shift each longitude by whole turns to within 180 degrees of ``reference``."""
    difference = longitudes - reference
    down = np.ceil((difference - HALF_LONGITUDE_TURN) / FULL_LONGITUDE_TURN).clip(min=0)
    up = np.ceil((-HALF_LONGITUDE_TURN - difference) / FULL_LONGITUDE_TURN).clip(min=0)
    return longitudes - FULL_LONGITUDE_TURN * down + FULL_LONGITUDE_TURN * up


def _wrap_longitude(longitude: float) -> float:
    return (longitude + HALF_LONGITUDE_TURN) % FULL_LONGITUDE_TURN - HALF_LONGITUDE_TURN


def _contains(rings: tuple[Ring, ...], point: Position) -> bool:
    return _in_ring(rings[0], point) and not any(_in_ring(ring, point) for ring in rings[1:])


def _in_ring(ring: Ring, point: Position) -> bool:
    longitude, latitude = point
    inside = False
    previous_lon, previous_lat = ring[-1]
    for current_lon, current_lat in ring:
        if (current_lat > latitude) != (previous_lat > latitude):
            crossing_lon = (previous_lon - current_lon) * (latitude - current_lat) / (
                previous_lat - current_lat
            ) + current_lon
            if longitude < crossing_lon:
                inside = not inside
        previous_lon, previous_lat = current_lon, current_lat
    return inside


def _polygon_centroid(rings: tuple[Ring, ...]) -> Position:
    area_total = 0.0
    lon_total = 0.0
    lat_total = 0.0
    for index, ring in enumerate(rings):
        twice_area = 0.0
        centroid_lon = 0.0
        centroid_lat = 0.0
        for first, second in zip(ring, (*ring[1:], ring[0]), strict=True):
            cross = first[0] * second[1] - second[0] * first[1]
            twice_area += cross
            centroid_lon += (first[0] + second[0]) * cross
            centroid_lat += (first[1] + second[1]) * cross
        if abs(twice_area) < GEOMETRY_AREA_EPSILON:
            continue
        area = abs(twice_area) / 2
        signed_area = area if index == 0 else -area
        centroid = (centroid_lon / (3 * twice_area), centroid_lat / (3 * twice_area))
        area_total += signed_area
        lon_total += centroid[0] * signed_area
        lat_total += centroid[1] * signed_area
    if abs(area_total) < GEOMETRY_AREA_EPSILON:
        return (
            sum(point[0] for point in rings[0]) / len(rings[0]),
            sum(point[1] for point in rings[0]) / len(rings[0]),
        )
    return lon_total / area_total, lat_total / area_total


def _interior_fallback(rings: tuple[Ring, ...]) -> Position:
    outer = rings[0]
    first = outer[0]
    for point in outer[1:]:
        midpoint = ((first[0] + point[0]) / 2, (first[1] + point[1]) / 2)
        if _contains(rings, midpoint):
            return midpoint
        first = point
    msg = "Could not find an interior point for polygon."
    raise ValueError(msg)


def merge_geometries(geometries: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine Polygon/MultiPolygon GeoJSON geometries into one geometry (no dissolving)."""
    polygons: list[Any] = []
    for geometry in geometries:
        if geometry.get("type") == "Polygon":
            polygons.append(geometry["coordinates"])
        elif geometry.get("type") == "MultiPolygon":
            polygons.extend(geometry["coordinates"])
        else:
            msg = f"Unsupported boundary geometry type: {geometry.get('type')!r}."
            raise ValueError(msg)
    if len(polygons) == 1:
        return {"type": "Polygon", "coordinates": polygons[0]}
    return {"type": "MultiPolygon", "coordinates": polygons}
