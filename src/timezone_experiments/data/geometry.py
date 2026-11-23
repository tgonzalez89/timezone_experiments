"""Lightweight GeoJSON polygon operations used by boundary loaders."""

from __future__ import annotations

# Spherical and coordinate-domain constants are intrinsic to geometry calculations.
import logging
import math
from array import array
from collections import Counter
from typing import TYPE_CHECKING

from timezone_experiments.constants import (
    FULL_LONGITUDE_TURN,
    GEOMETRY_AREA_EPSILON,
    HALF_LONGITUDE_TURN,
    MIN_POLYGON_RING_POINTS,
    POSITION_DIMENSIONS,
)
from timezone_experiments.data.models import TerritorySample

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

type Position = tuple[float, float]
type Ring = tuple[Position, ...]

logger = logging.getLogger(__name__)


class PreparedPolygon:
    """Cache polygon bounds and latitude-indexed edges for repeated exact lookups."""

    def __init__(self, rings: tuple[Ring, ...]) -> None:
        """Prepare already-unwrapped rings without changing ray-crossing rules."""
        if not rings or any(not ring for ring in rings):
            msg = "Prepared polygons require non-empty rings."
            raise ValueError(msg)
        self.reference_longitude = rings[0][0][0]
        self._rings = tuple(_PreparedRing(ring) for ring in rings)
        self.bounds = self._rings[0].bounds

    def contains(self, point: Position) -> bool:
        """Test an unwrapped coordinate, respecting holes and boundary conventions."""
        return self._rings[0].contains(point) and not any(ring.contains(point) for ring in self._rings[1:])


class _PreparedRing:
    def __init__(self, ring: Ring) -> None:
        self.bounds = (
            min(point[0] for point in ring),
            min(point[1] for point in ring),
            max(point[0] for point in ring),
            max(point[1] for point in ring),
        )
        self._edges = array("d")
        band_count = max(1, min(1024, len(ring) // 16))
        height = self.bounds[3] - self.bounds[1]
        self._scale = band_count / height if height else 0
        self._bands = tuple(array("I") for _ in range(band_count))
        previous_lon, previous_lat = ring[-1]
        for current_lon, current_lat in ring:
            if previous_lat != current_lat:
                offset = len(self._edges)
                self._edges.extend((previous_lon, previous_lat, current_lon, current_lat))
                first = self._band(min(previous_lat, current_lat))
                last = self._band(max(previous_lat, current_lat))
                for band in self._bands[first : last + 1]:
                    band.append(offset)
            previous_lon, previous_lat = current_lon, current_lat

    def _band(self, latitude: float) -> int:
        return max(0, min(len(self._bands) - 1, int((latitude - self.bounds[1]) * self._scale)))

    def contains(self, point: Position) -> bool:
        longitude, latitude = point
        min_lon, min_lat, max_lon, max_lat = self.bounds
        if not (min_lon <= longitude <= max_lon and min_lat <= latitude < max_lat):
            return False
        inside = False
        for offset in self._bands[self._band(latitude)]:
            previous_lon = self._edges[offset]
            previous_lat = self._edges[offset + 1]
            current_lon = self._edges[offset + 2]
            current_lat = self._edges[offset + 3]
            if (current_lat > latitude) != (previous_lat > latitude):
                # Match _in_ring's operation order, including rounding at polygon edges.
                crossing_lon = (previous_lon - current_lon) * (latitude - current_lat) / (
                    previous_lat - current_lat
                ) + current_lon
                if longitude < crossing_lon:
                    inside = not inside
        return inside


def iter_polygons(geometry: dict[str, Any]) -> Iterator[tuple[Ring, ...]]:
    """Yield polygons from GeoJSON Polygon or MultiPolygon geometry."""
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        yield _polygon_rings(coordinates)
    elif geometry_type == "MultiPolygon":
        if not isinstance(coordinates, list):
            msg = "MultiPolygon coordinates must be an array."
            raise ValueError(msg)
        for polygon in coordinates:
            yield _polygon_rings(polygon)
    else:
        msg = f"Expected Polygon or MultiPolygon geometry, got {geometry_type!r}."
        raise ValueError(msg)


def sample_geometry(geometry: dict[str, Any], grid_size: int) -> tuple[TerritorySample, ...]:
    """Create approximate area-weighted samples for polygonal GeoJSON."""
    if grid_size < 1:
        msg = "grid_size must be positive."
        raise ValueError(msg)
    samples: list[TerritorySample] = []
    boundary_positions: list[Position] = []
    for polygon in iter_polygons(geometry):
        rings = tuple(_unwrap_ring(ring) for ring in polygon)
        if not rings or not _ring_has_area(rings[0]):
            logger.warning("Skipping degenerate polygon component with zero area.")
            continue
        boundary_positions.extend(position for ring in rings for position in ring[:-1])
        outer = rings[0]
        min_lon = min(point[0] for point in outer)
        max_lon = max(point[0] for point in outer)
        min_lat = min(point[1] for point in outer)
        max_lat = max(point[1] for point in outer)
        grid_cell_weight = (max_lon - min_lon) * (max_lat - min_lat) / (grid_size * grid_size)
        longitude_cell_weight = (max_lon - min_lon) / grid_size
        polygon_sample_start = len(samples)
        polygon_samples: list[tuple[int, float, float, float]] = []
        for row in range(grid_size):
            latitude = min_lat + (row + 0.5) * (max_lat - min_lat) / grid_size
            for column in range(grid_size):
                longitude = min_lon + (column + 0.5) * (max_lon - min_lon) / grid_size
                if _contains(rings, (longitude, latitude)):
                    polygon_samples.append(
                        (
                            column,
                            latitude,
                            longitude,
                            grid_cell_weight * max(math.cos(math.radians(latitude)), 1e-8),
                        ),
                    )
        column_counts = Counter(column for column, *_ in polygon_samples)
        samples.extend(
            TerritorySample(
                latitude=latitude,
                longitude=_wrap_longitude(longitude),
                area_weight=area_weight,
                longitude_weight=longitude_cell_weight / column_counts[column],
            )
            for column, latitude, longitude, area_weight in polygon_samples
        )
        if len(samples) == polygon_sample_start:
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
    samples.extend(
        TerritorySample(latitude=latitude, longitude=_wrap_longitude(longitude), area_weight=0, longitude_weight=0)
        for longitude, latitude in boundary_positions
    )
    return tuple(samples)


def _ring_has_area(ring: Ring) -> bool:
    return _ring_winding(ring) != 0


def _ring_winding(ring: Ring) -> float:
    if len(ring) < MIN_POLYGON_RING_POINTS:
        return 0.0
    # Translation avoids cancelling tiny areas at large longitude/latitude coordinates.
    origin_lon, origin_lat = ring[0]
    return math.fsum(
        (first[0] - origin_lon) * (second[1] - origin_lat) - (second[0] - origin_lon) * (first[1] - origin_lat)
        for first, second in zip(ring, (*ring[1:], ring[0]), strict=True)
    )


def point_in_geometry(longitude: float, latitude: float, geometry: dict[str, Any]) -> bool:
    """Return whether a coordinate lies in a polygon or multipolygon."""
    for polygon in iter_polygons(geometry):
        rings = tuple(_unwrap_ring(ring) for ring in polygon)
        adjusted_longitude = _unwrap_longitude(longitude, rings[0][0][0])
        if _contains(rings, (adjusted_longitude, latitude)):
            return True
    return False


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
        raise TypeError(msg)
    rings: list[Ring] = []
    for ring in coordinates:
        if not isinstance(ring, list):
            msg = "Polygon ring must be an array."
            raise TypeError(msg)
        positions: list[Position] = []
        for point in ring:
            if (
                not isinstance(point, list)
                or len(point) < POSITION_DIMENSIONS
                or not isinstance(point[0], (float, int))
                or not isinstance(point[1], (float, int))
            ):
                msg = "Polygon positions must contain numeric longitude and latitude."
                raise TypeError(msg)
            positions.append((float(point[0]), float(point[1])))
        rings.append(tuple(positions))
    return tuple(rings)


def _unwrap_ring(ring: Ring) -> Ring:
    if not ring:
        return ring
    unwrapped = [ring[0]]
    for longitude, latitude in ring[1:]:
        unwrapped.append((_unwrap_longitude(longitude, unwrapped[-1][0]), latitude))
    return tuple(unwrapped)


def _unwrap_longitude(longitude: float, reference: float) -> float:
    while longitude - reference > HALF_LONGITUDE_TURN:
        longitude -= FULL_LONGITUDE_TURN
    while longitude - reference < -HALF_LONGITUDE_TURN:
        longitude += FULL_LONGITUDE_TURN
    return longitude


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
