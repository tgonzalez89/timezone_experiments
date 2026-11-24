"""Map geometry: Plotly ring winding, unwrapped/simplified gradient areas, and border lines."""

from __future__ import annotations

import logging
from math import floor
from typing import TYPE_CHECKING, Any, cast

from shapely import get_parts, make_valid, union_all
from shapely.affinity import translate
from shapely.errors import GEOSException
from shapely.geometry import Polygon

from timezone_experiments.core.constants import DEFAULT_MAP_SIMPLIFICATION_DEGREES, DEGREES_PER_HOUR
from timezone_experiments.core.geometry import iter_polygons, ring_winding, unwrap_longitude, unwrap_ring

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from timezone_experiments.core.geometry import Ring

logger = logging.getLogger(__name__)
type BorderLine = tuple[tuple[float, float], ...]
_MAX_SIMPLIFICATION_AREA_ERROR = 0.001


def plotly_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with clockwise exteriors and zero-area rings removed, as Plotly's renderer expects."""
    # Plotly's spherical renderer expects clockwise exteriors, unlike RFC 7946.
    polygons: list[list[Ring]] = []
    skipped = 0
    for polygon in iter_polygons(geometry):
        rings: list[Ring] = []
        for index, ring in enumerate(polygon):
            winding = ring_winding(unwrap_ring(ring))
            if winding == 0:
                skipped += 1
                if index == 0:
                    break
                continue
            closed = ring if ring[0] == ring[-1] else (*ring, ring[0])
            if (winding < 0) != (index == 0):
                closed = tuple(reversed(closed))
            rings.append(closed)
        if rings:
            polygons.append(rings)
    if not polygons:
        msg = "Interactive map geometry contains no nonzero-area polygon components."
        raise ValueError(msg)
    if skipped:
        logger.warning("Skipped %d zero-area rings in interactive map geometry.", skipped)
    coordinates = [[[list(point) for point in ring] for ring in polygon] for polygon in polygons]
    return {"type": geometry["type"], "coordinates": coordinates[0] if geometry["type"] == "Polygon" else coordinates}


def gradient_geometry(
    feature: dict[str, Any],
    simplification_degrees: float = DEFAULT_MAP_SIMPLIFICATION_DEGREES,
) -> BaseGeometry:
    """Return a territory's area unwrapped around its proposed offset's meridian, optionally simplified.

    Simplification is kept only where it preserves each part's topology and the area within 0.1%.
    """
    reference = feature["properties"]["proposed_utc_offset_hours"] * DEGREES_PER_HOUR
    polygons = []
    for source in iter_polygons(plotly_geometry(feature["geometry"])):
        outer = unwrap_ring(source[0])
        anchor = unwrap_longitude(outer[0][0], reference)
        shift = anchor - outer[0][0]
        rings = [
            [(x + shift, y) for x, y in outer],
            *[[(unwrap_longitude(x, anchor), y) for x, y in hole] for hole in source[1:]],
        ]
        polygon = Polygon(rings[0], rings[1:])
        valid = polygon if polygon.is_valid else make_valid(polygon)
        polygons.extend(part for part in get_parts(valid) if isinstance(part, Polygon) and part.area > 0)
    geometry = cast("BaseGeometry", union_all(polygons))
    if simplification_degrees == 0:
        return geometry
    simplified = cast(
        "BaseGeometry", union_all([_simplified_part(part, simplification_degrees) for part in get_parts(geometry)])
    )
    area_error = abs(simplified.area - geometry.area) / max(geometry.area, 1e-12)
    return simplified if area_error <= _MAX_SIMPLIFICATION_AREA_ERROR else geometry


def _simplified_part(polygon: BaseGeometry, tolerance: float) -> BaseGeometry:
    try:
        candidate = polygon.simplify(tolerance, preserve_topology=True)
        parts: list[BaseGeometry] = list(get_parts(candidate))
    except GEOSException:
        return polygon
    safe = (
        candidate.is_valid
        and len(parts) == 1
        and parts[0].geom_type == "Polygon"
        and len(parts[0].interiors) == len(polygon.interiors)
        and abs(candidate.area - polygon.area) / max(polygon.area, 1e-12) <= _MAX_SIMPLIFICATION_AREA_ERROR
    )
    return parts[0] if safe else polygon


def wrapped_multipolygon(area: BaseGeometry, longitude: float) -> dict[str, Any]:
    """Return an unwrapped area as GeoJSON MultiPolygon, shifted back into [-180, 180] by whole turns."""
    band = floor((longitude + 180) / 360)
    return {
        "type": "MultiPolygon",
        "coordinates": [
            [[list(point) for point in ring.coords] for ring in (part.exterior, *part.interiors)]
            for part in get_parts(translate(area, xoff=-360 * band))
        ],
    }
