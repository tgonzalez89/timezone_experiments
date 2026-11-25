"""Map geometry: Plotly ring winding, unwrapped/simplified gradient areas, and border lines."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from math import floor
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from shapely import (
    STRtree,
    bounds,
    box,
    coverage_clean,
    coverage_simplify,
    get_num_coordinates,
    get_num_geometries,
    get_parts,
    get_type_id,
    intersects,
    is_empty,
    is_valid,
    make_valid,
    multipolygons,
    prepare,
    union_all,
)
from shapely import area as shapely_area
from shapely.affinity import translate
from shapely.errors import GEOSException
from shapely.geometry import Polygon

from timezone_experiments.core.constants import DEFAULT_MAP_SIMPLIFICATION_DEGREES, DEGREES_PER_HOUR
from timezone_experiments.core.geometry import iter_polygons, ring_winding, unwrap_longitude, unwrap_ring

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shapely.geometry.base import BaseGeometry

    from timezone_experiments.core.geometry import Ring

logger = logging.getLogger(__name__)
type BorderLine = tuple[tuple[float, float], ...]
_MAX_SIMPLIFICATION_AREA_ERROR = 0.001
# Each gradient cell or band is drawn this far (degrees) under its neighbors, inside its own territory, so
# viewers that anti-alias every polygon separately cannot leave hairline seams between abutting cells.
GRADIENT_OVERLAP_DEGREES = 0.125
# Set on a collection whose feature geometry has already been jointly simplified (see maps/simplified.py).
SIMPLIFIED = "geometry_simplified"
# shapely.get_type_id values of the polygonal geometry types.
_POLYGON_TYPE, _MULTIPOLYGON_TYPE = 3, 6
# Gaps between neighbors narrower than this (degrees; about 11 km) are closed. Thousands of slivers from sources that
# disagree on a border sit below it, while genuine lakes and seas inside a territory are far wider.
_CLEANING_GAP_DEGREES = 0.1
_MAX_CLEANING_AREA_CHANGE = 0.05
_HALF_TURN = 180


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
    """Return one territory's area unwrapped around its proposed offset's meridian, optionally simplified.

    Simplification is kept only where it preserves each part's topology and the area within 0.1%. It treats the
    territory alone; use ``gradient_geometries`` to draw several territories, so that neighbors still fit together.
    """
    geometry = _unwrapped_geometry(feature)
    return geometry if simplification_degrees == 0 else _simplified_alone(geometry, simplification_degrees)


def gradient_geometries(
    features: Sequence[dict[str, Any]],
    simplification_degrees: float = DEFAULT_MAP_SIMPLIFICATION_DEGREES,
    workers: int = 1,
) -> list[BaseGeometry]:
    """Return every territory's unwrapped area, simplified together so that neighbors keep sharing their borders.

    Simplifying each territory on its own moves the two sides of a shared border differently and opens gaps (and
    overlaps) between neighbors, which show as uncolored slivers when zoomed in. A coverage simplification moves a
    shared border once for both sides. Islands that would change by more than 0.1% keep their exact shape, as do
    territories the simplification would empty or invalidate.
    """
    # The median-noon and sunrise/sunset layers draw the same territories one after the other; computing the joint
    # simplification (the costliest step of either) once serves both. Holding the features keeps their ids unique.
    key = (simplification_degrees, tuple((id(feature["geometry"]), id(feature["properties"])) for feature in features))
    cached = _recent_gradient_geometries
    if cached is not None and cached[0] == key:
        return list(cached[2])
    result = _joint_gradient_geometries(features, simplification_degrees, workers)
    _remember_gradient_geometries(key, list(features), result)
    return list(result)


_recent_gradient_geometries: tuple[tuple[Any, ...], list[dict[str, Any]], list[BaseGeometry]] | None = None


def _remember_gradient_geometries(
    key: tuple[Any, ...], features: list[dict[str, Any]], result: list[BaseGeometry]
) -> None:
    global _recent_gradient_geometries  # noqa: PLW0603
    _recent_gradient_geometries = (key, features, result)


def _joint_gradient_geometries(
    features: Sequence[dict[str, Any]], simplification_degrees: float, workers: int
) -> list[BaseGeometry]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        exact = list(executor.map(_unwrapped_geometry, features))
    if simplification_degrees == 0:
        return exact
    exact = _cleaned_coverage(exact, _CLEANING_GAP_DEGREES)
    try:
        simplified = list(coverage_simplify(exact, simplification_degrees))
    except GEOSException as error:
        logger.warning("Coverage simplification failed (%s); simplifying each territory on its own.", error)
        return [_simplified_alone(geometry, simplification_degrees) for geometry in exact]
    return _guard_simplified(exact, simplified, simplification_degrees)


def _cleaned_coverage(exact: list[BaseGeometry], gap_width: float) -> list[BaseGeometry]:
    """Make neighbors share their borders exactly: no overlaps, gaps under ``gap_width`` closed, edges identical.

    Neighbors from different sources (or the same source at 1:50m) rarely agree on a shared border; each draws
    its own slightly different outline, which shows as doubled lines and slivers. A territory whose area would
    change by more than 5%, or that would turn invalid, keeps its own geometry (overlapping duplicates such as
    Monaco and its Natural Earth twin are the usual cases). Touching islands may merge into one part.
    """
    original = np.asarray(exact, dtype=object)
    try:
        cleaned = np.asarray(coverage_clean(original, gap_width=gap_width), dtype=object)
    except GEOSException as error:
        logger.warning("Coverage cleaning failed (%s); neighbors keep their own borders.", error)
        return exact
    change = np.abs(shapely_area(cleaned) - shapely_area(original)) / np.maximum(shapely_area(original), 1e-12)
    keep = (
        ~is_empty(cleaned)
        & is_valid(cleaned)
        & np.isin(get_type_id(cleaned), (_POLYGON_TYPE, _MULTIPOLYGON_TYPE))
        & (change <= _MAX_CLEANING_AREA_CHANGE)
    )
    return list(np.where(keep, cleaned, original))


def _guard_simplified(
    exact: list[BaseGeometry], simplified: list[BaseGeometry], tolerance: float
) -> list[BaseGeometry]:
    """Return the coverage-simplified territories, restoring the parts that would change too much.

    A part (an island) whose area would change by more than 0.1% keeps its exact shape unless it touches another
    territory, where the shared border must stay as simplified for both sides. A territory the simplification
    emptied, invalidated, or whose parts it dropped, merged, or moved keeps its exact shape altogether.
    """
    exact_array, simplified_array = np.asarray(exact, dtype=object), np.asarray(simplified, dtype=object)
    usable = (
        ~is_empty(simplified_array)
        & is_valid(simplified_array)
        & np.isin(get_type_id(simplified_array), (_POLYGON_TYPE, _MULTIPOLYGON_TYPE))
        & (get_num_geometries(exact_array) == get_num_geometries(simplified_array))
    )
    result = list(exact)
    positions = np.flatnonzero(usable)
    if not len(positions):
        return result
    exact_parts, owner = get_parts(exact_array[positions], return_index=True)
    simplified_parts = get_parts(simplified_array[positions])
    # Parts keep their order, so a simplified part must still sit where its exact part is (a guard against a mix-up).
    moved = np.abs(bounds(exact_parts) - bounds(simplified_parts)).max(axis=1) > 10 * tolerance
    area = shapely_area(exact_parts)
    too_different = (
        np.abs(shapely_area(simplified_parts) - area) / np.maximum(area, 1e-12) > _MAX_SIMPLIFICATION_AREA_ERROR
    )
    candidates = np.flatnonzero(too_different)
    shared = np.zeros(len(exact_parts), dtype=bool)
    if len(candidates):
        # Which restorable parts touch another territory? Index the small parts (not whole territories, which are
        # expensive to test), find bounding-box overlaps, and test only those few pairs exactly.
        every_part, every_owner = get_parts(exact_array, return_index=True)
        asked, found = STRtree(every_part).query(exact_parts[candidates])
        other = every_owner[found] != positions[owner[candidates[asked]]]
        asked, found = asked[other], found[other]
        # Preparing the (often huge) neighbors, not the small candidates, makes each pair test cheap.
        prepare(every_part)
        touching = intersects(every_part[found], exact_parts[candidates[asked]])
        shared[candidates[asked[touching]]] = True
    restore = too_different & ~shared
    unusable = np.bincount(owner[moved], minlength=len(positions)) > 0
    changed = np.bincount(owner[restore], minlength=len(positions)) > 0
    for territory, position in enumerate(positions):
        if unusable[territory]:
            continue
        if not changed[territory]:
            result[position] = simplified[position]
            continue
        first, last = np.searchsorted(owner, [territory, territory + 1])
        kept = np.where(restore[first:last], exact_parts[first:last], simplified_parts[first:last])
        combined = multipolygons(kept)
        # Parts of one territory do not overlap, but a restored island may now touch a simplified neighbor part.
        result[position] = combined if combined.is_valid else cast("BaseGeometry", union_all(kept))
    return result


def _unwrapped_geometry(feature: dict[str, Any]) -> BaseGeometry:
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
    return _dissolved(polygons)


def _dissolved(polygons: list[BaseGeometry]) -> BaseGeometry:
    """Return the union of polygons, dissolving only those that touch or overlap another one.

    Territories are mostly disjoint islands, for which a full union spends its time noding geometry that never meets.
    """
    if len(polygons) == 1:
        return polygons[0]
    parts = np.asarray(polygons, dtype=object)
    # Bounding-box overlaps first, then exact tests on those pairs with the (often huge) second part prepared.
    asked, found = STRtree(parts).query(parts)
    later = asked < found
    asked, found = asked[later], found[later]
    sizes = get_num_coordinates(parts)
    swap = sizes[asked] > sizes[found]
    larger, smaller = np.where(swap, asked, found), np.where(swap, found, asked)
    prepare(parts[np.unique(larger)])
    touching = intersects(parts[larger], parts[smaller])
    asked, found = larger, smaller
    involved = np.unique(np.concatenate([asked[touching], found[touching]]))
    if not len(involved):
        return cast("BaseGeometry", multipolygons(parts))
    dissolved = list(get_parts(union_all(parts[involved])))
    rest = np.setdiff1d(np.arange(len(parts)), involved)
    merged = [*dissolved, *parts[rest]]
    return cast("BaseGeometry", merged[0] if len(merged) == 1 else multipolygons(merged))


def _simplified_alone(geometry: BaseGeometry, simplification_degrees: float) -> BaseGeometry:
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


def wrapped_geometry(area: BaseGeometry, source_type: str = "MultiPolygon") -> dict[str, Any]:
    """Return an unwrapped area as GeoJSON, cut at the dateline into pieces shifted back into [-180, 180].

    A source ``Polygon`` stays a ``Polygon`` unless the dateline cut it in two.
    """
    west, _, east, _ = area.bounds
    pieces: list[BaseGeometry] = []
    for band in range(floor((west + _HALF_TURN) / 360), floor((east + _HALF_TURN) / 360) + 1):
        inside = area
        if band != 0 or west < -_HALF_TURN or east > _HALF_TURN:
            inside = area.intersection(box(-_HALF_TURN + 360 * band, -90, _HALF_TURN + 360 * band, 90))
        pieces.extend(
            translate(part, xoff=-360 * band)
            for part in get_parts(inside)
            if part.geom_type == "Polygon" and part.area > 0
        )
    coordinates = [
        [[list(point) for point in ring.coords] for ring in (part.exterior, *part.interiors)] for part in pieces
    ]
    if source_type == "Polygon" and len(coordinates) == 1:
        return {"type": "Polygon", "coordinates": coordinates[0]}
    return {"type": "MultiPolygon", "coordinates": coordinates}


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
