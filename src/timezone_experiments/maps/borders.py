"""Country and territory outlines dissolved from the mapped territories, split at the dateline."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from math import floor
from typing import TYPE_CHECKING, Any, Literal

from shapely import LineString, box, get_parts, line_merge, make_valid, union_all
from shapely.geometry import Polygon, shape

from timezone_experiments.core.fingerprint import json_digest
from timezone_experiments.core.geometry import unwrap_longitude
from timezone_experiments.core.progress import Progress
from timezone_experiments.maps.cache import cache_feature_data, read_feature_cache, write_feature_cache
from timezone_experiments.maps.spec import options_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from timezone_experiments.maps.geometry import BorderLine

logger = logging.getLogger(__name__)
type BorderGrouping = Literal["country_code", "territory_id"]


def cached_borders(
    collection: dict[str, Any], group_by: BorderGrouping, geometry_digests: Sequence[str]
) -> list[BorderLine]:
    """Return the outlines of features grouped by country or territory, reusing the disk cache when unchanged."""
    options = options_of(collection)
    excluded = collection.get("excluded_areas", []) if group_by == "country_code" else []
    dependency: list[dict[str, Any]] = [
        {"id": feature["id"], "group": feature["properties"][group_by], "geometry": digest}
        for feature, digest in zip(collection["features"], geometry_digests, strict=True)
    ]
    if excluded:
        dependency.append({"excluded_areas": json_digest(excluded)})
    cache = cache_feature_data(options.cache_directory, f"{group_by}_borders", dependency)
    cached = read_feature_cache(cache)
    if isinstance(cached, list):
        try:
            borders = [
                tuple((float(point[0]), float(point[1])) for point in line) for line in cached if isinstance(line, list)
            ]
        except IndexError, TypeError, ValueError:
            pass
        else:
            if len(borders) == len(cached):
                logger.info("Using cached %s borders.", group_by)
                return borders
    borders = country_borders(collection, workers=options.workers, group_by=group_by)
    if excluded:
        borders.extend(excluded_outlines(excluded))
    if group_by == "territory_id":
        borders = internal_borders(borders, cached_borders(collection, "country_code", geometry_digests))
    write_feature_cache(cache, borders)
    return borders


def excluded_outlines(areas: Sequence[dict[str, Any]]) -> list[BorderLine]:
    """Return the outlines of areas left out of the territories (see ``load_excluded_areas``), holes included."""
    return [
        tuple((x, y) for x, y, *_ in ring.coords)
        for area in areas
        for part in get_parts(shape(area))
        if isinstance(part, Polygon) and not part.is_empty
        for ring in (part.exterior, *part.interiors)
    ]


def internal_borders(territory_lines: list[BorderLine], country_lines: list[BorderLine]) -> list[BorderLine]:
    """Return the territory outlines that are not country outlines, merged into long lines.

    Coastlines and international borders appear in both layers and are drawn by the country layer on top, so the
    territory layer keeps only the borders inside countries. Both layers come from the same vertices, so shared
    edges match exactly, and the drawing is unchanged while the territory layer shrinks to a fraction of its size.
    """
    if not territory_lines or not country_lines:
        return territory_lines
    internal = union_all([LineString(line) for line in territory_lines]).difference(
        union_all([LineString(line) for line in country_lines])
    )
    merged = line_merge(internal) if not internal.is_empty else internal
    return [tuple(line.coords) for line in get_parts(merged) if line.geom_type == "LineString" and not line.is_empty]


def country_borders(
    collection: dict[str, Any], *, workers: int, group_by: BorderGrouping = "country_code"
) -> list[BorderLine]:
    """Dissolve the features of each group into outline lines, in group order regardless of worker count."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for feature in collection["features"]:
        groups.setdefault(feature["properties"][group_by], []).append(feature["geometry"])
    kind = "countries" if group_by == "country_code" else "territories"
    logger.info("Preparing outlines for %d %s with %d workers.", len(groups), kind, workers)
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        Progress(
            "Preparing country borders" if group_by == "country_code" else "Preparing territory borders",
            len(groups),
            enabled=len(groups) > 1,
        ) as progress,
    ):
        borders: list[BorderLine] = []
        # GEOS releases the GIL; map preserves country order across worker counts.
        for completed, lines in enumerate(executor.map(country_lines, groups.values()), start=1):
            borders.extend(lines)
            progress.update(completed)
    return borders


def country_lines(geometries: list[dict[str, Any]]) -> list[BorderLine]:
    """Return one group's dissolved outline: exteriors plus genuine holes, clipped and wrapped at the dateline."""
    polygons = []
    reference = None
    for geometry in geometries:
        for part in get_parts(shape(geometry)):
            if not isinstance(part, Polygon) or part.is_empty:
                continue
            if reference is None:
                reference = part.exterior.coords[0][0]
            rings = [
                [(unwrap_longitude(x, reference), y) for x, y in ring.coords]
                for ring in (part.exterior, *part.interiors)
            ]
            polygon = Polygon(rings[0], rings[1:])
            if polygon.area > 0:
                valid = polygon if polygon.is_valid else make_valid(polygon)
                polygons.extend(part for part in get_parts(valid) if part.geom_type in {"Polygon", "MultiPolygon"})
    if not polygons:
        msg = "Country outlines require nonzero-area polygon geometry."
        raise ValueError(msg)
    source_hole_edges = union_all(
        [
            ring
            for geometry in polygons
            for polygon in get_parts(geometry)
            if isinstance(polygon, Polygon)
            for ring in polygon.interiors
        ]
    )
    rings = []
    for polygon in get_parts(union_all(polygons)):
        rings.append(polygon.exterior)
        # Mismatched subdivision edges can enclose sliver gaps after merging.
        # Only original interior-ring edges can define genuine country holes.
        rings.extend(ring for ring in polygon.interiors if source_hole_edges.covers(ring))
    boundary = line_merge(union_all(rings))  # fewer, longer lines
    west, _, east, _ = boundary.bounds
    lines: list[BorderLine] = []
    # Clip before wrapping, so dateline crossings never draw a line across the map.
    for band in range(floor((west + 180) / 360), floor((east + 180) / 360) + 1):
        clipped = boundary.intersection(box(-180 + 360 * band, -90, 180 + 360 * band, 90))
        lines.extend(
            tuple((x - 360 * band, y) for x, y in line.coords)
            for line in get_parts(clipped)
            if line.geom_type == "LineString" and not line.is_empty
        )
    return lines
