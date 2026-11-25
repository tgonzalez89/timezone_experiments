"""Median solar-noon map: 0.25-degree longitude bands colored by the annual median noon under the proposed offset."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from math import ceil, floor
from typing import TYPE_CHECKING, Any, override

import numpy as np
from shapely import box, get_parts, intersection, union_all
from shapely.geometry import Polygon

from timezone_experiments.core.constants import DEGREES_PER_HOUR, MINUTES_PER_HOUR, mean_solar_noon_utc_minutes
from timezone_experiments.core.geometry import unwrap_longitude
from timezone_experiments.core.progress import Progress
from timezone_experiments.maps.geometry import (
    GRADIENT_OVERLAP_DEGREES,
    SIMPLIFIED,
    gradient_geometries,
    wrapped_multipolygon,
)
from timezone_experiments.maps.scales import (
    NOON_MINUTES,
    SHIFT_COLORS,
    Scale,
    blend_colors,
    clock,
    clock_scale,
    noon_hover,
)
from timezone_experiments.maps.spec import GradientLayer, MapSpec, observed_values, options_of

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from timezone_experiments.maps.spec import MapOptions

logger = logging.getLogger(__name__)
_BAND_DEGREES = 0.25
# Color limits: noon +/- 3 hours. Values outside get the endpoint colors; hover and CSV values are not clamped.
_COLOR_LIMITS = (9 * 60, 15 * 60)


def median_noon_features(collection: dict[str, Any]) -> list[dict[str, Any]]:
    """Clip every territory into longitude bands, each carrying the median noon at its midpoint."""
    options = options_of(collection)
    workers = options.workers
    simplification_degrees = 0 if collection.get(SIMPLIFIED) else options.map_geometry_simplification_degrees
    features = collection["features"]
    logger.info("Preparing median-noon gradients for %d territories with %d workers.", len(features), workers)
    merged = gradient_geometries(features, simplification_degrees, workers)
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        Progress("Preparing median-noon gradients", len(features), enabled=len(features) > 1) as progress,
    ):
        bands: list[dict[str, Any]] = []
        tasks = ((index, feature, merged[index]) for index, feature in enumerate(features))
        for completed, slices in enumerate(executor.map(_territory_bands, tasks), start=1):
            bands.extend(slices)
            progress.update(completed)
    return bands


def _territory_bands(task: tuple[int, dict[str, Any], BaseGeometry]) -> list[dict[str, Any]]:
    territory_index, feature, merged = task
    properties = feature["properties"]
    reference = properties["proposed_utc_offset_hours"] * DEGREES_PER_HOUR
    west, _, east, _ = merged.bounds
    indices = np.arange(floor(west / _BAND_DEGREES), ceil(east / _BAND_DEGREES))
    edges = indices * _BAND_DEGREES
    # Bands are drawn west to east, each one slightly under its eastern neighbor (but within its dateline half), so
    # no seams show between bands.
    half = np.floor((edges + _BAND_DEGREES / 2 + 180) / 360)
    drawn_east = np.minimum(edges + _BAND_DEGREES + GRADIENT_OVERLAP_DEGREES, 180 + 360 * half)
    # Vectorized GEOS clipping releases the GIL for the shared worker pool.
    slices = intersection(merged, box(edges, -90, edges + _BAND_DEGREES, 90))
    drawn_slices = intersection(merged, box(edges, -90, drawn_east, 90))
    features: list[dict[str, Any]] = []
    for index, clipped, drawn in zip(indices, slices, drawn_slices, strict=True):
        parts = [part for part in get_parts(clipped) if isinstance(part, Polygon) and part.area > 0]
        if not parts:
            continue
        area = parts[0] if len(parts) == 1 else union_all(parts)
        drawn_parts = [part for part in get_parts(drawn) if isinstance(part, Polygon) and part.area > 0]
        drawn_area = drawn_parts[0] if len(drawn_parts) == 1 else union_all(drawn_parts)
        west, _, east, _ = area.bounds
        longitude = (west + east) / 2
        median = (
            mean_solar_noon_utc_minutes(unwrap_longitude(longitude, reference))
            - properties["noon"]["median_equation_of_time_minutes"]
            + MINUTES_PER_HOUR * properties["proposed_utc_offset_hours"]
        )
        features.append(
            {
                "type": "Feature",
                "id": f"{feature['id']}:{index}",
                "properties": {
                    "territory_index": territory_index,
                    "median_solar_noon": median,
                    "longitude": longitude - 360 * floor((longitude + 180) / 360),
                },
                "geometry": wrapped_multipolygon(drawn_area, longitude),
            }
        )
    return features


def noon_scale(values: list[float]) -> Scale:
    """Fit the observed medians to whole hours within 09:00-15:00; 12:00 is white, earlier blue, later orange."""
    lower, upper = _COLOR_LIMITS
    minimum = floor(max(lower, min(upper, min(values, default=NOON_MINUTES))) / 60) * 60
    maximum = ceil(max(lower, min(upper, max(values, default=NOON_MINUTES))) / 60) * 60
    if minimum == maximum:
        minimum, maximum = max(lower, minimum - 60), min(upper, maximum + 60)
    blue, white, orange = (color for _, color in SHIFT_COLORS)
    if maximum <= NOON_MINUTES:
        colors = [(0.0, blue), (1.0, blend_colors(blue, white, (maximum - minimum) / (NOON_MINUTES - minimum)))]
    elif minimum >= NOON_MINUTES:
        colors = [
            (0.0, blend_colors(white, orange, (minimum - NOON_MINUTES) / (maximum - NOON_MINUTES))),
            (1.0, orange),
        ]
    else:
        colors = [(0.0, blue), ((NOON_MINUTES - minimum) / (maximum - minimum), white), (1.0, orange)]
    return clock_scale(
        "Median noon\n(proposed)",
        minimum,
        maximum,
        colors,
        clamped=(min(values, default=NOON_MINUTES) < minimum, max(values, default=NOON_MINUTES) > maximum),
    )


MEDIAN_NOON_LAYER = GradientLayer(
    name="median_noon",
    key="median_noon_features",
    build=median_noon_features,
    dependency=lambda properties, _options: {
        "median_equation_of_time_minutes": properties["noon"]["median_equation_of_time_minutes"]
    },
)


class MedianNoonMap(MapSpec):
    """Annual median of daily solar noon at each longitude, under the territory's proposed offset."""

    field = "median_solar_noon"
    title = "Annual Median Solar Noon (Proposed)"
    layer: GradientLayer | None = MEDIAN_NOON_LAYER
    hover_label = "Annual median noon here (approx.)"
    missing_label: str | None = None

    @override
    def scale(self, features: list[dict[str, Any]]) -> Scale:
        return noon_scale(observed_values(features, self.field))

    @override
    def value_text(self, value: float | None) -> object:
        return None if value is None else clock(value)

    @override
    def details(self, properties: dict[str, Any], row: dict[str, Any]) -> str:
        return noon_hover(row["noon"])

    @override
    def footnote(self, options: MapOptions) -> str | None:
        return None
