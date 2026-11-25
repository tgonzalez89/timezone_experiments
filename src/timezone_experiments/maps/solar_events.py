"""Sunrise/sunset maps: grid cells colored by the annual earliest/latest sunrise or sunset at each location."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from math import ceil, floor, sqrt
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
from shapely import box, contains_xy, get_parts, intersection, points, union_all
from shapely.strtree import STRtree

from timezone_experiments.astronomy.events import SOLAR_EVENT_FIELDS, annual_solar_events
from timezone_experiments.core.constants import MINUTES_PER_DAY
from timezone_experiments.core.progress import Progress
from timezone_experiments.maps.geometry import (
    GRADIENT_OVERLAP_DEGREES,
    SIMPLIFIED,
    gradient_geometries,
    wrapped_multipolygon,
)
from timezone_experiments.maps.scales import Scale, clock, clock_scale, ramp_window
from timezone_experiments.maps.spec import GradientLayer, MapSpec, observed_values, options_of

if TYPE_CHECKING:
    from numpy import int64
    from numpy.typing import NDArray
    from shapely.geometry.base import BaseGeometry

    from timezone_experiments.maps.spec import MapOptions

logger = logging.getLogger(__name__)
# Fixed color per clock time, shared by all four maps: white at noon, yellow-orange-red toward earlier times and
# green-teal-blue toward later ones, ending in a very dark maroon at 00:00 and a dark navy at 24:00. The two sides
# mirror each other in lightness, and the stops are densest at 03:00-09:00 and 15:00-21:00, where almost all sunrises
# and sunsets fall.
_EVENT_RAMP = (
    (0, "#000000"),  # 00:00
    (120, "#7f007f"),  # 02:00
    (240, "#ff00ff"),  # 04:00
    (360, "#ff0000"),  # 06:00
    (480, "#ffff00"),  # 08:00
    (600, "#ffff7f"),  # 10:00
    (720, "#ffffff"),  # 12:00
    (840, "#00aa00"),  # 14:00
    (960, "#005500"),  # 16:00
    (1080, "#007f7f"),  # 18:00
    (1200, "#0000ff"),  # 20:00
    (1320, "#00007f"),  # 22:00
    (1440, "#000000"),  # 24:00
)


def solar_event_features(collection: dict[str, Any]) -> list[dict[str, Any]]:
    """Clip every territory into grid cells, each carrying all four annual event extremes at a sampled point."""
    options = options_of(collection)
    workers = options.workers
    cell_degrees = options.event_map_cell_degrees
    simplification_degrees = 0 if collection.get(SIMPLIFIED) else options.map_geometry_simplification_degrees
    features = collection["features"]
    logger.info("Preparing annual sunrise/sunset gradients for %d territories with %d workers.", len(features), workers)
    merged = gradient_geometries(features, simplification_degrees, workers)
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        Progress("Preparing sunrise/sunset gradients", len(features), enabled=len(features) > 1) as progress,
    ):
        cells: list[dict[str, Any]] = []
        tasks = ((index, feature, merged[index], cell_degrees) for index, feature in enumerate(features))
        for completed, slices in enumerate(executor.map(_territory_cells, tasks), start=1):
            cells.extend(slices)
            progress.update(completed)
    return cells


def _territory_cells(task: tuple[int, dict[str, Any], BaseGeometry, float]) -> list[dict[str, Any]]:
    territory_index, feature, merged, step = task
    source = feature["properties"]
    west, south, east, north = merged.bounds
    columns = np.arange(floor(west / step), ceil(east / step))
    boundary_index = STRtree(list(get_parts(merged.boundary)))
    features: list[dict[str, Any]] = []
    for row in range(floor(south / step), ceil(north / step)):
        # Cells are sampled at their nominal area but drawn slightly under their neighbors (see event_cell_areas).
        for column, area, grown in event_cell_areas(merged, boundary_index, columns=columns, row=row, step=step):
            point = area.representative_point()
            longitude = (point.x + 180) % 360 - 180
            events = annual_solar_events(point.y, longitude, source["solar_year"], source["proposed_utc_offset_hours"])
            properties: dict[str, Any] = {
                "territory_index": territory_index,
                "latitude": point.y,
                "longitude": longitude,
                "solar_year": source["solar_year"],
                "polar_day_days": events.polar_day_days,
                "polar_night_days": events.polar_night_days,
                "event_dates": {},
            }
            for field in SOLAR_EVENT_FIELDS:
                event = getattr(events, field)
                properties[field] = None if event is None else event.minutes
                properties["event_dates"][field] = None if event is None else event.date.isoformat()
            features.append(
                {
                    "type": "Feature",
                    "id": f"{feature['id']}:{row}:{column}",
                    "properties": properties,
                    "geometry": wrapped_multipolygon(grown, point.x),
                }
            )
    return features


def event_cell_areas(
    merged: BaseGeometry,
    boundary_index: STRtree,
    *,
    columns: NDArray[int64],
    row: int,
    step: float,
) -> list[tuple[int, BaseGeometry, BaseGeometry]]:
    """Return one grid row's cells inside a territory as (column, area, grown area), clipping only near the boundary.

    ``area`` is the cell's nominal area, where its value is sampled. ``grown area`` is that area enlarged by
    ``GRADIENT_OVERLAP_DEGREES`` to the east and north, but never past the territory or the dateline half it lies
    in; it is what gets drawn. Cells are drawn west to east and south to north, so each one slides under its eastern
    and northern neighbors, and viewers cannot leave hairline seams between them. Growing one way only keeps every
    border vertex in about half as many cells as growing on all sides would.
    """
    margin = GRADIENT_OVERLAP_DEGREES
    edges = columns * step
    centers_y = np.full(len(columns), (row + 0.5) * step)
    near = np.zeros(len(columns), dtype=bool)
    half = np.floor((edges + step / 2 + 180) / 360)
    grown_east = np.minimum(edges + step + margin, 180 + 360 * half)
    south, north = row * step, (row + 1) * step
    grown_north = min(north + margin, 90)
    # Half a grown cell's diagonal is the farthest any point of it lies from its center, so a cell whose center is
    # farther than that from the boundary is wholly inside or outside; only the cells near the boundary are clipped.
    # (The distance is generous for a cell grown only east and north, which is safe.)
    near_positions = boundary_index.query(
        points(edges + step / 2, centers_y), predicate="dwithin", distance=(step / 2 + margin) * sqrt(2)
    )[0]
    near[near_positions] = True
    inside = np.asarray(contains_xy(merged, edges + step / 2, centers_y))
    areas: list[tuple[int, BaseGeometry, BaseGeometry]] = [
        (
            int(columns[position]),
            box(edges[position], south, edges[position] + step, north),
            box(edges[position], south, grown_east[position], grown_north),
        )
        for position in np.flatnonzero(inside & ~near)
    ]
    uncertain = np.flatnonzero(near)
    if not len(uncertain):
        return areas
    count = len(uncertain)
    nominal_boxes = box(edges[uncertain], np.full(count, south), edges[uncertain] + step, np.full(count, north))
    grown_boxes = box(edges[uncertain], np.full(count, south), grown_east[uncertain], np.full(count, grown_north))
    for position, clipped, grown in zip(
        uncertain, intersection(merged, nominal_boxes), intersection(merged, grown_boxes), strict=True
    ):
        area, grown_area = _polygonal(clipped), _polygonal(grown)
        if area is None or grown_area is None:
            continue
        areas.append((int(columns[position]), area, grown_area))
    areas.sort(key=lambda item: item[0])
    return areas


def _polygonal(geometry: BaseGeometry) -> BaseGeometry | None:
    """Return the polygons of a clipping result as one geometry, or None when nothing with area remains."""
    parts = [part for part in get_parts(geometry) if part.geom_type == "Polygon" and part.area > 0]
    if not parts:
        return None
    return cast("BaseGeometry", parts[0] if len(parts) == 1 else union_all(parts))


def event_scale(title: str, values: list[float]) -> Scale:
    """Return the fixed time-to-color ramp, trimmed to the whole hours the values occupy within 00:00-24:00.

    A color always means the same clock time, so the window only decides how much of the ramp the colorbar shows.
    Values outside the day (a polar or date-line shift into the neighboring day) take the end colors, and that end's
    tick reads "≤ 00:00" or "≥ 24:00".
    """
    minimum = floor(max(0, min(values, default=0)) / 60) * 60
    maximum = ceil(min(MINUTES_PER_DAY, max(values, default=MINUTES_PER_DAY)) / 60) * 60
    if minimum == maximum:
        minimum, maximum = max(0, minimum - 60), min(MINUTES_PER_DAY, maximum + 60)
    return clock_scale(
        title,
        minimum,
        maximum,
        ramp_window(_EVENT_RAMP, minimum, maximum),
        clamped=(min(values, default=minimum) < minimum, max(values, default=maximum) > maximum),
    )


def event_hover(properties: dict[str, Any], field: str) -> str:
    """Return the hover block for one sampled cell."""
    solar_date = properties["event_dates"][field]
    return "<br>".join(
        [
            f"<b>Annual solar events ({properties['solar_year']}) | sampled location</b>",
            f"Location: {properties['latitude']:.3f}, {properties['longitude']:.3f}",
            f"Solar-cycle date: {solar_date}" if solar_date else "Not applicable: polar day or night during the year",
            f"Polar day: {properties['polar_day_days']} days; polar night: {properties['polar_night_days']} days",
            "No extremes where any day has polar day or night; day suffixes are relative to the solar-cycle date.",
            "Flat apparent horizon; standard refraction; no terrain/elevation correction.",
        ]
    )


SOLAR_EVENT_LAYER = GradientLayer(
    name="solar_events",
    key="solar_event_features",
    build=solar_event_features,
    dependency=lambda properties, options: {
        "solar_year": properties["solar_year"],
        "cell_degrees": options.event_map_cell_degrees,
    },
)


class SolarEventMap(MapSpec):
    """Annual earliest/latest sunrise or sunset at each location, under the territory's proposed offset."""

    layer: GradientLayer | None = SOLAR_EVENT_LAYER
    missing_label: str | None = "Polar day or night in the year"

    def __init__(self, field: str) -> None:
        """Create the map for one of ``SOLAR_EVENT_FIELDS``."""
        self.field = field
        self.title = f"Annual {field.replace('_', ' ').title()} (Proposed)"
        self.hover_label = f"Annual {field.replace('_', ' ')} here (approx.)"

    @override
    def scale(self, features: list[dict[str, Any]]) -> Scale:
        title = f"{self.field.replace('_', ' ').title()}\n(proposed)"
        return event_scale(title, observed_values(features, self.field))

    @override
    def value_text(self, value: float | None) -> object:
        return "Not applicable" if value is None else clock(value)

    @override
    def details(self, properties: dict[str, Any], row: dict[str, Any]) -> str:
        return event_hover(properties, self.field)

    @override
    def footnote(self, options: MapOptions) -> str | None:
        return (
            f"Approximate {options.event_map_cell_degrees:g}-degree cells; yearly extremes; "
            "grey = polar day or night on at least one day; colors show clock times (white = noon, dark = midnight)"
        )
