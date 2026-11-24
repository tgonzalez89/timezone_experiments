"""Sunrise/sunset gradient maps."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
from factories import (
    map_source,
)
from matplotlib.figure import Figure
from plotly.colors import sample_colorscale, unlabel_rgb
from shapely import box, get_parts, intersection, union_all
from shapely.geometry import shape
from shapely.strtree import STRtree

from timezone_experiments import maps
from timezone_experiments.astronomy.events import SOLAR_EVENT_FIELDS, AnnualSolarEvents, annual_solar_events
from timezone_experiments.maps import solar_events as solar_events_module
from timezone_experiments.maps.borders import country_borders
from timezone_experiments.maps.geometry import gradient_geometry
from timezone_experiments.maps.interactive import plotly_figure
from timezone_experiments.maps.registry import spec_for
from timezone_experiments.maps.solar_events import event_cell_areas, event_scale, solar_event_features
from timezone_experiments.maps.static import export_static_map

if TYPE_CHECKING:
    from pathlib import Path


def test_event_gradients_are_two_dimensional_preserve_holes_islands_and_share_pair_scales() -> None:

    geometry: dict[str, object] = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [[0, 50], [1, 50], [1, 51], [0, 51], [0, 50]],
                [[0.1, 50.1], [0.2, 50.1], [0.2, 50.2], [0.1, 50.2], [0.1, 50.1]],
            ],
            [[[2.01, 50], [2.02, 50], [2.02, 50.01], [2.01, 50.01], [2.01, 50]]],
        ],
    }
    territory, result = map_source(geometry)
    collection = maps.result_feature_collection([territory], [result])
    cells = solar_event_features({**collection, "options": maps.MapOptions(workers=2)})
    assert union_all([shape(cell["geometry"]) for cell in cells]).equals(shape(geometry))
    assert len({cell["id"] for cell in cells}) == len(cells)
    assert len({cell["properties"]["latitude"] for cell in cells}) > 1
    assert len({cell["properties"]["longitude"] for cell in cells}) > 1
    for cell in cells:
        props = cell["properties"]
        exact = annual_solar_events(props["latitude"], props["longitude"], 2024, 1)
        assert props["solar_year"] == result.solar_year == 2024
        for field in SOLAR_EVENT_FIELDS:
            assert props[field] == getattr(exact, field).minutes
            assert props["event_dates"][field] == getattr(exact, field).date.isoformat()
    collection["solar_event_features"] = cells
    traces = [plotly_figure(collection, spec_for(field)).data[0] for field in SOLAR_EVENT_FIELDS]
    for trace, field in zip(traces, SOLAR_EVENT_FIELDS, strict=True):
        assert list(trace.z) == [cell["properties"][field] for cell in cells]
        assert trace.marker.line.width == 0
        assert "2024" in trace.customdata[0][4]
        assert "Solar-cycle date" in trace.customdata[0][4]
        assert trace.zmin % 60 == trace.zmax % 60 == 0


@pytest.mark.parametrize("cell_degrees", [0.25, 0.5, 1.0, 2.0])
def test_event_gradient_resolution_preserves_geometry(cell_degrees: float) -> None:

    geometry: dict[str, object] = {
        "type": "Polygon",
        "coordinates": [[[0, 40], [2.1, 40], [2.1, 42.1], [0, 42.1], [0, 40]]],
    }
    territory, result = map_source(geometry)
    collection = maps.result_feature_collection([territory], [result])
    cells = solar_event_features({**collection, "options": maps.MapOptions(event_map_cell_degrees=cell_degrees)})
    assert union_all([shape(cell["geometry"]) for cell in cells]).equals(shape(geometry))
    assert cells
    for cell in cells:
        west, south, east, north = shape(cell["geometry"]).bounds
        assert east - west <= cell_degrees
        assert north - south <= cell_degrees


def test_event_cell_fast_path_matches_exact_clipping_for_boundaries_and_holes() -> None:

    geometry: dict[str, object] = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [4, 0], [4, 4], [2, 4], [2, 2], [0, 2], [0, 0]],
            [[2.5, 0.5], [3.5, 0.5], [3.5, 1.5], [2.5, 1.5], [2.5, 0.5]],
        ],
    }
    territory, result = map_source(geometry)
    feature = maps.result_feature_collection([territory], [result])["features"][0]
    merged = gradient_geometry(feature)
    step = 0.5
    west, south, east, north = merged.bounds
    columns = np.arange(math.floor(west / step), math.ceil(east / step))
    edges = columns * step
    boundary = STRtree(list(get_parts(merged.boundary)))
    for row in range(math.floor(south / step), math.ceil(north / step)):
        fast = dict(event_cell_areas(merged, boundary, columns=columns, row=row, step=step))
        exact = {}
        slices = intersection(
            merged,
            box(edges, row * step, edges + step, (row + 1) * step),
        )
        for column, clipped in zip(columns, slices, strict=True):
            parts = [part for part in get_parts(clipped) if part.geom_type == "Polygon" and part.area > 0]
            if parts:
                exact[int(column)] = parts[0] if len(parts) == 1 else union_all(parts)
        assert set(fast) == set(exact)
        for column, value in fast.items():
            assert value.equals(exact[column])


def test_event_gradient_resolution_changes_cache_and_interactive_cell_count() -> None:
    geometry: dict[str, object] = {
        "type": "Polygon",
        "coordinates": [[[0, 40], [2.1, 40], [2.1, 42.1], [0, 42.1], [0, 40]]],
    }
    territory, result = map_source(geometry)
    detailed = maps.build_plotly_figure(
        [territory], [result], "earliest_sunrise", maps.MapOptions(event_map_cell_degrees=0.5)
    ).data[0]
    coarse = maps.build_plotly_figure(
        [territory], [result], "earliest_sunrise", maps.MapOptions(event_map_cell_degrees=1)
    ).data[0]
    assert len(coarse.z) < len(detailed.z)
    assert (
        "Approximate 0.5-degree cells"
        in maps.build_plotly_figure(
            [territory], [result], "earliest_sunrise", maps.MapOptions(event_map_cell_degrees=0.5)
        )
        .layout.annotations[0]
        .text
    )
    assert (
        "Approximate 1-degree cells"
        in maps.build_plotly_figure(
            [territory], [result], "earliest_sunrise", maps.MapOptions(event_map_cell_degrees=1)
        )
        .layout.annotations[0]
        .text
    )
    with pytest.raises(ValueError, match="event_map_cell_degrees"):
        maps.build_plotly_figure(
            [territory], [result], "earliest_sunrise", maps.MapOptions(event_map_cell_degrees=0.75)
        )


def test_event_gradients_split_dateline_without_clock_or_geometry_seams() -> None:

    territory, result = map_source(
        {"type": "Polygon", "coordinates": [[[179, 10], [-179, 10], [-179, 11], [179, 11], [179, 10]]]}, 14
    )
    cells = solar_event_features(maps.result_feature_collection([territory], [result]))
    expected = union_all([box(179, 10, 180, 11), box(-180, 10, -179, 11)])
    assert union_all([shape(cell["geometry"]) for cell in cells]).equals(expected)
    for field in SOLAR_EVENT_FIELDS:
        values = [cell["properties"][field] for cell in cells]
        assert max(values) - min(values) < 15
    assert all(-180 <= shape(cell["geometry"]).bounds[0] < shape(cell["geometry"]).bounds[2] <= 180 for cell in cells)


def test_missing_events_are_grey_hoverable_and_polar_seasons_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    territory, result = map_source(
        {"type": "Polygon", "coordinates": [[[0, 80], [0.5, 80], [0.5, 80.5], [0, 80.5], [0, 80]]]}
    )
    seasonal = maps.build_plotly_figure([territory], [result], "earliest_sunrise")
    polar_days = re.search(r"Polar day: (\d+) days", seasonal.data[0].customdata[0][4])
    assert polar_days is not None
    assert int(polar_days.group(1)) > 0
    monkeypatch.setattr(
        solar_events_module,
        "annual_solar_events",
        lambda *_args: AnnualSolarEvents(None, None, None, None, 190, 176),
    )
    missing = maps.build_plotly_figure([territory], [result], "latest_sunset")
    assert missing.data[0].z == (None,)
    assert missing.data[1].name == "No sunrise/sunset events"
    assert missing.data[1].customdata[0][3] == "Not applicable"
    assert "no sunrise/sunset events" in missing.data[1].customdata[0][4]
    assert not missing.data[1].showscale


@pytest.mark.parametrize("field", SOLAR_EVENT_FIELDS)
@pytest.mark.parametrize("rasterize", [False, True])
def test_static_event_maps_match_interactive_values_scales_and_clock_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, *, rasterize: bool
) -> None:

    territory, result = map_source({"type": "Polygon", "coordinates": [[[0, 50], [1, 50], [1, 51], [0, 51], [0, 50]]]})
    collection = maps.result_feature_collection([territory], [result])
    collection["options"] = maps.MapOptions(rasterize_static_map_fills=rasterize)
    collection["solar_event_features"] = solar_event_features(collection)
    collection["country_borders"] = country_borders(collection, workers=1)
    collection["territory_borders"] = country_borders(collection, workers=1, group_by="territory_id")
    captured = []
    interactive = plotly_figure(collection, spec_for(field)).data[0]

    def capture(figure: Figure, path: Path, **_kwargs: object) -> None:
        fill = figure.axes[0].collections[0]
        observed = fill.get_array()
        assert observed is not None
        np.testing.assert_allclose(sorted(observed), sorted(interactive.z))
        assert fill.norm.vmin == interactive.zmin
        assert fill.norm.vmax == interactive.zmax
        positions = np.linspace(0, 1, 9)
        samples = interactive.zmin + positions * (interactive.zmax - interactive.zmin)
        colors = [list(stop) for stop in interactive.colorscale]
        expected = np.array([unlabel_rgb(color) for color in sample_colorscale(colors, positions)]) / 255
        np.testing.assert_allclose(np.asarray(fill.cmap(fill.norm(samples)))[:, :3], expected, atol=1 / 128)
        assert [tick.get_text() for tick in figure.axes[1].get_yticklabels()] == list(interactive.colorbar.ticktext)
        assert np.asarray(fill.get_edgecolor()).size == 0
        assert fill.get_rasterized() is rasterize
        captured.append(path.name)

    monkeypatch.setattr(Figure, "savefig", capture)
    export_static_map(collection, spec_for(field), tmp_path)
    assert len(captured) == 2


def test_event_scale_trims_to_data_hours_and_keeps_fixed_colors() -> None:
    scale = event_scale("t", [330.0, 400.0, 430.0])
    assert (scale.minimum, scale.maximum) == (300, 480)
    assert not any(label.startswith(("≤", "≥")) for label in scale.tick_labels)
    positions = [position for position, _ in scale.colors]
    assert positions[0] == 0
    assert positions[-1] == 1
    assert positions == sorted(positions)
    # 05:00 is the red stop whatever the window is.
    wide = event_scale("t", [0.0, 700.0])
    for window in (scale, wide):
        fraction = (300 - window.minimum) / (window.maximum - window.minimum)
        assert dict(window.colors)[fraction] == "#d7301f"


def test_event_scale_clamps_to_the_day_and_marks_the_ends() -> None:
    scale = event_scale("t", [-20.0, 600.0, 1500.0])
    assert (scale.minimum, scale.maximum) == (0, 1440)
    assert scale.tick_labels[0] == "≤ 00:00"
    assert scale.tick_labels[-1] == "≥ 24:00"
    assert scale.colors[0][1] == "#3a0f16"
    assert scale.colors[-1][1] == "#14143c"
