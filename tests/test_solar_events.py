from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from matplotlib.figure import Figure
from plotly.colors import sample_colorscale, unlabel_rgb
from shapely import box, get_parts, intersection, union_all
from shapely.geometry import shape
from shapely.strtree import STRtree

from timezone_experiments.astronomy.events import (
    SOLAR_EVENT_FIELDS,
    AnnualSolarEvents,
    annual_solar_events,
)
from timezone_experiments.config import ExperimentConfig
from timezone_experiments.data.models import Territory, TerritorySample
from timezone_experiments.optimization.optimizer import AssignmentResult, optimize_territory
from timezone_experiments.visualization import maps

if TYPE_CHECKING:
    from pathlib import Path


def test_annual_solar_events_capture_london_seasonal_extremes_not_averages() -> None:
    events = annual_solar_events(51.5, 0, 2025, 0)
    # Approximate London times with permanent UTC, not summer DST.
    for field, expected, months in zip(
        SOLAR_EVENT_FIELDS, (223, 486, 951, 1221), ((6,), (1, 12), (12,), (6,)), strict=True
    ):
        event = getattr(events, field)
        assert event is not None
        assert event.minutes == pytest.approx(expected, abs=3)
        assert event.date.month in months
    assert events.polar_day_days == events.polar_night_days == 0


def test_solar_events_preserve_fractional_offsets_day_displacement_and_dateline_aliases() -> None:
    baseline = annual_solar_events(70, 0, 2024, 0)
    shifted = annual_solar_events(70, 15, 2024, 5.75)
    for field in SOLAR_EVENT_FIELDS:
        original, actual = getattr(baseline, field), getattr(shifted, field)
        assert original is not None
        assert actual is not None
        assert actual.minutes == pytest.approx(original.minutes + 285)
        assert actual.date == original.date
    latest = shifted.latest_sunset
    assert latest is not None
    assert latest.minutes > 1440
    assert "(+1 day)" in maps._noon_clock(latest.minutes)
    assert annual_solar_events(10, -179, 2024, 14) == annual_solar_events(10, -179, 2024, -10)


@pytest.mark.parametrize("latitude", [-80, 80])
def test_polar_extremes_use_only_real_event_days_and_support_leap_years(latitude: float) -> None:
    events = annual_solar_events(latitude, 0, 2024, 0)
    assert 0 < events.polar_day_days < 366
    assert 0 < events.polar_night_days < 366
    assert events.polar_day_days + events.polar_night_days < 366
    minutes = []
    for field in SOLAR_EVENT_FIELDS:
        event = getattr(events, field)
        assert event is not None
        assert event.date.year == 2024
        minutes.append(event.minutes)
    assert minutes[0] < minutes[1]
    assert minutes[2] < minutes[3]


@pytest.mark.parametrize(("latitude", "longitude", "offset"), [(91, 0, 0), (0, 181, 0), (0, 0, float("inf"))])
def test_solar_events_reject_invalid_coordinates_and_offsets(latitude: float, longitude: float, offset: float) -> None:
    with pytest.raises(ValueError, match="valid latitude/longitude"):
        annual_solar_events(latitude, longitude, 2025, offset)


def _map_source(geometry: dict[str, object], offset: float = 1) -> tuple[Territory, AssignmentResult]:
    territory = Territory(
        territory_id="TST:1",
        country_code="TST",
        name="Test",
        admin_level=0,
        samples=(TerritorySample(50, 0),),
        geometry=geometry,
    )
    result = optimize_territory(territory, ExperimentConfig(year=2024, minimum_noon=None, utc_offsets=(offset,)))
    return territory, result


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
    territory, result = _map_source(geometry)
    collection = maps._result_feature_collection([territory], [result])
    cells = maps._solar_event_features(collection, workers=2)
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
    traces = [maps._build_plotly_figure(collection, field).data[0] for field in SOLAR_EVENT_FIELDS]
    for trace, field in zip(traces, SOLAR_EVENT_FIELDS, strict=True):
        assert list(trace.z) == [cell["properties"][field] for cell in cells]
        assert trace.marker.line.width == 0
        assert "2024" in trace.customdata[0][4]
        assert "Solar-cycle date" in trace.customdata[0][4]
        assert trace.zmin % 60 == trace.zmax % 60 == 0
    for first, last in ((0, 1), (2, 3)):
        assert traces[first].zmin == traces[last].zmin
        assert traces[first].zmax == traces[last].zmax


@pytest.mark.parametrize("cell_degrees", [0.25, 0.5, 1.0, 2.0])
def test_event_gradient_resolution_preserves_geometry(cell_degrees: float) -> None:

    geometry: dict[str, object] = {
        "type": "Polygon",
        "coordinates": [[[0, 40], [2.1, 40], [2.1, 42.1], [0, 42.1], [0, 40]]],
    }
    territory, result = _map_source(geometry)
    collection = maps._result_feature_collection([territory], [result])
    cells = maps._solar_event_features(collection, workers=1, cell_degrees=cell_degrees)
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
    territory, result = _map_source(geometry)
    feature = maps._result_feature_collection([territory], [result])["features"][0]
    merged = maps._gradient_geometry(feature)
    step = 0.5
    west, south, east, north = merged.bounds
    columns = np.arange(math.floor(west / step), math.ceil(east / step))
    edges = columns * step
    boundary = STRtree(list(get_parts(merged.boundary)))
    for row in range(math.floor(south / step), math.ceil(north / step)):
        fast = dict(maps._event_cell_areas(merged, boundary, columns=columns, edges=edges, row=row, step=step))
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


def test_map_geometry_simplification_preserves_topology_and_is_opt_in() -> None:

    points = [
        [0, 0],
        [1, 0],
        [1.0001, 0.0001],
        [2, 0],
        [2, 1],
        [1.0001, 0.9999],
        [1, 1],
        [0, 1],
        [0, 0],
    ]
    geometry: dict[str, object] = {
        "type": "Polygon",
        "coordinates": [points, [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6], [0.4, 0.4]]],
    }
    territory, result = _map_source(geometry)
    feature = maps._result_feature_collection([territory], [result])["features"][0]
    exact = maps._gradient_geometry(feature, 0)
    simplified = maps._gradient_geometry(feature, 0.02)
    assert exact.equals(maps._gradient_geometry(feature))
    assert simplified.is_valid
    simplified_polygon = next(iter(get_parts(simplified)))
    exact_polygon = next(iter(get_parts(exact)))
    assert len(simplified_polygon.exterior.coords) < len(exact_polygon.exterior.coords)
    assert len(simplified_polygon.interiors) == len(exact_polygon.interiors) == 1
    assert simplified.symmetric_difference(exact).area / exact.area < 0.02
    with pytest.raises(ValueError, match="map_geometry_simplification_degrees"):
        maps._gradient_geometry(feature, 0.001)


def test_event_gradient_resolution_changes_cache_and_interactive_cell_count() -> None:
    geometry: dict[str, object] = {
        "type": "Polygon",
        "coordinates": [[[0, 40], [2.1, 40], [2.1, 42.1], [0, 42.1], [0, 40]]],
    }
    territory, result = _map_source(geometry)
    detailed = maps.build_plotly_figure([territory], [result], "earliest_sunrise", event_map_cell_degrees=0.5).data[0]
    coarse = maps.build_plotly_figure([territory], [result], "earliest_sunrise", event_map_cell_degrees=1).data[0]
    assert len(coarse.z) < len(detailed.z)
    assert (
        "Approximate 0.5-degree cells"
        in maps.build_plotly_figure([territory], [result], "earliest_sunrise", event_map_cell_degrees=0.5)
        .layout.annotations[0]
        .text
    )
    assert (
        "Approximate 1-degree cells"
        in maps.build_plotly_figure([territory], [result], "earliest_sunrise", event_map_cell_degrees=1)
        .layout.annotations[0]
        .text
    )
    with pytest.raises(ValueError, match="event_map_cell_degrees"):
        maps.build_plotly_figure([territory], [result], "earliest_sunrise", event_map_cell_degrees=0.75)


def test_event_gradients_split_dateline_without_clock_or_geometry_seams() -> None:

    territory, result = _map_source(
        {"type": "Polygon", "coordinates": [[[179, 10], [-179, 10], [-179, 11], [179, 11], [179, 10]]]}, 14
    )
    cells = maps._solar_event_features(maps._result_feature_collection([territory], [result]), workers=1)
    expected = union_all([box(179, 10, 180, 11), box(-180, 10, -179, 11)])
    assert union_all([shape(cell["geometry"]) for cell in cells]).equals(expected)
    for field in SOLAR_EVENT_FIELDS:
        values = [cell["properties"][field] for cell in cells]
        assert max(values) - min(values) < 15
    assert all(-180 <= shape(cell["geometry"]).bounds[0] < shape(cell["geometry"]).bounds[2] <= 180 for cell in cells)


def test_missing_events_are_grey_hoverable_and_polar_seasons_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    territory, result = _map_source(
        {"type": "Polygon", "coordinates": [[[0, 80], [0.5, 80], [0.5, 80.5], [0, 80.5], [0, 80]]]}
    )
    seasonal = maps.build_plotly_figure([territory], [result], "earliest_sunrise")
    assert "Polar day:" in seasonal.data[0].customdata[0][4]
    assert "Polar day: 0 days" not in seasonal.data[0].customdata[0][4]
    monkeypatch.setattr(
        maps,
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

    territory, result = _map_source({"type": "Polygon", "coordinates": [[[0, 50], [1, 50], [1, 51], [0, 51], [0, 50]]]})
    collection = maps._result_feature_collection([territory], [result])
    collection["solar_event_features"] = maps._solar_event_features(collection, workers=1)
    collection["rasterize_static_map_fills"] = rasterize
    collection["country_borders"] = maps._country_borders(collection, workers=1)
    collection["territory_borders"] = maps._country_borders(collection, workers=1, group_by="territory_id")
    captured = []
    interactive = maps._build_plotly_figure(collection, field).data[0]

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
    maps._export_static_map(collection, field, tmp_path)
    assert len(captured) == 2
