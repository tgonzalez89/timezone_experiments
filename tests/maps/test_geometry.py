"""Map geometry: Plotly winding and gradient simplification."""

from __future__ import annotations

from copy import deepcopy

import pytest
from factories import (
    make_territory,
    map_source,
)
from shapely import get_parts
from shapely.geometry import Polygon, shape

from timezone_experiments import maps
from timezone_experiments.core.config import (
    ExperimentConfig,
)
from timezone_experiments.maps import (
    MAP_FIELDS,
    build_plotly_figure,
)
from timezone_experiments.maps.geometry import gradient_geometry
from timezone_experiments.optimization.optimizer import optimize_territory


@pytest.mark.parametrize("clockwise", [False, True])
def test_plotly_winding_preserves_polygon_holes_without_mutating_sources(*, clockwise: bool) -> None:

    outer = [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]
    hole = [[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]
    if clockwise:
        outer.reverse()
        hole.reverse()
    geometry: dict[str, object] = {"type": "Polygon", "coordinates": [outer, hole]}
    original = deepcopy(geometry)
    territory = make_territory(geometry=geometry)
    result = optimize_territory(territory, ExperimentConfig(minimum_noon=None, utc_offsets=(1,)))
    figure = build_plotly_figure([territory], [result], MAP_FIELDS[0])
    rendered = shape(figure.data[0].geojson["features"][0]["geometry"])
    assert isinstance(rendered, Polygon)
    assert not rendered.exterior.is_ccw
    assert rendered.interiors[0].is_ccw
    assert rendered.equals(shape(original))
    assert geometry == original
    assert len(figure.data) == 5


@pytest.mark.parametrize("dateline", [False, True])
def test_plotly_removes_degenerate_components_and_handles_dateline_winding(
    caplog: pytest.LogCaptureFixture, *, dateline: bool
) -> None:

    west, east = (170, -170) if dateline else (0, 20)
    outer = [[west, 0], [east, 0], [east, 10], [west, 10], [west, 0]]
    degenerate = [[west, 0]] * 4
    geometry: dict[str, object] = {"type": "MultiPolygon", "coordinates": [[degenerate], [outer, degenerate]]}
    original = deepcopy(geometry)
    territory = make_territory(geometry=geometry)
    result = optimize_territory(territory, ExperimentConfig(minimum_noon=None, utc_offsets=(1,)))
    figure = build_plotly_figure([territory], [result], MAP_FIELDS[0])
    coordinates = figure.data[0].geojson["features"][0]["geometry"]["coordinates"]
    assert len(coordinates) == len(coordinates[0]) == 1
    ring = [(lon + 360 if dateline and lon < 0 else lon, lat) for lon, lat in coordinates[0][0]]
    polygon = Polygon(ring)
    assert not polygon.exterior.is_ccw
    assert polygon.area == 200
    assert geometry == original
    assert "Skipped 2 zero-area rings" in caplog.text


def test_plotly_rejects_entirely_degenerate_geometry() -> None:
    territory = make_territory(geometry={"type": "Polygon", "coordinates": [[[1, 1]] * 4]})
    result = optimize_territory(make_territory(), ExperimentConfig(minimum_noon=None, utc_offsets=(1,)))
    with pytest.raises(ValueError, match="nonzero-area polygon"):
        build_plotly_figure([territory], [result], MAP_FIELDS[0])


def test_map_geometry_simplification_preserves_topology_and_zero_keeps_exact_geometry() -> None:

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
    territory, result = map_source(geometry)
    feature = maps.result_feature_collection([territory], [result])["features"][0]
    exact = gradient_geometry(feature, 0)
    simplified = gradient_geometry(feature, 0.02)
    assert not exact.equals(simplified)
    assert simplified.is_valid
    simplified_polygon = next(iter(get_parts(simplified)))
    exact_polygon = next(iter(get_parts(exact)))
    assert len(simplified_polygon.exterior.coords) < len(exact_polygon.exterior.coords)
    assert len(simplified_polygon.interiors) == len(exact_polygon.interiors) == 1
    assert simplified.symmetric_difference(exact).area / exact.area < 0.02
    with pytest.raises(ValueError, match="map_geometry_simplification_degrees"):
        maps.build_plotly_figure(
            [territory], [result], "median_solar_noon", maps.MapOptions(map_geometry_simplification_degrees=0.001)
        )
