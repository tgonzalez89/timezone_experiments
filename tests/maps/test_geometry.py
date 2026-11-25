"""Map geometry: Plotly winding and gradient simplification."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest
from factories import (
    make_territory,
    map_source,
    territory_trace,
)
from shapely import get_parts, union_all
from shapely.geometry import Polygon, shape

from timezone_experiments import maps
from timezone_experiments.core.config import (
    ExperimentConfig,
)
from timezone_experiments.maps import (
    MAP_FIELDS,
    build_plotly_figure,
)
from timezone_experiments.maps.geometry import gradient_geometries, gradient_geometry
from timezone_experiments.optimization.optimizer import optimize_territory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shapely.geometry.base import BaseGeometry


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
    rendered = shape(territory_trace(figure).geojson["features"][0]["geometry"])
    assert isinstance(rendered, Polygon)
    assert not rendered.exterior.is_ccw
    assert rendered.interiors[0].is_ccw
    assert rendered.equals(shape(original))
    assert geometry == original
    assert len(figure.data) == 6  # fills, zone bands, and the borders and guides


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
    coordinates = territory_trace(figure).geojson["features"][0]["geometry"]["coordinates"]
    # Only the real polygon survives; at the dateline it is cut into its two sides.
    assert len(coordinates) == (2 if dateline else 1)
    assert all(len(polygon) == 1 for polygon in coordinates)
    polygons = [Polygon(polygon[0]) for polygon in coordinates]
    assert all(not polygon.exterior.is_ccw for polygon in polygons)
    assert sum(polygon.area for polygon in polygons) == 200
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


def _zigzag_pair(amplitude: float) -> list[dict[str, object]]:
    """Two territories sharing a finely zigzagging border, as neighboring subdivisions do."""
    border = [(1 + (amplitude if step % 2 else -amplitude), step / 10) for step in range(11)]
    left = [(0, 0), *border, (0, 1), (0, 0)]
    right = [*border, (2, 1), (2, 0), border[0]]
    return [
        {
            "geometry": {"type": "Polygon", "coordinates": [[[x, y] for x, y in ring]]},
            "properties": {"proposed_utc_offset_hours": 0},
        }
        for ring in (left, right)
    ]


def test_neighbors_are_simplified_together_so_no_gap_opens_between_them() -> None:
    features = _zigzag_pair(0.001)
    exact = gradient_geometries(features, 0)
    simplified = gradient_geometries(features, 0.02)
    assert sum(len(shape_.exterior.coords) for shape_ in simplified) < sum(
        len(shape_.exterior.coords) for shape_ in exact
    )
    exact_union, simplified_union = union_all(exact), union_all(simplified)
    assert exact_union.buffer(-0.02).difference(simplified_union).is_empty
    assert simplified[0].intersection(simplified[1]).area == pytest.approx(0, abs=1e-12)


def test_coverage_simplification_keeps_islands_and_exact_geometry_at_zero() -> None:
    island = [[3.01, 0], [3.02, 0], [3.02, 0.01], [3.01, 0.01], [3.01, 0]]
    feature = {
        "geometry": {"type": "MultiPolygon", "coordinates": [[[[0, 0], [2, 0], [2, 3], [0, 3], [0, 0]]], [island]]},
        "properties": {"proposed_utc_offset_hours": 0},
    }
    (simplified,) = gradient_geometries([feature], 0.02)
    assert simplified.equals(gradient_geometry(feature, 0))
    assert gradient_geometries([feature], 0)[0].equals(gradient_geometry(feature, 0))


def test_gradient_geometries_are_computed_once_for_the_same_features(monkeypatch: pytest.MonkeyPatch) -> None:
    from timezone_experiments.maps import geometry as geometry_module  # noqa: PLC0415

    features = _zigzag_pair(0.004)
    calls: list[int] = []
    original = geometry_module._joint_gradient_geometries

    def counted(features: Sequence[dict[str, Any]], tolerance: float, workers: int) -> list[BaseGeometry]:
        calls.append(1)
        return original(features, tolerance, workers)

    monkeypatch.setattr(geometry_module, "_joint_gradient_geometries", counted)
    first = gradient_geometries(features, 0.02)
    second = gradient_geometries(features, 0.02)
    gradient_geometries(features, 0.03)
    assert len(calls) == 2  # the second call reused the first; a different tolerance computed again
    assert all(a.equals(b) for a, b in zip(first, second, strict=True))


def test_territories_dissolve_only_the_parts_that_touch() -> None:
    separate = {
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                [[[2, 0], [3, 0], [3, 1], [2, 1], [2, 0]]],
                [[[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]],  # touches both of the others
                [[[5, 0], [6, 0], [6, 1], [5, 1], [5, 0]]],
            ],
        },
        "properties": {"proposed_utc_offset_hours": 0},
    }
    (merged,) = gradient_geometries([separate], 0)
    assert merged.is_valid
    assert merged.area == pytest.approx(4)
    assert len(get_parts(merged)) == 2  # the three touching squares became one part


def test_neighbors_that_disagree_on_their_shared_border_are_made_to_match() -> None:
    def square(west: float, east: float) -> dict[str, object]:
        ring = [[west, 0], [east, 0], [east, 1], [west, 1], [west, 0]]
        return {"geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {"proposed_utc_offset_hours": 0}}

    # The first reaches 0.004 degrees past the second's edge: they overlap, as neighbors from two sources do.
    features = [square(0, 1.004), square(1.0, 2)]
    exact = gradient_geometries(features, 0)
    assert exact[0].intersection(exact[1]).area > 0
    matched = gradient_geometries(features, 0.01)
    assert matched[0].intersection(matched[1]).area == pytest.approx(0, abs=1e-12)
    assert matched[0].touches(matched[1])
    assert sum(shape_.area for shape_ in matched) == pytest.approx(2, abs=0.01)


def test_a_territory_whose_touching_islands_merge_still_gets_its_overlap_resolved() -> None:
    def feature(rings: list[list[list[float]]]) -> dict[str, object]:
        return {
            "geometry": {"type": "MultiPolygon", "coordinates": [[ring] for ring in rings]},
            "properties": {"proposed_utc_offset_hours": 0},
        }

    # Two islands touching at a point merge into one part when cleaned; the territory must still be cleaned.
    first = feature(
        [[[0, 0], [1.004, 0], [1.004, 1], [0, 1], [0, 0]], [[1.004, 1], [1.5, 1], [1.5, 1.5], [1.004, 1.5], [1.004, 1]]]
    )
    second = feature([[[1.0, 0], [2, 0], [2, 1], [1.0, 1], [1.0, 0]]])
    assert gradient_geometries([first, second], 0)[0].intersection(gradient_geometries([first, second], 0)[1]).area > 0
    cleaned = gradient_geometries([first, second], 0.01)
    assert cleaned[0].intersection(cleaned[1]).area == pytest.approx(0, abs=1e-12)
