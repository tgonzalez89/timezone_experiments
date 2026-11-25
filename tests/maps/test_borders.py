"""Country and territory border lines."""

from __future__ import annotations

from copy import deepcopy
from itertools import pairwise

import pytest
from factories import (
    make_territory,
)
from shapely.geometry import LineString, MultiLineString, box, mapping

from timezone_experiments.core.config import (
    ExperimentConfig,
)
from timezone_experiments.maps import (
    result_feature_collection,
)
from timezone_experiments.maps.borders import country_borders, country_lines
from timezone_experiments.optimization.optimizer import optimize_territory


@pytest.mark.parametrize("workers", [1, 2])
def test_country_borders_dissolve_internal_edges_but_keep_international_edges(workers: int) -> None:

    territories = [
        make_territory(
            identifier,
            geometry={
                "type": "Polygon",
                "coordinates": [[[west, 0], [east, 0], [east, 2], [west, 2], [west, 0]]],
            },
        )
        for identifier, west, east in (("AAA:1", 0, 1), ("AAA:2", 1, 2), ("BBB:1", 2, 3))
    ]
    config = ExperimentConfig(minimum_noon=None, utc_offsets=(0,))
    results = [optimize_territory(territory, config) for territory in territories]
    original = deepcopy([territory.geometry for territory in territories])
    collection = result_feature_collection(territories, results)
    lines = country_borders(collection, workers=workers)
    boundary = MultiLineString(lines)
    assert boundary.intersection(LineString([(1, 0.1), (1, 1.9)])).is_empty
    assert boundary.intersection(LineString([(2, 0.1), (2, 1.9)])).length == pytest.approx(1.8)
    assert [territory.geometry for territory in territories] == original


def test_country_borders_join_dateline_subdivisions_without_cross_map_segments() -> None:
    geometries = [
        {
            "type": "Polygon",
            "coordinates": [[[west, -5], [east, -5], [east, 5], [west, 5], [west, -5]]],
        }
        for west, east in ((170, 180), (-180, -170))
    ]
    lines = country_lines(geometries)
    assert lines
    for line in lines:
        assert all(-180 <= longitude <= 180 for longitude, _ in line)
        for first, second in pairwise(line):
            assert abs(second[0] - first[0]) <= 10
            assert not (abs(first[0]) == abs(second[0]) == 180 and first[1] != second[1])


def test_country_borders_remove_overlaps_and_preserve_holes() -> None:

    geometries = [
        {
            "type": "Polygon",
            "coordinates": [
                [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
                [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]],
            ],
        },
        {"type": "Polygon", "coordinates": [[[3, 0], [5, 0], [5, 4], [3, 4], [3, 0]]]},
    ]
    boundary = MultiLineString(country_lines(geometries))
    assert boundary.intersection(LineString([(3, 0.1), (3, 3.9)])).is_empty
    assert boundary.intersection(LineString([(4, 0.1), (4, 3.9)])).is_empty
    assert boundary.intersection(LineString([(1, 1), (2, 1)])).length == 1


@pytest.mark.parametrize("gap", [0.001, 1e-9])
def test_country_borders_remove_enclosed_subdivision_seams_without_removing_small_islands(gap: float) -> None:

    hole = [(0.2, 1.5), (0.4, 1.5), (0.4, 2), (0.2, 2), (0.2, 1.5)]
    geometries = [
        {
            "type": "Polygon",
            "coordinates": [
                [[0, 0], [1, 0], [1, 4], [0, 4], [0, 0]],
                hole,
            ],
        },
        mapping(box(1 + gap, 0, 3, 4)),
        mapping(box(0, 0, 3, 1)),
        mapping(box(0, 3, 3, 4)),
        mapping(box(5, 0, 5 + gap, gap)),
    ]
    original = deepcopy(geometries)
    boundary = MultiLineString(country_lines(geometries))
    expected = MultiLineString(
        [
            [(0, 0), (3, 0), (3, 4), (0, 4), (0, 0)],
            hole,
            list(box(5, 0, 5 + gap, gap).exterior.coords),
        ],
    )
    assert boundary.equals(expected)
    assert geometries == original


def test_excluded_areas_are_outlined_with_their_holes() -> None:
    from timezone_experiments.maps.borders import excluded_outlines  # noqa: PLC0415

    area = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
            [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]],
        ],
    }
    outer, hole = excluded_outlines([area])
    assert outer[0] == outer[-1] == (0.0, 0.0)
    assert len(outer) == len(hole) == 5
    assert excluded_outlines([]) == []
