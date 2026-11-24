"""Polygon geometry: sampling, unwrapping, pole caps, and point-in-polygon indexes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from timezone_experiments.core.geometry import PolygonIndex, aggregate_samples, point_in_geometry, sample_geometry
from timezone_experiments.core.models import TerritorySample

if TYPE_CHECKING:
    from collections.abc import Sequence


# The same coast without explicit pole vertices; the ring only "closes" by wrapping around the globe.
POLE_CAP_COAST_ONLY = [[-180, -70], [-90, -75], [0, -70], [90, -75], [180, -70], [-180, -70]]


# Antarctica-like ring as distributed by Natural Earth: the coast runs -180 -> 180 and closes along the pole.
POLE_CAP_WITH_POLE_EDGE = [
    [-180, -70],
    [-90, -75],
    [0, -70],
    [90, -75],
    [180, -70],
    [180, -90],
    [-180, -90],
    [-180, -70],
]


def _polygon(*rings: Sequence[Sequence[float]]) -> dict[str, Any]:
    return {"type": "Polygon", "coordinates": [*rings]}


def test_polygon_sampling_preserves_holes_and_dateline_crossings() -> None:
    holed_polygon = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
            [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]],
        ],
    }
    samples = sample_geometry(holed_polygon, 10)
    assert not any(
        4 <= sample.longitude <= 6 and 4 <= sample.latitude <= 6 and sample.area_weight > 0 for sample in samples
    )
    dateline = {
        "type": "Polygon",
        "coordinates": [[[170, -10], [-170, -10], [-170, 10], [170, 10], [170, -10]]],
    }
    assert point_in_geometry(179, 0, dateline)
    assert point_in_geometry(-179, 0, dateline)
    assert not point_in_geometry(0, 0, dateline)
    dateline_samples = sample_geometry(dateline, 4)
    assert max(sample.area_weight for sample in dateline_samples) < 100
    weighted_samples = sample_geometry(
        {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 2], [0, 2], [0, 0]]]},
        4,
    )
    longitude_weights: dict[float, float] = {}
    for sample in weighted_samples:
        if sample.area_weight > 0:
            longitude_weights[sample.longitude] = longitude_weights.get(sample.longitude, 0) + sample.longitude_weight
    assert len(set(longitude_weights.values())) == 1
    with pytest.raises(ValueError, match="Polygon or MultiPolygon"):
        sample_geometry({"type": "Point", "coordinates": [0, 0]}, 4)
    with pytest.raises(ValueError, match="grid_size"):
        sample_geometry(dateline, 0)
    multi = {"type": "MultiPolygon", "coordinates": [dateline["coordinates"], dateline["coordinates"]]}
    # Both components are sampled on their own 2x2 grid; boundary vertices are no longer samples.
    assert len(sample_geometry(multi, 2)) == 2 * 2 * 2
    assert all(sample.area_weight > 0 for sample in dateline_samples)


@pytest.mark.parametrize(
    "ring",
    [
        [[121.203, -6.7779], [121.2032, -6.7766], [121.203, -6.7779], [121.203, -6.7779]],
        [[0, 0], [1, 1], [2, 2], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
    ],
)
def test_polygon_sampling_skips_zero_area_components(ring: list[list[float]], caplog: pytest.LogCaptureFixture) -> None:
    valid = [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]]
    geometry = {"type": "MultiPolygon", "coordinates": [valid, [ring]]}
    samples = sample_geometry(geometry, 4)
    assert samples == sample_geometry({"type": "Polygon", "coordinates": valid}, 4)
    assert "zero area" in caplog.text
    with pytest.raises(ValueError, match="no non-zero-area"):
        sample_geometry({"type": "Polygon", "coordinates": [ring]}, 4)


def test_polygon_sampling_preserves_tiny_nonzero_area() -> None:
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [[121.203, -6.7779], [121.203000001, -6.7779], [121.203, -6.777899999], [121.203, -6.7779]],
        ],
    }
    samples = sample_geometry(geometry, 1)
    assert any(sample.area_weight > 0 for sample in samples)


def test_sample_aggregation_keeps_order_and_optional_population_weights() -> None:
    samples = [
        TerritorySample(1, 2, area_weight=0.1, longitude_weight=0.2),
        TerritorySample(3, 4, area_weight=2, population_weight=0, longitude_weight=4),
        TerritorySample(1, 2, area_weight=0.2, population_weight=5, longitude_weight=0.3),
        TerritorySample(5, 6, area_weight=0, longitude_weight=0),
    ]
    assert aggregate_samples(samples) == (
        TerritorySample(1, 2, area_weight=0.1 + 0.2, population_weight=5, longitude_weight=0.2 + 0.3),
        samples[1],
        samples[3],
    )


@pytest.mark.parametrize("ring", [POLE_CAP_WITH_POLE_EDGE, POLE_CAP_COAST_ONLY])
def test_pole_cap_contains_points_at_every_longitude(ring: list[list[float]]) -> None:
    geometry = _polygon(ring)
    for longitude in (-179.5, -120, 0, 60, 179.5):
        assert point_in_geometry(longitude, -85, geometry)
    assert not point_in_geometry(0, -60, geometry)


@pytest.mark.parametrize("ring", [POLE_CAP_WITH_POLE_EDGE, POLE_CAP_COAST_ONLY])
def test_pole_cap_samples_reach_the_pole_and_span_all_longitudes(ring: list[list[float]]) -> None:
    samples = [sample for sample in sample_geometry(_polygon(ring), 12) if sample.area_weight > 0]
    assert min(sample.latitude for sample in samples) < -85
    assert min(sample.longitude for sample in samples) < -150
    assert max(sample.longitude for sample in samples) > 150


def test_polygon_index_matches_points_in_pole_caps_and_across_the_antimeridian() -> None:
    dateline = _polygon([[170, -10], [-170, -10], [-170, 10], [170, 10], [170, -10]])
    index = PolygonIndex([_polygon(POLE_CAP_COAST_ONLY), dateline, dateline])
    longitudes = np.array([-179.5, 0, 179.5, 180, 175, 0])
    latitudes = np.array([-85, -85, -85, 0, 0, 0])
    points, owners = index.matches(longitudes, latitudes, workers=2)
    assert list(zip(points.tolist(), owners.tolist(), strict=True)) == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 1),
        (4, 1),
        (3, 2),
        (4, 2),
    ]
    assert index.first_owner(longitudes, latitudes).tolist() == [0, 0, 0, 1, 1, -1]


def test_hole_listed_in_another_longitude_frame_still_excludes_points() -> None:
    outer = [[170, -10], [-170, -10], [-170, 10], [170, 10], [170, -10]]
    hole = [[-178, -2], [-178, 2], [178, 2], [178, -2], [-178, -2]]
    geometry = _polygon(outer, hole)
    assert not point_in_geometry(180, 0, geometry)
    assert point_in_geometry(175, 0, geometry)
    assert not point_in_geometry(0, 0, geometry)


@pytest.mark.parametrize(
    "ring",
    [
        # Wide bar with a tall left post: the bounding-box centre is empty, but the area centroid is inside the bar.
        [[0, 0], [10, 0], [10, 1], [1, 1], [1, 4], [0, 4], [0, 0]],
        # U shape: both the bounding-box centre and the area centroid fall in the notch.
        [[0, 0], [3, 0], [3, 3], [2, 3], [2, 1], [1, 1], [1, 3], [0, 3], [0, 0]],
    ],
    ids=["centroid-inside", "centroid-outside"],
)
def test_polygons_missed_by_the_grid_get_one_interior_sample(ring: list[list[float]]) -> None:
    geometry = {"type": "Polygon", "coordinates": [ring]}
    (sample,) = sample_geometry(geometry, 1)
    assert point_in_geometry(sample.longitude, sample.latitude, geometry)
    assert sample.area_weight > 0
    assert sample.longitude_weight > 0
