"""Verify polygon unwrapping, pole caps, and antimeridian point tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from timezone_experiments.data.geometry import (
    PreparedPolygon,
    _unwrap_longitude,
    point_in_geometry,
    sample_geometry,
    unwrap_polygon,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

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
# The same coast without explicit pole vertices; the ring only "closes" by wrapping around the globe.
POLE_CAP_COAST_ONLY = [[-180, -70], [-90, -75], [0, -70], [90, -75], [180, -70], [-180, -70]]


def _polygon(*rings: Sequence[Sequence[float]]) -> dict[str, Any]:
    return {"type": "Polygon", "coordinates": [*rings]}


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


def test_prepared_pole_cap_contains_points_unwrapped_around_its_reference_longitude() -> None:
    ring = tuple((float(longitude), float(latitude)) for longitude, latitude in POLE_CAP_COAST_ONLY)
    prepared = PreparedPolygon(unwrap_polygon((ring,)))
    assert prepared.bounds[2] - prepared.bounds[0] == pytest.approx(360)
    for longitude in (-179.5, 0, 179.5):
        assert prepared.contains((_unwrap_longitude(longitude, prepared.reference_longitude), -85))


def test_hole_listed_in_another_longitude_frame_still_excludes_points() -> None:
    outer = [[170, -10], [-170, -10], [-170, 10], [170, 10], [170, -10]]
    hole = [[-178, -2], [-178, 2], [178, 2], [178, -2], [-178, -2]]
    geometry = _polygon(outer, hole)
    assert not point_in_geometry(180, 0, geometry)
    assert point_in_geometry(175, 0, geometry)
    assert not point_in_geometry(0, 0, geometry)
