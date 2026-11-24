"""Current-timezone boundary index and overlay."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest
from factories import (
    box_ring,
    make_territory,
)

from timezone_experiments.core import geometry as geometry_module
from timezone_experiments.core.geometry import point_in_geometry
from timezone_experiments.core.models import Territory, TerritorySample
from timezone_experiments.data.boundaries import load_timezone_boundaries
from timezone_experiments.data.timezone_index import TimezoneBoundaryIndex, annotate_current_timezones

if TYPE_CHECKING:
    from pathlib import Path


def test_boundary_lookup_annotates_only_territories_without_a_timezone(tmp_path: Path) -> None:
    timezone_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"tzid": "Etc/UTC"},
                "geometry": {"type": "Polygon", "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]]},
            },
        ],
    }
    path = tmp_path / "timezones.json"
    path.write_text(json.dumps(timezone_geojson), encoding="utf-8")
    index = TimezoneBoundaryIndex(load_timezone_boundaries(path))
    assert index.timezone_at(0, 0) == "Etc/UTC"
    assert index.timezone_at(0, 20) is None
    annotated = annotate_current_timezones([make_territory(current_timezone=None)], index)
    assert annotated[0].current_timezone == "Etc/UTC"
    assert annotate_current_timezones([make_territory(longitude=20)], index)[0].current_timezone == "UTC+00:00"


def test_index_requires_at_least_one_boundary() -> None:
    with pytest.raises(ValueError, match="At least one"):
        TimezoneBoundaryIndex([])


@pytest.mark.slow
def test_index_matches_reference_for_holes_overlaps_dateline_and_ring_edges() -> None:
    circle = [
        [50 + 3 * math.cos(angle * math.tau / 512), 20 + 3 * math.sin(angle * math.tau / 512)] for angle in range(513)
    ]
    boundaries: list[tuple[str, dict[str, object]]] = [
        ("first", {"type": "Polygon", "coordinates": [box_ring(0, 0, 10, 10), box_ring(2, 2, 4, 4)]}),
        ("overlapping", {"type": "Polygon", "coordinates": [box_ring(5, 0, 15, 10)]}),
        ("dateline", {"type": "Polygon", "coordinates": [box_ring(179, -5, -179, 5), box_ring(179.2, -1, 179.8, 1)]}),
        ("multipart", {"type": "MultiPolygon", "coordinates": [[circle], [box_ring(-110, 20, -105, 25)]]}),
    ]
    index = TimezoneBoundaryIndex(boundaries)
    points: list[tuple[float, float]] = [
        (latitude, longitude) for latitude in range(-10, 31, 2) for longitude in range(-185, 186, 5)
    ]
    points.extend((latitude + delta, longitude) for longitude, latitude in circle for delta in (-1e-10, 0, 1e-10))
    points.extend(
        (latitude, longitude)
        for longitude in (-181, -180, -179, 179, 179.5, 180, 181, 540)
        for latitude in (-5, -1, 0, 1, 5)
    )
    for latitude, longitude in points:
        expected = next(
            (zone for zone, geometry in boundaries if point_in_geometry(longitude, latitude, geometry)), None
        )
        assert index.timezone_at(latitude, longitude) == expected, (latitude, longitude)


def test_index_prepares_geojson_once(monkeypatch: pytest.MonkeyPatch) -> None:
    index = TimezoneBoundaryIndex(
        [
            (str(west), {"type": "Polygon", "coordinates": [box_ring(west, 0, west + 1, 1)]})
            for west in range(-170, 171, 10)
        ]
    )

    def unexpected_parse(geometry: dict[str, object]) -> None:
        msg = "Repeated lookup reparsed raw GeoJSON."
        raise AssertionError(msg)

    monkeypatch.setattr(geometry_module, "iter_polygons", unexpected_parse)
    for _ in range(3):
        assert index.timezone_at(0.5, 0.5) == "0"
        assert index.timezone_at(80, 0.5) is None


def test_parallel_overlay_preserves_weight_summation_ties_order_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    index = TimezoneBoundaryIndex(
        [
            ("west", {"type": "Polygon", "coordinates": [box_ring(-2, -2, 0, 2)]}),
            ("east", {"type": "Polygon", "coordinates": [box_ring(0, -2, 2, 2)]}),
        ]
    )
    samples = [
        (
            TerritorySample(0, 1, area_weight=1e16),
            TerritorySample(0, -1, area_weight=1e16),
            TerritorySample(0, -1),
            TerritorySample(0, -1),
        ),
        (TerritorySample(50, 50), TerritorySample(0, -1, area_weight=0)),
        (TerritorySample(0, 1, area_weight=0),),
        (TerritorySample(0, 1, area_weight=3), TerritorySample(0, -1, area_weight=1)),
    ]
    territories = [
        Territory(str(number), "TST", str(number), 0, points, current_timezone="Etc/UTC")
        for number, points in enumerate(samples)
    ]
    sequential = annotate_current_timezones(territories, index)
    parallel = annotate_current_timezones(territories, index, workers=2)
    assert parallel == sequential
    assert [territory.current_timezone for territory in parallel] == ["east", "Etc/UTC", "Etc/UTC", "east"]
    assert [territory.territory_id for territory in parallel] == ["0", "1", "2", "3"]
    assert all(result.samples is original.samples for result, original in zip(parallel, territories, strict=True))
    assert "with 2 workers" in caplog.text
