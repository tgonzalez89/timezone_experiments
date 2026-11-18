"""Verify accelerated spatial operations against the original geometric rules."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from timezone_experiments.config import ExperimentConfig
from timezone_experiments.data import boundaries as boundary_module
from timezone_experiments.data import timezone_index as timezone_module
from timezone_experiments.data.boundaries import load_territories
from timezone_experiments.data.geometry import PreparedPolygon, aggregate_samples, point_in_geometry
from timezone_experiments.data.models import Territory, TerritorySample
from timezone_experiments.data.population import PopulationPoint, attach_population_weights
from timezone_experiments.data.timezone_index import TimezoneBoundaryIndex, annotate_current_timezones

if TYPE_CHECKING:
    from pathlib import Path


def _box(west: float, south: float, east: float, north: float) -> list[list[float]]:
    return [[west, south], [east, south], [east, north], [west, north], [west, south]]


def test_index_matches_reference_for_holes_overlaps_dateline_and_ring_edges() -> None:
    circle = [
        [50 + 3 * math.cos(angle * math.tau / 512), 20 + 3 * math.sin(angle * math.tau / 512)] for angle in range(513)
    ]
    boundaries: list[tuple[str, dict[str, object]]] = [
        ("first", {"type": "Polygon", "coordinates": [_box(0, 0, 10, 10), _box(2, 2, 4, 4)]}),
        ("overlapping", {"type": "Polygon", "coordinates": [_box(5, 0, 15, 10)]}),
        ("dateline", {"type": "Polygon", "coordinates": [_box(179, -5, -179, 5), _box(179.2, -1, 179.8, 1)]}),
        ("multipart", {"type": "MultiPolygon", "coordinates": [[circle], [_box(-110, 20, -105, 25)]]}),
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


def test_index_reuses_preparation_and_only_checks_nearby_polygons(monkeypatch: pytest.MonkeyPatch) -> None:
    index = TimezoneBoundaryIndex(
        [(str(west), {"type": "Polygon", "coordinates": [_box(west, 0, west + 1, 1)]}) for west in range(-170, 171, 10)]
    )
    original = PreparedPolygon.contains
    checked: list[tuple[float, float]] = []

    def contains(polygon: PreparedPolygon, point: tuple[float, float]) -> bool:
        checked.append(point)
        return original(polygon, point)

    def unexpected_parse(geometry: dict[str, object]) -> None:
        msg = "Repeated lookup reparsed raw GeoJSON."
        raise AssertionError(msg)

    monkeypatch.setattr(PreparedPolygon, "contains", contains)
    monkeypatch.setattr(timezone_module, "iter_polygons", unexpected_parse)
    for _ in range(3):
        assert index.timezone_at(0.5, 0.5) == "0"
        assert index.timezone_at(80, 0.5) is None
    assert len(checked) == 3


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


def test_parallel_sampling_matches_sequential_and_preserves_error_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(boundary_module, "MIN_PARALLEL_SAMPLING_WORK", 0)
    collapsed = [[1, 1], [1, 2], [1, 1], [1, 1]]
    features = [
        {
            "type": "Feature",
            "properties": {"GID_0": "TST", "GID_1": f"TST.{number}_1", "NAME_1": f"Province {number}"},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [_box(number * 5, 0, number * 5 + 2, 2), _box(number * 5 + 0.5, 0.5, number * 5 + 1.5, 1.5)],
                    [collapsed],
                ],
            },
        }
        for number in (1, 2)
    ]
    path = tmp_path / "boundaries.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    config = {"subdivision_rules": {"TST": 1}, "sample_grid_size": 4}
    sequential = load_territories(path, ExperimentConfig.from_mapping({**config, "workers": 1}))
    parallel = load_territories(path, ExperimentConfig.from_mapping({**config, "workers": 2}))
    assert parallel == sequential
    features[1]["geometry"] = {"type": "MultiPolygon", "coordinates": [[collapsed]]}
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"TST:TST.2_1.*non-zero-area"):
        load_territories(path, ExperimentConfig.from_mapping({**config, "workers": 2}))


def test_parallel_overlay_preserves_weight_summation_ties_order_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    monkeypatch.setattr(timezone_module, "LOOKUP_BATCH_SIZE", 2)
    index = TimezoneBoundaryIndex(
        [
            ("west", {"type": "Polygon", "coordinates": [_box(-2, -2, 0, 2)]}),
            ("east", {"type": "Polygon", "coordinates": [_box(0, -2, 2, 2)]}),
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


def test_parallel_population_overlay_preserves_holes_dateline_samples_and_result_order() -> None:
    territories = [
        Territory(
            "TST:1",
            "TST",
            "West",
            0,
            (TerritorySample(0, 0, longitude_weight=3),),
            geometry={"type": "Polygon", "coordinates": [_box(0, 0, 10, 10), _box(2, 2, 4, 4)]},
        ),
        Territory(
            "TST:2",
            "TST",
            "Dateline",
            0,
            (TerritorySample(0, 179, population_weight=1),),
            geometry={"type": "Polygon", "coordinates": [_box(179, -2, -179, 2)]},
        ),
    ]
    points = [PopulationPoint(1, 1, 10), PopulationPoint(3, 3, 20), PopulationPoint(0, -179.5, 30)]
    sequential = attach_population_weights(territories, points)
    parallel = attach_population_weights(territories, points, workers=2)
    assert parallel == sequential
    assert [territory.territory_id for territory in parallel] == ["TST:1", "TST:2"]
    assert [sum(sample.population_weight or 0 for sample in territory.samples) for territory in parallel] == [10, 30]
    with pytest.raises(ValueError, match="requires geometry for 'TST:2'"):
        attach_population_weights([territories[0], replace(territories[1], geometry=None)], points, workers=2)
    with pytest.raises(ValueError, match="workers must be positive"):
        attach_population_weights(territories, points, workers=0)


@pytest.mark.parametrize("workers", [1, 2])
def test_population_index_matches_reference_without_duplicating_overlapping_components(workers: int) -> None:
    geometry: dict[str, object] = {
        "type": "MultiPolygon",
        "coordinates": [
            [_box(170, -10, -170, 10), _box(175, -2, 179, 2)],
            [_box(178, -5, -175, 5)],
        ],
    }
    units = [
        Territory(str(index), "TST", str(index), 0, (TerritorySample(0, 170),), geometry=geometry) for index in range(2)
    ]
    points = [
        PopulationPoint(0, -179, 10),
        PopulationPoint(0, 177, 20),
        PopulationPoint(0, 180, 30),
        PopulationPoint(0, -180, 40),
        PopulationPoint(0, -179, 50),
        PopulationPoint(10, 170, 60),
        PopulationPoint(0, 0, 70),
    ]
    expected = [point for point in points if point_in_geometry(point.longitude, point.latitude, geometry)]
    results = attach_population_weights(units, points, workers=workers)
    for result in results:
        assert [(sample.latitude, sample.longitude, sample.population_weight) for sample in result.samples[1:]] == [
            (point.latitude, point.longitude, point.population) for point in expected
        ]
