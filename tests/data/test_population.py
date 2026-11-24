"""Population grid validation, overlay, and weighted coverage."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from factories import (
    SQUARE,
    TOY_BOUNDARIES,
    box_ring,
    make_territory,
)

from timezone_experiments.core.config import (
    ExperimentConfig,
)
from timezone_experiments.core.geometry import point_in_geometry
from timezone_experiments.core.models import Territory, TerritorySample
from timezone_experiments.data.boundaries import load_territories
from timezone_experiments.data.population import PopulationPoint, attach_population_weights, load_population_csv
from timezone_experiments.optimization.assessment import (
    evaluate_coverage,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_population_overlay_and_weighted_coverage(tmp_path: Path) -> None:
    territory = load_territories(TOY_BOUNDARIES, ExperimentConfig(sample_grid_size=3))[0]
    with_population = attach_population_weights(
        [territory],
        [PopulationPoint(latitude=4, longitude=4, population=90)],
    )[0]
    assert any(sample.population_weight == 90 for sample in with_population.samples)
    passes = [sample.population_weight == 90 for sample in with_population.samples]
    summary = evaluate_coverage(with_population, passes, "population")
    assert summary.compliance_rate == 1
    assert summary.population_covered == 1
    csv_path = tmp_path / "population_example.csv"
    csv_path.write_text("latitude,longitude,population\n4,4,90\n", encoding="utf-8")
    assert load_population_csv(csv_path)[0].population == 90


def test_population_csv_requires_the_documented_header(tmp_path: Path) -> None:
    path = tmp_path / "population.csv"
    path.write_text("lat,lon,count\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain"):
        load_population_csv(path)


def test_parallel_population_overlay_preserves_holes_dateline_samples_and_result_order() -> None:
    territories = [
        Territory(
            "TST:1",
            "TST",
            "West",
            0,
            (TerritorySample(0, 0, longitude_weight=3),),
            geometry={"type": "Polygon", "coordinates": [box_ring(0, 0, 10, 10), box_ring(2, 2, 4, 4)]},
        ),
        Territory(
            "TST:2",
            "TST",
            "Dateline",
            0,
            (TerritorySample(0, 179, population_weight=1),),
            geometry={"type": "Polygon", "coordinates": [box_ring(179, -2, -179, 2)]},
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
            [box_ring(170, -10, -170, 10), box_ring(175, -2, 179, 2)],
            [box_ring(178, -5, -175, 5)],
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


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("1,2,nan", "population must be finite"),
        ("1,2,inf", "population must be finite"),
        ("1,2,-1", "population must be finite"),
        ("91,2,3", "latitude must be between"),
        ("1,-181,3", "longitude must be between"),
        ("1,2,abc", "could not convert"),
        ("1,2", "missing columns"),
    ],
)
def test_population_csv_rows_are_validated_with_file_and_line(tmp_path: Path, row: str, message: str) -> None:
    path = tmp_path / "population.csv"
    path.write_text(f"latitude,longitude,population\n0,0,1\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"population\.csv:3: .*{message}"):
        load_population_csv(path)


@pytest.mark.parametrize("workers", [1, 2])
def test_population_overlay_failures_name_the_territory(workers: int) -> None:
    territories = [
        make_territory("TST:good", samples=(TerritorySample(0.5, 0.5),), geometry=SQUARE),
        make_territory("TST:bad", samples=(TerritorySample(0.5, 0.5),)),
    ]
    with pytest.raises(ValueError, match="requires geometry for 'TST:bad'"):
        attach_population_weights(territories, [PopulationPoint(0.5, 0.5, 10)], workers=workers)
