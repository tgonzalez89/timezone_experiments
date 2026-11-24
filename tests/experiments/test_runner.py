"""Experiment runner: ordering, validation, progress, and failure context."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from factories import (
    SQUARE,
    make_territory,
)

from timezone_experiments.app.errors import describe_error
from timezone_experiments.core.config import (
    ExperimentConfig,
)
from timezone_experiments.core.models import TerritorySample
from timezone_experiments.experiments.results_io import write_results_csv
from timezone_experiments.experiments.runner import run_experiment

if TYPE_CHECKING:
    from pathlib import Path


def test_runner_csv_and_progress(tmp_path: Path) -> None:
    territories = [
        make_territory("AAA:0", -20, current_timezone="UTC+00:00"),
        make_territory("BBB:0", 10, current_timezone="UTC+01:00"),
    ]
    config = ExperimentConfig(utc_offsets=(1, 2), minimum_noon=None, workers=1)
    progress: list[tuple[int, int, str]] = []
    results = run_experiment(config, territories, on_territory_done=lambda *values: progress.append(values))
    assert progress == [(1, 2, "AAA:0"), (2, 2, "BBB:0")]
    path = write_results_csv(results, tmp_path / "results.csv")
    assert path.read_text(encoding="utf-8").splitlines()[0].split(",") == [
        "territory_id",
        "country_code",
        "territory_name",
        "admin_level",
        "current_timezone",
        "proposed_timezone",
        "proposed_utc_offset_hours",
        "timezone_shift_hours",
        "area_weighted_noon_minutes",
        "compliance_rate",
        "population_covered",
        "coverage_type_used",
        "fallback_used",
        "solar_year",
        "noon_year",
        "proposed_earliest_noon_minutes",
        "proposed_latest_noon_minutes",
        "current_timezone_description",
        "current_noon_ranges",
        "median_equation_of_time_minutes",
    ]
    assert run_experiment(config, territories[:1])[0].country_code == "AAA"


def test_runner_rejects_empty_or_duplicate_territories() -> None:
    config = ExperimentConfig(utc_offsets=(1,), minimum_noon=None)
    with pytest.raises(ValueError, match="At least one"):
        run_experiment(config, [])
    territory = make_territory()
    with pytest.raises(ValueError, match="unique"):
        run_experiment(config, [territory, territory])


def test_runner_progress_does_not_log_each_optimization(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    territories = [make_territory("AAA:0"), make_territory("BBB:0")]
    results = run_experiment(ExperimentConfig(utc_offsets=(1,), minimum_noon=None, workers=1), territories)
    assert len(results) == 2
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    assert "0/2" in lines[0]
    assert "2/2" in lines[1]
    assert sum(record.message.startswith("Optimizing ") for record in caplog.records) == 1


def test_optimization_failures_name_the_territory() -> None:
    territories = [
        make_territory("TST:good", samples=(TerritorySample(0.5, 0.5),), geometry=SQUARE),
        make_territory(
            "TST:bad", samples=(TerritorySample(0.5, 0.5),), geometry={"type": "Point", "coordinates": [0, 0]}
        ),
    ]
    with pytest.raises(ValueError, match="Expected Polygon or MultiPolygon") as error:
        run_experiment(ExperimentConfig(), territories)
    assert "While optimizing territory TST:bad (Test)." in describe_error(error.value)
