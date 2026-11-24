"""Result CSV files: round trips and validation."""

from __future__ import annotations

import csv
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from factories import (
    make_territory,
)

from timezone_experiments.core.config import (
    ExperimentConfig,
)
from timezone_experiments.experiments.results_io import read_results_csv, results_csv_text, write_results_csv
from timezone_experiments.optimization.assessment import (
    calculate_noon_statistics,
)
from timezone_experiments.optimization.optimizer import optimize_territory

if TYPE_CHECKING:
    from pathlib import Path


def test_saved_result_csv_round_trip(tmp_path: Path) -> None:
    assignment = optimize_territory(
        make_territory(current_timezone="America/New_York"), ExperimentConfig(minimum_noon=None, utc_offsets=(1,))
    )
    results = [
        replace(assignment, proposed_utc_offset_hours=5.75, timezone_shift_hours=1.25, population_covered=0),
        replace(
            assignment,
            territory_id="TST:2",
            territory_name="North, coast",
            current_timezone=None,
            timezone_shift_hours=None,
            noon=calculate_noon_statistics(make_territory(current_timezone=None), 1, year=2024),
        ),
    ]
    path = write_results_csv(results, tmp_path / "results.csv")
    assert read_results_csv(path) == results
    path.write_text(results_csv_text(results), encoding="utf-8")
    assert read_results_csv(path) == results


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("columns", "missing columns"),
        ("empty", "no territory assignments"),
        ("duplicate", "line 3.*unique"),
        ("number", "line 2.*could not convert"),
        ("nonfinite", "line 2.*must be finite"),
        ("short", "line 2.*Incorrect number"),
        ("noon_year", "line 2.*valid year"),
        ("noon_nonfinite", "line 2.*finite and ordered"),
        ("noon_json", "line 2.*JSON object"),
        ("noon_bounds", "line 2.*minute pairs"),
        ("noon_missing_current", "line 2.*exactly when"),
        ("noon_median", "line 2.*Median equation of time must be finite"),
    ],
)
def test_saved_result_csv_rejects_invalid_data(tmp_path: Path, problem: str, message: str) -> None:
    assignment = optimize_territory(make_territory(), ExperimentConfig(minimum_noon=None, utc_offsets=(1,)))
    path = write_results_csv([assignment], tmp_path / "results.csv")
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    if problem == "columns":
        rows = [["territory_id"], ["TST:1"]]
    elif problem == "empty":
        rows = rows[:1]
    elif problem == "duplicate":
        rows.append(rows[1])
    elif problem == "short":
        rows[1].pop()
    elif problem.startswith("noon_"):
        column, value = {
            "noon_year": ("noon_year", "0"),
            "noon_nonfinite": ("proposed_earliest_noon_minutes", "nan"),
            "noon_json": ("current_noon_ranges", "[]"),
            "noon_bounds": ("current_noon_ranges", '{"Standard time": [720, 700]}'),
            "noon_missing_current": ("current_noon_ranges", "{}"),
            "noon_median": ("median_equation_of_time_minutes", "inf"),
        }[problem]
        rows[1][rows[0].index(column)] = value
    else:
        rows[1][rows[0].index("proposed_utc_offset_hours")] = "nan" if problem == "nonfinite" else "invalid"
    with path.open("w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(rows)
    with pytest.raises(ValueError, match=message):
        read_results_csv(path)
