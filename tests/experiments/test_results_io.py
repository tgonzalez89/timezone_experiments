"""Result CSV files: round trips and validation."""

from __future__ import annotations

import csv
import platform
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from factories import (
    make_territory,
)

from timezone_experiments.core.config import (
    ExperimentConfig,
)
from timezone_experiments.core.fingerprint import file_digest
from timezone_experiments.core.timezones import tz_database_version
from timezone_experiments.data.download.catalog import SOURCES, write_source_metadata
from timezone_experiments.data.preparation import PreparedInputs
from timezone_experiments.experiments.results_io import (
    input_records,
    provenance_credits,
    read_results_csv,
    read_results_provenance,
    results_csv_text,
    results_provenance_path,
    write_results_csv,
    write_results_provenance,
)
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


def test_provenance_records_inputs_software_and_credits(tmp_path: Path) -> None:
    boundaries = tmp_path / "boundaries.geojson"
    timezones = tmp_path / "uploaded_timezones.geojson"
    for path in (boundaries, timezones):
        path.write_text("{}", encoding="utf-8")
    write_source_metadata(boundaries, SOURCES["natural_earth"], "https://example.org/world")
    records = input_records(PreparedInputs(boundaries=boundaries, timezones=timezones))
    results = tmp_path / "results.csv"
    path = write_results_provenance(results, records)
    assert path == results_provenance_path(results) == tmp_path / "results.provenance.json"
    provenance = read_results_provenance(results)
    assert provenance is not None
    assert provenance["inputs"]["boundaries"]["metadata"]["source"] == "natural_earth"
    assert provenance["inputs"]["timezones"] == {
        "file": "uploaded_timezones.geojson",
        "sha256": file_digest(timezones),
        "metadata": None,
    }
    assert provenance["inputs"]["population"] is None
    assert provenance["tz_database"] == tz_database_version()
    assert provenance["python"] == platform.python_version()
    assert set(provenance["packages"]) >= {"numpy", "shapely", "tzdata"}
    assert provenance_credits(provenance) == f"Data: Natural Earth {SOURCES['natural_earth'].version}."
    assert provenance_credits({"inputs": "damaged"}) == ""


@pytest.mark.parametrize("content", [None, "{damaged", "[]"], ids=["missing", "corrupt", "not-an-object"])
def test_missing_or_unusable_provenance_reads_as_none(tmp_path: Path, content: str | None) -> None:
    results = tmp_path / "results.csv"
    if content is not None:
        results_provenance_path(results).write_text(content, encoding="utf-8")
    assert read_results_provenance(results) is None
