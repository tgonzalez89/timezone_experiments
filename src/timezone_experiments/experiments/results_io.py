"""Result CSV and saved-configuration files: writing, reading, and validating them."""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import asdict, fields
from datetime import MAXYEAR, MINYEAR
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

from timezone_experiments.core.config import ConfigError, ExperimentConfig
from timezone_experiments.core.files import read_json_dataset, write_bytes_atomic, write_text_atomic
from timezone_experiments.optimization.assessment import NoonStatistics
from timezone_experiments.optimization.optimizer import AssignmentResult

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)
_NOON_COLUMNS = (
    "noon_year",
    "proposed_earliest_noon_minutes",
    "proposed_latest_noon_minutes",
    "current_timezone_description",
    "current_noon_ranges",
    "median_equation_of_time_minutes",
)
_NOON_RANGE_LENGTH = 2


def write_results_csv(results: Sequence[AssignmentResult], path: str | Path) -> Path:
    """Write experiment assignments and comparison statistics to CSV."""
    output = Path(path)
    logger.info("Writing %d territory assignments to %s.", len(results), output)
    write_bytes_atomic(output, results_csv_text(results).encode())
    return output


def results_csv_text(results: Sequence[AssignmentResult]) -> str:
    """Return the result CSV as text, identical for CLI files and dashboard downloads."""
    rows = result_csv_rows(results)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def results_config_path(results_path: str | Path) -> Path:
    """Return where the configuration of a result CSV is saved: ``results.csv`` -> ``results.config.json``."""
    return Path(results_path).with_suffix(".config.json")


def write_results_config(config: ExperimentConfig, results_path: str | Path) -> Path:
    """Save the configuration that produced a result CSV beside it, so maps can reuse and check it."""
    path = results_config_path(results_path)
    write_text_atomic(path, json.dumps(config.to_mapping(), indent=2) + "\n")
    return path


def read_results_config(results_path: str | Path) -> ExperimentConfig | None:
    """Return the configuration saved beside a result CSV, or None when there is none (e.g. dashboard downloads)."""
    path = results_config_path(results_path)
    if not path.is_file():
        return None
    values = read_json_dataset(path)
    if not isinstance(values, dict):
        msg = f"Saved configuration {path} must be a JSON object."
        raise ConfigError(msg)
    try:
        return ExperimentConfig.from_mapping(values)
    except ConfigError as error:
        msg = f"Invalid saved configuration {path}: {error}"
        raise ConfigError(msg) from error


def result_csv_rows(results: Sequence[AssignmentResult]) -> list[dict[str, Any]]:
    """Serialize assignments identically for CLI files and dashboard downloads."""
    rows = [asdict(result) for result in results]
    for row in rows:
        noon = row.pop("noon")
        row.update(
            zip(
                _NOON_COLUMNS,
                (
                    noon["year"],
                    noon["proposed_earliest_minutes"],
                    noon["proposed_latest_minutes"],
                    noon["current_timezone_description"],
                    json.dumps(noon["current_ranges"], allow_nan=False),
                    noon["median_equation_of_time_minutes"],
                ),
                strict=True,
            )
        )
    if not rows:
        msg = "Cannot write a result CSV without any assignments."
        raise ValueError(msg)
    return rows


def read_results_csv(path: str | Path) -> list[AssignmentResult]:
    """Read saved assignments, preserving optional comparison and population values."""
    source = Path(path)
    logger.info("Reading saved territory assignments: %s", source)
    results: list[AssignmentResult] = []
    identifiers: set[str] = set()
    with source.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {field.name for field in fields(AssignmentResult) if field.name != "noon"} | set(_NOON_COLUMNS)
        missing = required - set(reader.fieldnames or ())
        if missing:
            msg = (
                f"Result CSV {source} is missing columns: {', '.join(sorted(missing))}. "
                "Regenerate results with the run command to use the current CSV schema."
            )
            raise ValueError(msg)
        for row in reader:
            try:
                result = _read_result_row(row, identifiers)
            except (TypeError, ValueError) as error:
                msg = f"Invalid result CSV {source} at line {reader.line_num}: {error}"
                raise ValueError(msg) from error
            identifiers.add(result.territory_id)
            results.append(result)
    if not results:
        msg = f"Result CSV {source} contains no territory assignments."
        raise ValueError(msg)
    logger.info("Loaded %d saved territory assignments.", len(results))
    return results


def _read_result_row(row: dict[str, str], identifiers: set[str]) -> AssignmentResult:
    if None in row or any(value is None for value in row.values()):
        msg = "Incorrect number of CSV values."
        raise ValueError(msg)
    result = AssignmentResult(
        territory_id=row["territory_id"],
        country_code=row["country_code"],
        territory_name=row["territory_name"],
        admin_level=int(row["admin_level"]),
        current_timezone=row["current_timezone"] or None,
        proposed_timezone=row["proposed_timezone"],
        proposed_utc_offset_hours=float(row["proposed_utc_offset_hours"]),
        timezone_shift_hours=float(row["timezone_shift_hours"]) if row["timezone_shift_hours"] else None,
        area_weighted_noon_minutes=float(row["area_weighted_noon_minutes"]),
        compliance_rate=float(row["compliance_rate"]),
        population_covered=float(row["population_covered"]) if row["population_covered"] else None,
        coverage_type_used=row["coverage_type_used"],
        fallback_used=_read_boolean(row["fallback_used"], "fallback_used"),
        solar_year=int(row["solar_year"]),
        noon=_read_noon_statistics(row),
    )
    if not MINYEAR <= result.solar_year <= MAXYEAR:
        msg = "Solar year must be a valid calendar year."
        raise ValueError(msg)
    numeric = (
        result.proposed_utc_offset_hours,
        result.timezone_shift_hours,
        result.area_weighted_noon_minutes,
        result.compliance_rate,
        result.population_covered,
    )
    if any(value is not None and not isfinite(value) for value in numeric):
        msg = "Numeric result values must be finite."
        raise ValueError(msg)
    if not result.territory_id.strip() or result.territory_id in identifiers:
        msg = f"Territory identifiers must be nonempty and unique: {result.territory_id!r}."
        raise ValueError(msg)
    return result


def _read_boolean(value: str, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    msg = f"{field} must be True or False."
    raise ValueError(msg)


def _read_noon_statistics(row: dict[str, str]) -> NoonStatistics:
    year = int(row["noon_year"])
    earliest = float(row["proposed_earliest_noon_minutes"])
    latest = float(row["proposed_latest_noon_minutes"])
    median_equation = float(row["median_equation_of_time_minutes"])
    description = row["current_timezone_description"]
    if not MINYEAR <= year <= MAXYEAR or not description.strip():
        msg = "Noon statistics require a valid year and current timezone description."
        raise ValueError(msg)
    if not isfinite(earliest) or not isfinite(latest) or earliest > latest:
        msg = "Proposed noon ranges must be finite and ordered earliest to latest."
        raise ValueError(msg)
    if not isfinite(median_equation):
        msg = "Median equation of time must be finite."
        raise ValueError(msg)
    data = json.loads(row["current_noon_ranges"])
    if not isinstance(data, dict):
        msg = "current_noon_ranges must be a JSON object."
        raise TypeError(msg)
    ranges: dict[str, tuple[float, float]] = {}
    for label, bounds in data.items():
        if (
            not label.strip()
            or not isinstance(bounds, list)
            or len(bounds) != _NOON_RANGE_LENGTH
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
                for value in bounds
            )
            or bounds[0] > bounds[1]
        ):
            msg = "Current noon ranges require labels and finite [earliest, latest] minute pairs."
            raise ValueError(msg)
        ranges[label] = (float(bounds[0]), float(bounds[1]))
    if bool(row["current_timezone"]) != bool(ranges):
        msg = "Current noon ranges must be present exactly when a current timezone is available."
        raise ValueError(msg)
    return NoonStatistics(year, earliest, latest, description, ranges, median_equation)
