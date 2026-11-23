"""Run experiments and export their territory assignments."""

from __future__ import annotations

import csv
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields
from datetime import MAXYEAR, MINYEAR
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

from timezone_experiments.config import ExperimentConfig
from timezone_experiments.data.noon import NoonStatistics
from timezone_experiments.optimization.optimizer import AssignmentResult, optimize_territory
from timezone_experiments.progress import Progress

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from timezone_experiments.data.models import Territory

logger = logging.getLogger(__name__)
type ProgressCallback = Callable[[int, int, str], None]
_NOON_COLUMNS = (
    "noon_year",
    "proposed_earliest_noon_minutes",
    "proposed_latest_noon_minutes",
    "current_timezone_description",
    "current_noon_ranges",
    "median_equation_of_time_minutes",
)


def run_experiment(
    experiment: ExperimentConfig | Mapping[str, object],
    territories: Iterable[Territory],
    *,
    output_path: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> list[AssignmentResult]:
    """Optimize all territories, optionally writing a result CSV."""
    config = experiment if isinstance(experiment, ExperimentConfig) else ExperimentConfig.from_mapping(dict(experiment))
    units = list(territories)
    if not units:
        msg = "At least one territory is required."
        raise ValueError(msg)
    identifiers = [territory.territory_id for territory in units]
    if len(set(identifiers)) != len(identifiers):
        msg = "Territory identifiers must be unique within an experiment."
        raise ValueError(msg)
    started = time.perf_counter()
    logger.info(
        "Optimizing %d territories with %d workers (%s coverage, %d candidate offsets).",
        len(units),
        config.workers,
        config.coverage_type,
        len(config.utc_offsets),
    )
    with Progress("Optimizing", len(units), enabled=progress is None, log_interval=None) as display:

        def report_progress(index: int, territory: Territory, _result: AssignmentResult) -> None:
            if progress is not None:
                progress(index, len(units), territory.territory_id)
            else:
                display.update(index, f"finished {territory.territory_id} ({territory.name})")

        results = _optimize_all(units, config, report_progress)
    logger.info("Optimized %d territories in %.2f seconds.", len(results), time.perf_counter() - started)
    if output_path is not None:
        write_results_csv(results, output_path)
    return results


def _optimize_all(
    territories: Sequence[Territory],
    config: ExperimentConfig,
    report_progress: Callable[[int, Territory, AssignmentResult], None],
) -> list[AssignmentResult]:
    if config.workers == 1:
        assignments = (optimize_territory(territory, config) for territory in territories)
        return _collect_assignments(territories, assignments, report_progress)
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(_optimize_task, (territory, config)): index for index, territory in enumerate(territories)
        }
        results: dict[int, AssignmentResult] = {}
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            result = future.result()
            results[index] = result
            report_progress(completed, territories[index], result)
    return [results[index] for index in range(len(territories))]


def _collect_assignments(
    territories: Sequence[Territory],
    assignments: Iterable[AssignmentResult],
    report_progress: Callable[[int, Territory, AssignmentResult], None],
) -> list[AssignmentResult]:
    results: list[AssignmentResult] = []
    for index, (territory, result) in enumerate(zip(territories, assignments, strict=True), start=1):
        results.append(result)
        report_progress(index, territory, result)
    return results


def _optimize_task(task: tuple[Territory, ExperimentConfig]) -> AssignmentResult:
    territory, config = task
    return optimize_territory(territory, config)


def write_results_csv(results: Sequence[AssignmentResult], path: str | Path) -> Path:
    """Write experiment assignments and comparison statistics to CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = result_csv_rows(results)
    logger.info("Writing %d territory assignments to %s.", len(rows), output)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


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
        country=row["country"],
        region=row["region"],
        admin_level=int(row["admin_level"]),
        current_timezone=row["current_timezone"] or None,
        proposed_timezone=row["proposed_timezone"],
        proposed_utc_offset=float(row["proposed_utc_offset"]),
        timezone_shift_hours=float(row["timezone_shift_hours"]) if row["timezone_shift_hours"] else None,
        solar_noon_statistic=float(row["solar_noon_statistic"]),
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
        result.proposed_utc_offset,
        result.timezone_shift_hours,
        result.solar_noon_statistic,
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
            or len(bounds) != 2  # noqa: PLR2004
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
