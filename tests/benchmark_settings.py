"""Benchmark every coverage/solar-metric combination on real data."""

from __future__ import annotations

import argparse
import csv
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter, process_time
from typing import TYPE_CHECKING, get_args

from timezone_experiments.config import CoverageType, load_config
from timezone_experiments.data.boundaries import load_territories
from timezone_experiments.data.population import attach_population_weights, load_population_csv
from timezone_experiments.data.preparation import InputOptions, prepare_boundaries
from timezone_experiments.experiments.runner import run_experiment
from timezone_experiments.optimization.optimizer import optimize_territory
from timezone_experiments.progress import Progress

if TYPE_CHECKING:
    from timezone_experiments.config import ExperimentConfig, SolarMetric
    from timezone_experiments.data.models import Territory

METRICS: tuple[SolarMetric, ...] = ("min", "median", "max")
COVERAGES: tuple[CoverageType, ...] = ("longitude", "area", "population")
if set(COVERAGES) != set(get_args(CoverageType.__value__)):
    msg = "Update the performance matrix to cover every supported coverage type."
    raise RuntimeError(msg)


def _measure_territory(
    units: tuple[Territory, Territory], cases: list[ExperimentConfig], repeats: int
) -> list[list[float]]:
    measurements = [[0.0] * repeats for _ in cases]
    rotation = sum(map(ord, units[0].territory_id)) % len(cases)
    for position in range(len(cases)):
        index = (position + rotation) % len(cases)
        config = cases[index]
        needs_population = config.coverage_type == "population"
        territory = units[1] if needs_population else units[0]
        result = optimize_territory(territory, config)
        iterations = 1
        duration = 0.0
        # Batch short calls above Windows' process CPU-clock accounting granularity.
        while duration < 0.0625:
            start = process_time()
            for _ in range(iterations):
                optimize_territory(territory, config)
            duration = process_time() - start
            if duration < 0.0625:
                iterations *= 2
        for repeat in range(repeats):
            start = process_time()
            for _ in range(iterations):
                result = optimize_territory(territory, config)
            measurements[index][repeat] = (process_time() - start) / iterations
            if result.coverage_type_used != config.coverage_type:
                msg = f"Benchmark unexpectedly used fallback coverage for {territory.territory_id}."
                raise RuntimeError(msg)
    return measurements


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-directory", type=Path, default=Path("data/downloads"))
    parser.add_argument("--output", type=Path, default=Path("outputs/csv/settings-performance.csv"))
    parser.add_argument("--territories", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples-per-territory", type=int, default=1024, help="Matrix-only sample cap; 0 keeps all.")
    parser.add_argument("--world-check", action="store_true")
    args = parser.parse_args()
    if args.territories < 1 or args.repeats < 1 or args.samples_per_territory < 0:
        parser.error("territories/repeats must be positive and the sample cap must be non-negative")
    return args


def main() -> None:
    """Write CPU-time matrix measurements and optional full-world wall timings."""
    args = _arguments()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    rows: list[dict[str, object]] = []

    start = perf_counter()
    boundaries = prepare_boundaries(config, InputOptions(data_directory=args.data_directory))
    world = load_territories(boundaries, config)
    rows.append({"stage": "boundary_sampling_wall", "territories": len(world), "seconds": perf_counter() - start})
    ordered = sorted(world, key=lambda unit: (len(unit.samples), unit.territory_id))
    count = min(args.territories, len(ordered))
    selected = [ordered[round(index * (len(ordered) - 1) / max(count - 1, 1))] for index in range(count)]
    if args.samples_per_territory:
        selected = [
            replace(
                unit,
                samples=tuple(
                    unit.samples[round(index * (len(unit.samples) - 1) / max(args.samples_per_territory - 1, 1))]
                    for index in range(args.samples_per_territory)
                ),
            )
            if len(unit.samples) > args.samples_per_territory
            else unit
            for unit in selected
        ]
    start = perf_counter()
    weighted = attach_population_weights(
        selected, load_population_csv(args.data_directory / "population_2025.csv"), workers=config.workers
    )
    rows.append({"stage": "population_overlay_wall", "territories": len(selected), "seconds": perf_counter() - start})
    pairs = [
        (original, populated)
        for original, populated in zip(selected, weighted, strict=True)
        if any((sample.population_weight or 0) > 0 for sample in populated.samples)
    ]
    if not pairs:
        msg = "No selected territories contain population cells; supply a finer real population grid."
        raise RuntimeError(msg)
    print(f"Matrix: {len(pairs)}/{len(selected)} territories have real population weights; no fallback is timed.")
    cases = [
        replace(
            config,
            coverage_type=coverage,
            solar_metric=metric,
            minimum_noon=None,
            maximum_noon=None,
            workers=1,
        )
        for coverage in COVERAGES
        for metric in METRICS
    ]
    totals = [[0.0] * args.repeats for _ in cases]
    start = perf_counter()
    with (
        ProcessPoolExecutor(max_workers=min(config.workers, len(pairs))) as executor,
        Progress("Benchmarking settings", len(pairs)) as progress,
    ):
        futures = {executor.submit(_measure_territory, pair, cases, args.repeats): pair[0].name for pair in pairs}
        for completed, future in enumerate(as_completed(futures), start=1):
            for total, measured in zip(totals, future.result(), strict=True):
                for repeat, duration in enumerate(measured):
                    total[repeat] += duration
            progress.update(completed, futures[future])
    rows.append({"stage": "matrix_wall", "territories": len(pairs), "seconds": perf_counter() - start})
    baselines = {
        case.coverage_type: median(total)
        for case, total in zip(cases, totals, strict=True)
        if case.solar_metric == "min"
    }
    for case, total in zip(cases, totals, strict=True):
        needs_population = case.coverage_type == "population"
        rows.append(
            {
                "stage": "matrix_cpu",
                "coverage": case.coverage_type,
                "metric": str(case.solar_metric),
                "territories": len(pairs),
                "samples": sum(len(pair[int(needs_population)].samples) for pair in pairs),
                "workers": min(config.workers, len(pairs)),
                "repeats": args.repeats,
                "seconds": median(total),
                "relative_to_min": median(total) / baselines[case.coverage_type],
                "population_fallbacks": 0,
            }
        )
    if args.world_check:
        checks = [replace(config, coverage_type="longitude", solar_metric=metric) for metric in METRICS]
        for check in checks:
            start = perf_counter()
            results = run_experiment(check, world)
            rows.append(
                {
                    "stage": "world_wall",
                    "coverage": check.coverage_type,
                    "metric": str(check.solar_metric),
                    "territories": len(results),
                    "samples": sum(len(unit.samples) for unit in world),
                    "workers": config.workers,
                    "repeats": 1,
                    "seconds": perf_counter() - start,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(cases)} matrix cases and stage timings to {args.output}.")


if __name__ == "__main__":
    main()
