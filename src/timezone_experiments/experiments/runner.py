"""Run experiments and export their territory assignments."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from timezone_experiments.core.config import ExperimentConfig
from timezone_experiments.core.progress import Progress
from timezone_experiments.experiments.results_io import write_results_config, write_results_csv
from timezone_experiments.optimization.optimizer import AssignmentResult, optimize_territory

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from timezone_experiments.core.models import Territory

logger = logging.getLogger(__name__)
# Called after each territory with (completed count, total, territory_id) instead of the console progress bar.
type TerritoryDoneCallback = Callable[[int, int, str], None]


def run_experiment(
    experiment: ExperimentConfig | Mapping[str, object],
    territories: Iterable[Territory],
    *,
    output_path: str | Path | None = None,
    on_territory_done: TerritoryDoneCallback | None = None,
) -> list[AssignmentResult]:
    """Optimize all territories, optionally writing a result CSV and its configuration (see results_config_path)."""
    config = experiment if isinstance(experiment, ExperimentConfig) else ExperimentConfig.from_mapping(dict(experiment))
    territory_list = list(territories)
    if not territory_list:
        msg = "At least one territory is required."
        raise ValueError(msg)
    identifiers = [territory.territory_id for territory in territory_list]
    if len(set(identifiers)) != len(identifiers):
        msg = "Territory identifiers must be unique within an experiment."
        raise ValueError(msg)
    started = time.perf_counter()
    logger.info(
        "Optimizing %d territories (%s coverage, %d candidate offsets).",
        len(territory_list),
        config.coverage_type,
        len(config.utc_offsets),
    )
    # The optimizer is vectorized; a process pool spends more time pickling territories than it saves.
    results: list[AssignmentResult] = []
    with Progress("Optimizing", len(territory_list), enabled=on_territory_done is None) as display:
        for index, territory in enumerate(territory_list, start=1):
            results.append(_optimize(territory, config))
            if on_territory_done is not None:
                on_territory_done(index, len(territory_list), territory.territory_id)
            else:
                display.update(index, f"finished {territory.label}")
    logger.info("Optimized %d territories in %.2f seconds.", len(results), time.perf_counter() - started)
    if output_path is not None:
        write_results_csv(results, output_path)
        write_results_config(config, output_path)
    return results


def _optimize(territory: Territory, config: ExperimentConfig) -> AssignmentResult:
    try:
        return optimize_territory(territory, config)
    except Exception as error:
        error.add_note(f"While optimizing territory {territory.label}.")
        raise
