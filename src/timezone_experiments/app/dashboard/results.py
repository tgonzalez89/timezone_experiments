"""Showing a finished dashboard run: the results table, the selected map, and the CSV download."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from timezone_experiments.app.errors import USER_ERRORS, describe_error
from timezone_experiments.experiments.results_io import results_csv_text
from timezone_experiments.maps import MapOptions, build_plotly_figure

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from timezone_experiments.core.config import ExperimentConfig
    from timezone_experiments.core.models import Territory
    from timezone_experiments.optimization.optimizer import AssignmentResult


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """The latest run, kept in session state so widget changes (e.g. the map field) do not discard it."""

    config: ExperimentConfig
    territories: list[Territory]
    results: list[AssignmentResult]


def show_results(st: ModuleType, run: ExperimentRun, map_field: str, cache_directory: Path) -> None:
    """Show the results table, the map of the selected statistic, and the CSV download."""
    try:
        figure = build_plotly_figure(
            run.territories,
            run.results,
            map_field,
            MapOptions.from_config(run.config, cache_directory=cache_directory),
        )
    except USER_ERRORS as error:
        st.error(describe_error(error))
        return
    st.dataframe([_result_row(result) for result in run.results], width="stretch")
    st.plotly_chart(figure, width="stretch")
    st.download_button(
        "Download results CSV",
        data=results_csv_text(run.results),
        file_name="timezone_experiment.csv",
        mime="text/csv",
    )


def _result_row(result: AssignmentResult) -> dict[str, object]:
    """Return the columns of the on-screen table (the CSV download has every column)."""
    return {
        "country_code": result.country_code,
        "territory_name": result.territory_name,
        "current_timezone": result.current_timezone,
        "proposed_timezone": result.proposed_timezone,
        "timezone_shift_hours": result.timezone_shift_hours,
        "area_weighted_noon_minutes": result.area_weighted_noon_minutes,
        "compliance_rate": result.compliance_rate,
        "population_covered": result.population_covered,
        "coverage_type_used": result.coverage_type_used,
        "fallback_used": result.fallback_used,
    }
