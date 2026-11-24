"""Interactive app launched by ``timezone_experiments dashboard``."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import streamlit as st

from timezone_experiments.cli import build_parser, input_options
from timezone_experiments.config import (
    COVERAGE_TYPES,
    SOLAR_METRICS,
    ConfigError,
    ExperimentConfig,
    load_config,
    parse_countries,
    parse_country_levels,
)
from timezone_experiments.constants import (
    EVENT_MAP_CELL_DEGREES,
    MAP_SIMPLIFICATION_DEGREES,
    MAX_SOLAR_YEAR,
    MIN_SOLAR_YEAR,
    POPULATION_CELL_DEGREES,
)
from timezone_experiments.data.paths import cache_directory
from timezone_experiments.data.preparation import InputOptions, load_experiment_territories
from timezone_experiments.errors import USER_ERRORS, describe_error
from timezone_experiments.experiments.runner import result_csv_rows, run_experiment
from timezone_experiments.visualization.maps import MAP_FIELDS, MapOptions, build_plotly_figure, map_title

if TYPE_CHECKING:
    from types import ModuleType

    from timezone_experiments.config import CoverageType, SolarMetric
    from timezone_experiments.data.models import Territory
    from timezone_experiments.optimization.optimizer import AssignmentResult

# Sessions share the download and cache directories, so only one experiment prepares inputs at a time.
_RUN_LOCK = threading.Lock()
_LAST_RUN = "last_experiment_run"


class UploadedFile(Protocol):
    """Small subset of Streamlit's uploaded file interface."""

    name: str

    def getvalue(self) -> bytes:
        """Return the uploaded file contents."""
        ...


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """The latest run, kept in session state so widget changes (e.g. the map field) do not discard it."""

    config: ExperimentConfig
    territories: list[Territory]
    results: list[AssignmentResult]


# Configuration fields without a dashboard control, because they have no effect on what the dashboard shows.
NOT_IN_DASHBOARD = frozenset({"rasterize_static_map_fills"})  # Only static (PNG/SVG) maps are rasterized.


@dataclass(frozen=True, slots=True)
class DashboardInputs:
    """Values collected from the dashboard controls.

    Fields after the uploads are named exactly like ``ExperimentConfig`` fields, so every configuration setting
    has a control (except ``NOT_IN_DASHBOARD``); a test keeps the two in sync.
    """

    population_upload: UploadedFile | None
    timezone_upload: UploadedFile | None
    map_field: str
    minimum_noon: str
    maximum_noon: str
    solar_metric: SolarMetric
    year: int
    coverage_type: CoverageType
    coverage_threshold: float
    utc_offsets: str
    countries: str
    default_admin_level: int
    subdivision_rules: str
    merged_subdivision_levels: str
    include_antarctica: bool
    sample_grid_size: int
    population_cell_degrees: float
    workers: int
    event_map_cell_degrees: float
    map_geometry_simplification_degrees: float


def main(argv: list[str] | None = None) -> None:
    """Render the interactive experiment dashboard."""
    st.set_page_config(page_title="Timezone Experiments", layout="wide")
    st.title("Hypothetical timezone experiments")
    st.caption("Explore permanent UTC offsets using configurable solar-noon and coverage rules.")
    args = build_parser().parse_args(["dashboard", *(argv or [])])
    try:
        config = load_config(args.config)
        st.info("Required data is downloaded automatically and cached. Population and timezone uploads are optional.")
        inputs = _dashboard_inputs(st, config)
        if st.button("Run experiment", type="primary"):
            _execute_experiment(st, inputs, config, input_options(args))
        run = st.session_state.get(_LAST_RUN)
        if run is not None:
            _show_results(st, run, inputs.map_field, cache_directory(args.data_directory))
    except USER_ERRORS as error:
        st.error(describe_error(error))


def _dashboard_inputs(st: ModuleType, config: ExperimentConfig) -> DashboardInputs:
    values = _upload_inputs(st)
    noon, coverage, territories = st.columns(3)
    with noon:
        values |= _noon_inputs(st, config)
    with coverage:
        values |= _coverage_inputs(st, config)
    with territories:
        values |= _territory_inputs(st, config)
    values |= _map_inputs(st, config)
    return DashboardInputs(**values)


def _upload_inputs(st: ModuleType) -> dict[str, Any]:
    with st.expander(
        "Optional: use your own population or timezone files (normally downloaded automatically)", expanded=False
    ):
        population_upload = st.file_uploader(
            "Optional population grid (CSV)",
            type=("csv",),
            help="Columns latitude, longitude, population; replaces the GHSL download (--population).",
        )
        timezone_upload = st.file_uploader(
            "Optional current timezone boundaries (GeoJSON)",
            type=("json", "geojson"),
            help="Replaces the timezone-boundary-builder download (--timezones).",
        )
    return {
        "population_upload": population_upload,
        "timezone_upload": timezone_upload,
    }


def _territory_inputs(st: ModuleType, config: ExperimentConfig) -> dict[str, Any]:
    st.subheader("Territories")
    countries = st.text_input(
        "Countries (comma-separated codes, blank for all)",
        value=", ".join(config.countries or ()),
        help="countries: three-letter ISO codes such as USA, MEX; only these are downloaded and optimized.",
    )
    default_admin_level = st.number_input(
        "Default administrative level",
        min_value=0,
        value=config.default_admin_level,
        step=1,
        help=(
            "default_admin_level: level for countries without a rule. 0 = countries, 1 = states/provinces, "
            "2 = districts."
        ),
    )
    subdivision_rules = st.text_area(
        "Subdivision rules (JSON: country code to level)",
        value=json.dumps(config.subdivision_rules),
        help=(
            "subdivision_rules: countries split into subdivisions, each optimized separately. "
            "Defaults to the major multi-zone countries at level 1; an explicit mapping replaces them."
        ),
    )
    merged_subdivision_levels = st.text_area(
        "Merged subdivision levels (JSON: country code to level)",
        value=json.dumps(config.merged_subdivision_levels),
        help=(
            "merged_subdivision_levels: load these countries at a finer level but optimize them as one "
            "territory, so absent overseas areas are left out. Defaults to FRA and NLD at level 1; {} disables."
        ),
    )
    include_antarctica = st.checkbox(
        "Include Antarctica",
        value=config.include_antarctica,
        help="include_antarctica: usually needs relaxed noon bounds, since polar longitudes span every offset.",
    )
    sample_grid_size = st.number_input(
        "Sample grid size",
        min_value=1,
        value=config.sample_grid_size,
        step=1,
        help="sample_grid_size: N x N sample points per polygon; larger is more precise and slower.",
    )
    return {
        "countries": countries,
        "default_admin_level": int(default_admin_level),
        "subdivision_rules": subdivision_rules,
        "merged_subdivision_levels": merged_subdivision_levels,
        "include_antarctica": bool(include_antarctica),
        "sample_grid_size": int(sample_grid_size),
    }


def _noon_inputs(st: ModuleType, config: ExperimentConfig) -> dict[str, Any]:
    st.subheader("Solar-noon rule")
    minimum_noon = st.text_input(
        "Minimum solar noon (HH:MM, blank for none)",
        value=config.minimum_noon or "",
        help="minimum_noon: earliest acceptable solar noon; the earliest offset meeting it wins.",
    )
    maximum_noon = st.text_input(
        "Maximum solar noon (HH:MM, blank for none)",
        value=config.maximum_noon or "",
        help="maximum_noon: latest acceptable solar noon.",
    )
    solar_metric = st.selectbox(
        "Solar-noon statistic",
        SOLAR_METRICS,
        index=SOLAR_METRICS.index(config.solar_metric),
        help="solar_metric: which annual solar-noon value (min, max, or median) must meet the bounds.",
    )
    year = st.number_input(
        "Solar year",
        min_value=MIN_SOLAR_YEAR,
        max_value=MAX_SOLAR_YEAR,
        value=config.year,
        step=1,
        help="year: calendar year for the equation of time and current timezone rules.",
    )
    utc_offsets = st.text_input(
        "Candidate UTC offsets (hours, comma-separated)",
        value=", ".join(f"{offset:g}" for offset in config.utc_offsets),
        help="utc_offsets: the offsets the optimizer may choose from, e.g. -12, ..., 14 or 5.5, 5.75.",
    )
    return {
        "minimum_noon": minimum_noon,
        "maximum_noon": maximum_noon,
        "solar_metric": solar_metric,
        "year": int(year),
        "utc_offsets": utc_offsets,
    }


def _coverage_inputs(st: ModuleType, config: ExperimentConfig) -> dict[str, Any]:
    st.subheader("Coverage")
    coverage_type = st.selectbox(
        "Coverage rule",
        COVERAGE_TYPES,
        index=COVERAGE_TYPES.index(config.coverage_type),
        help="coverage_type: weight samples by longitude span, area, or population when checking coverage.",
    )
    coverage_threshold = st.slider(
        "Required coverage",
        min_value=0.0,
        max_value=1.0,
        value=config.coverage_threshold,
        step=0.01,
        help="coverage_threshold: share of the weighted samples that must meet the noon bounds.",
    )
    population_cell_degrees = st.selectbox(
        "Population grid cell size (degrees)",
        POPULATION_CELL_DEGREES,
        index=POPULATION_CELL_DEGREES.index(config.population_cell_degrees),
        help="population_cell_degrees: resolution the GHSL raster is aggregated to for population coverage.",
    )
    return {
        "coverage_type": coverage_type,
        "coverage_threshold": float(coverage_threshold),
        "population_cell_degrees": float(population_cell_degrees),
    }


def _map_inputs(st: ModuleType, config: ExperimentConfig) -> dict[str, Any]:
    st.subheader("Map and performance")
    map_column, cells, simplification, workers_column = st.columns(4)
    with map_column:
        map_field = st.selectbox(
            "Map statistic", MAP_FIELDS, format_func=map_title, help="Which result the map below shows."
        )
    with cells:
        event_map_cell_degrees = st.selectbox(
            "Sunrise/sunset cell size (degrees)",
            EVENT_MAP_CELL_DEGREES,
            index=EVENT_MAP_CELL_DEGREES.index(config.event_map_cell_degrees),
            help="event_map_cell_degrees: grid cell size of the sunrise/sunset maps; smaller is finer and slower.",
        )
    with simplification:
        map_geometry_simplification_degrees = st.selectbox(
            "Gradient map simplification (degrees)",
            MAP_SIMPLIFICATION_DEGREES,
            index=MAP_SIMPLIFICATION_DEGREES.index(config.map_geometry_simplification_degrees),
            help="map_geometry_simplification_degrees: outline simplification for gradient maps; 0 keeps exact.",
        )
    with workers_column:
        workers = st.number_input(
            "Workers",
            min_value=1,
            value=config.workers,
            step=1,
            help="workers: parallel threads/processes for sampling, overlays, and maps.",
        )
    return {
        "map_field": map_field,
        "event_map_cell_degrees": float(event_map_cell_degrees),
        "map_geometry_simplification_degrees": float(map_geometry_simplification_degrees),
        "workers": int(workers),
    }


def _execute_experiment(
    st: ModuleType,
    inputs: DashboardInputs,
    base_config: ExperimentConfig,
    options: InputOptions,
) -> None:
    if not _RUN_LOCK.acquire(blocking=False):
        st.warning("Another experiment is running on this server; try again when it finishes.")
        return
    try:
        config = _build_config(inputs, base_config)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            options = replace(
                options,
                population_path=_uploaded_path(
                    inputs.population_upload, directory / "population.csv", options.population_path
                ),
                timezones_path=_uploaded_path(
                    inputs.timezone_upload, directory / "timezones.geojson", options.timezones_path
                ),
            )
            st.info("Preparing input data. Initial downloads may take several minutes.")
            territories = load_experiment_territories(config, options)
            progress = st.progress(0, text="Optimizing territories...")

            def update_progress(index: int, total: int, territory_id: str) -> None:
                progress.progress(index / total, text=f"Optimizing {territory_id} ({index}/{total})")

            results = run_experiment(config, territories, on_territory_done=update_progress)
        st.session_state[_LAST_RUN] = ExperimentRun(config, territories, results)
        st.success(f"Optimized {len(results)} territories.")
    except USER_ERRORS as error:
        st.error(describe_error(error))
    finally:
        _RUN_LOCK.release()


def _show_results(st: ModuleType, run: ExperimentRun, map_field: str, cache_directory: Path) -> None:
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
        data=_csv_text(run.results),
        file_name="timezone_experiment.csv",
        mime="text/csv",
    )


def _uploaded_path(upload: UploadedFile | None, path: Path, fallback: Path | None) -> Path | None:
    if upload is None:
        return fallback
    path.write_bytes(upload.getvalue())
    return path


def _build_config(inputs: DashboardInputs, base: ExperimentConfig) -> ExperimentConfig:
    countries = [part for part in inputs.countries.replace(";", ",").split(",") if part.strip()]
    return replace(
        base,
        minimum_noon=inputs.minimum_noon.strip() or None,
        maximum_noon=inputs.maximum_noon.strip() or None,
        solar_metric=inputs.solar_metric,
        year=inputs.year,
        coverage_type=inputs.coverage_type,
        coverage_threshold=inputs.coverage_threshold,
        utc_offsets=_parse_offsets(inputs.utc_offsets),
        countries=parse_countries(countries, "countries") if countries else None,
        default_admin_level=inputs.default_admin_level,
        subdivision_rules=parse_country_levels(_json(inputs.subdivision_rules), "subdivision_rules"),
        merged_subdivision_levels=parse_country_levels(
            _json(inputs.merged_subdivision_levels), "merged_subdivision_levels", minimum=1
        ),
        include_antarctica=inputs.include_antarctica,
        sample_grid_size=inputs.sample_grid_size,
        population_cell_degrees=inputs.population_cell_degrees,
        workers=inputs.workers,
        event_map_cell_degrees=inputs.event_map_cell_degrees,
        map_geometry_simplification_degrees=inputs.map_geometry_simplification_degrees,
    )


def _json(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        msg = f'Expected a JSON object such as {{"USA": 1}}: {error}'
        raise ConfigError(msg) from error


def _parse_offsets(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(part) for part in value.replace(";", ",").split(",") if part.strip())
    except ValueError as error:
        msg = f"Candidate UTC offsets must be comma-separated numbers: {value!r}."
        raise ConfigError(msg) from error


def _result_row(result: AssignmentResult) -> dict[str, object]:
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


def _csv_text(results: list[AssignmentResult]) -> str:
    output = io.StringIO(newline="")
    rows = result_csv_rows(results)
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


if __name__ == "__main__":
    main(sys.argv[1:])
