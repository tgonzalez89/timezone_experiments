"""Interactive app launched by ``timezone_experiments dashboard``."""

from __future__ import annotations

import csv
import importlib
import io
import json
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast
from zipfile import BadZipFile

from timezone_experiments.cli import build_parser, input_options
from timezone_experiments.config import ExperimentConfig, load_config
from timezone_experiments.data.preparation import InputOptions, load_experiment_territories
from timezone_experiments.experiments.runner import result_csv_rows, run_experiment
from timezone_experiments.visualization.maps import MAP_FIELDS, build_plotly_figure

if TYPE_CHECKING:
    from types import ModuleType

    from timezone_experiments.config import CoverageType
    from timezone_experiments.optimization.optimizer import AssignmentResult


class UploadedFile(Protocol):
    """Small subset of Streamlit's uploaded file interface."""

    name: str

    def getvalue(self) -> bytes:
        """Return the uploaded file contents."""
        ...


@dataclass(frozen=True, slots=True)
class DashboardInputs:
    """Values collected from the dashboard controls."""

    boundary_upload: UploadedFile | None
    population_upload: UploadedFile | None
    timezone_upload: UploadedFile | None
    boundary_source: str
    source_admin_level: int
    default_admin_level: int
    subdivision_rules: str
    country_boundary_levels: str
    solar_metric: str
    coverage_type: CoverageType
    coverage_threshold: float
    minimum_noon: str
    maximum_noon: str
    year: int
    map_field: str
    compare_current_timezones: bool
    include_antarctica: bool


def main(argv: list[str] | None = None) -> None:
    """Render the interactive experiment dashboard."""
    try:
        st = importlib.import_module("streamlit")
    except ImportError as error:
        msg = "The dashboard requires Streamlit; install with `uv sync --extra visualization`."
        raise RuntimeError(msg) from error
    st.set_page_config(page_title="Timezone Experiments", layout="wide")
    st.title("Hypothetical timezone experiments")
    st.caption("Explore permanent UTC offsets using configurable solar-noon and coverage rules.")
    args = build_parser().parse_args(["dashboard", *(argv or [])])
    try:
        config = load_config(args.config)
        st.info("Required data is downloaded automatically and cached. File uploads are optional overrides.")
        inputs = _dashboard_inputs(st, config)
        if st.button("Run experiment", type="primary"):
            _execute_experiment(st, inputs, config, input_options(args))
    except (TypeError, ValueError, OSError, RuntimeError, BadZipFile) as error:
        st.error(str(error))


def _dashboard_inputs(st: ModuleType, config: ExperimentConfig) -> DashboardInputs:
    with st.expander("Advanced: use your own input files", expanded=False):
        boundary_upload = st.file_uploader("Administrative boundaries (GeoJSON)", type=("json", "geojson"))
        population_upload = st.file_uploader("Optional population grid (CSV)", type=("csv",))
        timezone_upload = st.file_uploader("Optional current timezone boundaries (GeoJSON)", type=("json", "geojson"))
        boundary_source = st.selectbox(
            "Boundary source for optional uploads", ("Natural Earth", "GADM", "GeoBoundaries")
        )
        source_admin_level = st.number_input(
            "Boundary file administrative level (for files without GID fields)",
            min_value=0,
            value=0 if boundary_source == "Natural Earth" else 1,
            step=1,
        )
    left, right = st.columns(2)
    with left:
        default_level = st.number_input(
            "Administrative level for countries without an override",
            min_value=0,
            value=config.default_admin_level,
            step=1,
            help=(
                "Configuration: default_admin_level. 0 = countries, 1 = states/provinces, 2 = districts. "
                "Country subdivision rules and country boundary levels below take precedence over this fallback level."
            ),
        )
        rules = st.text_area(
            "Country subdivision rules (ISO code to level)",
            value=json.dumps(config.subdivision_rules),
            help=(
                "Configuration: subdivision_rules. Defaults to the major-multizone countries at level 1. "
                "An explicit mapping replaces those defaults. Country boundary levels below remain separate overrides."
            ),
        )
        country_boundaries = st.text_area(
            "Country boundary levels (ISO code to source level)",
            value=json.dumps(config.country_boundary_levels),
            help=(
                "Configuration: country_boundary_levels. Load finer boundaries but combine them into one country. "
                "Defaults to FRA and NLD at level 1; absent overseas areas are excluded from calculations and maps. "
                "Use {} to disable these overrides."
            ),
        )
        metrics = ("min", "max", "median")
        solar_metric = st.selectbox("Solar-noon statistic", metrics, index=metrics.index(config.solar_metric))
    with right:
        coverage_types = ("area", "longitude", "population")
        coverage_type = st.selectbox("Coverage rule", coverage_types, index=coverage_types.index(config.coverage_type))
        threshold = st.slider(
            "Required coverage", min_value=0.0, max_value=1.0, value=config.coverage_threshold, step=0.01
        )
        minimum_noon = st.text_input("Minimum solar noon (HH:MM)", value=config.minimum_noon or "")
        maximum_noon = st.text_input("Maximum solar noon (optional, HH:MM)", value=config.maximum_noon or "")
    year = st.number_input("Solar year", min_value=1900, max_value=2200, value=config.year, step=1)
    map_field = st.selectbox("Map statistic", MAP_FIELDS)
    compare_timezones = st.checkbox("Compare with current timezones", value=config.compare_current_timezones)
    include_antarctica = st.checkbox(
        "Include Antarctica (requires relaxed noon constraints)", value=config.include_antarctica
    )
    return DashboardInputs(
        boundary_upload=boundary_upload,
        population_upload=population_upload,
        timezone_upload=timezone_upload,
        boundary_source=boundary_source,
        source_admin_level=int(source_admin_level),
        default_admin_level=int(default_level),
        subdivision_rules=rules,
        country_boundary_levels=country_boundaries,
        solar_metric=solar_metric,
        coverage_type=cast("CoverageType", coverage_type),
        coverage_threshold=threshold,
        minimum_noon=minimum_noon,
        maximum_noon=maximum_noon,
        year=int(year),
        map_field=map_field,
        compare_current_timezones=compare_timezones,
        include_antarctica=include_antarctica,
    )


def _execute_experiment(
    st: ModuleType,
    inputs: DashboardInputs,
    base_config: ExperimentConfig,
    options: InputOptions,
) -> None:
    try:
        config = _build_config(inputs, base_config)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            options = replace(
                options,
                boundaries_path=_uploaded_path(
                    inputs.boundary_upload, directory / "boundaries.geojson", options.boundaries_path
                ),
                population_path=_uploaded_path(
                    inputs.population_upload, directory / "population.csv", options.population_path
                ),
                timezones_path=_uploaded_path(
                    inputs.timezone_upload, directory / "timezones.geojson", options.timezones_path
                ),
                source_admin_level=(
                    inputs.source_admin_level if inputs.boundary_upload is not None else options.source_admin_level
                ),
            )
            st.info("Preparing input data. Initial downloads may take several minutes.")
            territories = load_experiment_territories(config, options)
            progress = st.progress(0, text="Optimizing territories...")

            def update_progress(index: int, total: int, territory_id: str) -> None:
                progress.progress(index / total, text=f"Optimizing {territory_id} ({index}/{total})")

            results = run_experiment(config, territories, progress=update_progress)
        st.success(f"Optimized {len(results)} territories.")
        st.dataframe([_result_row(result) for result in results], use_container_width=True)
        st.plotly_chart(
            build_plotly_figure(territories, results, inputs.map_field, workers=config.workers),
            use_container_width=True,
        )
        st.download_button(
            "Download results CSV",
            data=_csv_text(results),
            file_name="timezone_experiment.csv",
            mime="text/csv",
        )
    except (TypeError, ValueError, OSError, RuntimeError) as error:
        st.error(str(error))


def _uploaded_path(upload: UploadedFile | None, path: Path, fallback: Path | None) -> Path | None:
    if upload is None:
        return fallback
    path.write_bytes(upload.getvalue())
    return path


def _build_config(inputs: DashboardInputs, base: ExperimentConfig) -> ExperimentConfig:
    return replace(
        base,
        default_admin_level=inputs.default_admin_level,
        subdivision_rules=_parse_subdivision_rules(inputs.subdivision_rules),
        country_boundary_levels=_parse_subdivision_rules(inputs.country_boundary_levels),
        solar_metric=inputs.solar_metric,
        coverage_type=inputs.coverage_type,
        coverage_threshold=inputs.coverage_threshold,
        minimum_noon=inputs.minimum_noon or None,
        maximum_noon=inputs.maximum_noon or None,
        year=inputs.year,
        compare_current_timezones=inputs.compare_current_timezones,
        include_antarctica=inputs.include_antarctica,
    )


def _parse_subdivision_rules(value: str) -> dict[str, int]:
    rules = json.loads(value)
    if not isinstance(rules, dict):
        msg = "Country rules must be a JSON object with integer levels."
        raise TypeError(msg)
    parsed_rules: dict[str, int] = {}
    for country, level in rules.items():
        if not isinstance(country, str) or not isinstance(level, int) or isinstance(level, bool):
            msg = "Country rules must be a JSON object with integer levels."
            raise TypeError(msg)
        parsed_rules[country] = level
    return parsed_rules


def _result_row(result: AssignmentResult) -> dict[str, object]:
    return {
        "country": result.country,
        "region": result.region,
        "current_timezone": result.current_timezone,
        "proposed_timezone": result.proposed_timezone,
        "timezone_shift_hours": result.timezone_shift_hours,
        "solar_noon_statistic": result.solar_noon_statistic,
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
