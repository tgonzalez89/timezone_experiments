"""The dashboard controls, grouped like the configuration settings; each group returns its values by field name."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from timezone_experiments.app.dashboard.settings import DashboardInputs
from timezone_experiments.core.config import COVERAGE_TYPES, SOLAR_METRICS
from timezone_experiments.core.constants import (
    EVENT_MAP_CELL_DEGREES,
    MAP_SIMPLIFICATION_DEGREES,
    MAX_SOLAR_YEAR,
    MIN_SOLAR_YEAR,
    POPULATION_CELL_DEGREES,
)
from timezone_experiments.maps import MAP_FIELDS, map_title

if TYPE_CHECKING:
    from types import ModuleType

    from timezone_experiments.core.config import ExperimentConfig


def dashboard_inputs(st: ModuleType, config: ExperimentConfig) -> DashboardInputs:
    """Draw every control, initialized from the configuration, and collect their current values."""
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
