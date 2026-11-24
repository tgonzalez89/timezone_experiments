"""Dashboard control values and their translation into an experiment configuration (no Streamlit needed)."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from timezone_experiments.core.config import ConfigError, parse_countries, parse_country_levels

if TYPE_CHECKING:
    from timezone_experiments.core.config import CoverageType, ExperimentConfig, SolarMetric

# Configuration fields without a dashboard control, because they have no effect on what the dashboard shows.
NOT_IN_DASHBOARD = frozenset({"rasterize_static_map_fills"})  # Only static (PNG/SVG) maps are rasterized.


class UploadedFile(Protocol):
    """Small subset of Streamlit's uploaded file interface."""

    name: str

    def getvalue(self) -> bytes:
        """Return the uploaded file contents."""
        ...


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


def build_config(inputs: DashboardInputs, base: ExperimentConfig) -> ExperimentConfig:
    """Apply the control values to the configuration the dashboard was started with."""
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
