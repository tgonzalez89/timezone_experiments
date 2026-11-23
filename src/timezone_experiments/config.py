"""Configuration models for timezone experiments."""

from __future__ import annotations

import json
import os
import sys

# Domain values such as degrees, minutes, and percentage bounds are intentionally literal.
# ruff: noqa: PLR2004
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

type SolarMetric = Literal["min", "max", "median"]
type CoverageType = Literal["longitude", "area", "population"]

MAJOR_MULTIZONE_COUNTRIES: dict[str, int] = {
    "USA": 1,  # Wide landmass longitude span; geographically separated territories.
    "CAN": 1,  # Wide landmass longitude span.
    "RUS": 1,  # Wide landmass longitude span; geographically separated territories.
    "CHN": 1,  # Wide landmass longitude span.
    "AUS": 1,  # Wide landmass longitude span.
    "BRA": 1,  # Wide landmass longitude span.
    "IND": 1,  # Wide landmass longitude span.
    "IDN": 1,  # Island groups span many longitudes.
    "MEX": 1,  # Wide landmass longitude span.
    "KAZ": 1,  # Wide landmass longitude span.
    "MNG": 1,  # Wide landmass longitude span.
    "KIR": 1,  # Island groups span many longitudes.
    "GRL": 1,  # Wide landmass longitude span.
}

COUNTRY_BOUNDARY_LEVELS: dict[str, int] = {
    "FRA": 1,  # Geographically separated territories.
    "NLD": 1,  # Geographically separated territories.
}


def _default_workers() -> int:
    available = os.process_cpu_count() or 1
    # ProcessPoolExecutor on Windows supports at most 61 workers.
    return min(available, 61) if sys.platform == "win32" else available


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Controls territory, solar-noon, coverage, and optimization choices.

    See the "Experiment settings" reference in README.md for field meanings,
    interactions, and supported values. examples/default.json is a runnable
    configuration containing the static defaults; workers are detected at runtime.
    """

    default_admin_level: int = 0
    subdivision_rules: dict[str, int] = field(default_factory=MAJOR_MULTIZONE_COUNTRIES.copy)
    country_boundary_levels: dict[str, int] = field(default_factory=COUNTRY_BOUNDARY_LEVELS.copy)
    solar_metric: SolarMetric | str = "median"
    coverage_type: CoverageType = "longitude"
    coverage_threshold: float = 0.95
    minimum_noon: str | None = "12:00"
    maximum_noon: str | None = None
    utc_offsets: tuple[float, ...] = tuple(float(offset) for offset in range(-12, 15))
    year: int = 2025
    sample_grid_size: int = 12
    event_map_cell_degrees: float = 0.5
    map_geometry_simplification_degrees: float = 0.01
    rasterize_static_map_fills: bool = False
    workers: int = field(default_factory=_default_workers)
    compare_current_timezones: bool = False
    population_cell_degrees: float = 1.0
    include_antarctica: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values eagerly."""
        for name, minimum in (("subdivision_rules", 0), ("country_boundary_levels", 1)):
            object.__setattr__(self, name, _country_levels(getattr(self, name), name, minimum))
        self._validate_admin_level()
        self._validate_search_options()
        self._validate_solar_metric()
        self._validate_clock_options()
        self._validate_data_options()

    def _validate_data_options(self) -> None:
        if not isinstance(self.compare_current_timezones, bool) or not isinstance(self.include_antarctica, bool):
            msg = "compare_current_timezones and include_antarctica must be booleans."
            raise TypeError(msg)
        if not isinstance(self.rasterize_static_map_fills, bool):
            msg = "rasterize_static_map_fills must be a boolean."
            raise TypeError(msg)
        if isinstance(self.population_cell_degrees, bool) or self.population_cell_degrees not in {0.1, 0.25, 0.5, 1.0}:
            msg = "population_cell_degrees must be 0.1, 0.25, 0.5, or 1.0."
            raise ValueError(msg)
        if isinstance(self.event_map_cell_degrees, bool) or self.event_map_cell_degrees not in {0.25, 0.5, 1.0, 2.0}:
            msg = "event_map_cell_degrees must be 0.25, 0.5, 1.0, or 2.0."
            raise ValueError(msg)
        if isinstance(
            self.map_geometry_simplification_degrees, bool
        ) or self.map_geometry_simplification_degrees not in {0.0, 0.005, 0.01, 0.02}:
            msg = "map_geometry_simplification_degrees must be 0, 0.005, 0.01, or 0.02."
            raise ValueError(msg)

    def _validate_admin_level(self) -> None:
        if self.default_admin_level < 0:
            msg = "default_admin_level must be non-negative."
            raise ValueError(msg)
        conflicts = {country for country in self.country_boundary_levels if self.subdivision_rules.get(country, 0) > 0}
        if conflicts:
            msg = f"country_boundary_levels cannot overlap positive subdivision_rules: {', '.join(sorted(conflicts))}."
            raise ValueError(msg)

    def _validate_search_options(self) -> None:
        if not 0.0 <= self.coverage_threshold <= 1.0:
            msg = "coverage_threshold must be between 0 and 1."
            raise ValueError(msg)
        if not self.utc_offsets:
            msg = "At least one UTC offset must be configured."
            raise ValueError(msg)
        if any(not -14 <= offset <= 14 for offset in self.utc_offsets):
            msg = "UTC offsets must be between -14 and +14 hours."
            raise ValueError(msg)
        if self.sample_grid_size < 1:
            msg = "sample_grid_size must be positive."
            raise ValueError(msg)
        if self.workers < 1:
            msg = "workers must be positive."
            raise ValueError(msg)

    def _validate_solar_metric(self) -> None:
        if self.coverage_type not in {"longitude", "area", "population"}:
            msg = f"Unsupported coverage type: {self.coverage_type!r}."
            raise ValueError(msg)
        if not isinstance(self.solar_metric, str) or self.solar_metric not in {"min", "max", "median"}:
            msg = f"Unsupported solar metric: {self.solar_metric!r}."
            raise ValueError(msg)

    def _validate_clock_options(self) -> None:
        minimum = _parse_clock(self.minimum_noon)
        maximum = _parse_clock(self.maximum_noon)
        if minimum is not None and maximum is not None and minimum > maximum:
            msg = "minimum_noon cannot be later than maximum_noon."
            raise ValueError(msg)

    def admin_level_for(self, country_code: str) -> int:
        """Return the configured subdivision level for a country."""
        country_code = country_code.upper()
        if country_code in self.country_boundary_levels:
            return 0
        return self.subdivision_rules.get(country_code, self.default_admin_level)

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> ExperimentConfig:
        """Build a configuration from a JSON-compatible mapping."""
        data = dict(values)
        allowed = {
            "default_admin_level",
            "subdivision_rules",
            "country_boundary_levels",
            "solar_metric",
            "coverage_type",
            "coverage_threshold",
            "minimum_noon",
            "maximum_noon",
            "utc_offsets",
            "year",
            "sample_grid_size",
            "event_map_cell_degrees",
            "map_geometry_simplification_degrees",
            "rasterize_static_map_fills",
            "workers",
            "compare_current_timezones",
            "population_cell_degrees",
            "include_antarctica",
        }
        unknown = data.keys() - allowed
        if unknown:
            msg = f"Unknown experiment configuration fields: {', '.join(sorted(unknown))}."
            raise ValueError(msg)
        defaults = cls()
        rules = data.get("subdivision_rules", defaults.subdivision_rules)
        boundary_levels = data.get("country_boundary_levels", defaults.country_boundary_levels)
        offsets = data.get("utc_offsets")
        if offsets is None:
            offset_values = defaults.utc_offsets
        elif isinstance(offsets, (list, tuple)) and all(
            isinstance(offset, (float, int)) and not isinstance(offset, bool) for offset in offsets
        ):
            offset_values = tuple(float(offset) for offset in offsets)
        else:
            msg = "utc_offsets must be a sequence of numbers."
            raise TypeError(msg)
        default_admin_level = _mapping_integer(data, "default_admin_level", 0)
        coverage_threshold = _mapping_number(data, "coverage_threshold", defaults.coverage_threshold)
        year = _mapping_integer(data, "year", 2025)
        sample_grid_size = _mapping_integer(data, "sample_grid_size", 12)
        workers = _mapping_integer(data, "workers", defaults.workers)
        minimum_noon = _mapping_optional_string(data, "minimum_noon", "12:00")
        maximum_noon = _mapping_optional_string(data, "maximum_noon", None)
        return cls(
            default_admin_level=default_admin_level,
            subdivision_rules=_country_levels(rules, "subdivision_rules", 0),
            country_boundary_levels=_country_levels(boundary_levels, "country_boundary_levels", 1),
            solar_metric=cast("SolarMetric | str", data.get("solar_metric", "median")),
            coverage_type=cast("CoverageType", data.get("coverage_type", defaults.coverage_type)),
            coverage_threshold=coverage_threshold,
            minimum_noon=minimum_noon,
            maximum_noon=maximum_noon,
            utc_offsets=offset_values,
            year=year,
            sample_grid_size=sample_grid_size,
            event_map_cell_degrees=_mapping_number(data, "event_map_cell_degrees", 0.5),
            map_geometry_simplification_degrees=_mapping_number(data, "map_geometry_simplification_degrees", 0.01),
            rasterize_static_map_fills=_mapping_boolean(data, "rasterize_static_map_fills", default=False),
            workers=workers,
            compare_current_timezones=_mapping_boolean(data, "compare_current_timezones", default=False),
            include_antarctica=_mapping_boolean(data, "include_antarctica", default=False),
            population_cell_degrees=_mapping_number(data, "population_cell_degrees", 1.0),
        )


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load a configuration JSON, or use built-in defaults when omitted."""
    if path is None:
        return ExperimentConfig()
    with Path(path).open(encoding="utf-8") as file:
        values = json.load(file)
    if not isinstance(values, dict):
        msg = "Experiment configuration must be a JSON object."
        raise TypeError(msg)
    return ExperimentConfig.from_mapping(values)


def _country_levels(value: object, name: str, minimum: int) -> dict[str, int]:
    if not isinstance(value, dict):
        msg = f"{name} must map country codes to integer administrative levels."
        raise TypeError(msg)
    levels: dict[str, int] = {}
    for country, level in value.items():
        if not isinstance(country, str) or not isinstance(level, int) or isinstance(level, bool):
            msg = f"{name} must map country codes to integer administrative levels."
            raise TypeError(msg)
        if level < minimum:
            msg = f"{name} administrative levels must be {'positive' if minimum else 'non-negative'}."
            raise ValueError(msg)
        levels[country.upper()] = level
    return levels


def _mapping_boolean(values: dict[str, object], key: str, *, default: bool) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        msg = f"{key} must be a boolean."
        raise TypeError(msg)
    return value


def _parse_clock(value: str | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"Clock time must be a string or null: {value!r}."
        raise TypeError(msg)
    try:
        hour_text, minute_text = value.split(":")
        hour, minute = int(hour_text), int(minute_text)
    except (ValueError, AttributeError) as error:
        msg = f"Clock time must use HH:MM format: {value!r}."
        raise ValueError(msg) from error
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        msg = f"Clock time must be between 00:00 and 23:59: {value!r}."
        raise ValueError(msg)
    return hour * 60 + minute


def clock_minutes(value: str | None) -> int | None:
    """Convert an optional HH:MM clock value to minutes after midnight."""
    return _parse_clock(value)


def _mapping_integer(values: dict[str, object], key: str, default: int) -> int:
    value = values.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer."
        raise TypeError(msg)
    return value


def _mapping_number(values: dict[str, object], key: str, default: float) -> float:
    value = values.get(key, default)
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        msg = f"{key} must be a number."
        raise TypeError(msg)
    return float(value)


def _mapping_optional_string(values: dict[str, object], key: str, default: str | None) -> str | None:
    value = values.get(key, default)
    if value is not None and not isinstance(value, str):
        msg = f"{key} must be a string or null."
        raise TypeError(msg)
    return value
