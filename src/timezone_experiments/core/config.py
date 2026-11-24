"""Configuration models for timezone experiments."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from timezone_experiments.core.constants import (
    DEFAULT_MAP_SIMPLIFICATION_DEGREES,
    EVENT_MAP_CELL_DEGREES,
    MAP_SIMPLIFICATION_DEGREES,
    MAX_SOLAR_YEAR,
    MAX_UTC_OFFSET_HOURS,
    MIN_SOLAR_YEAR,
    MIN_UTC_OFFSET_HOURS,
    POPULATION_CELL_DEGREES,
    normalize_country_code,
    validate_choice,
)
from timezone_experiments.core.models import NoonConstraint
from timezone_experiments.core.timezones import parse_clock_minutes

type SolarMetric = Literal["min", "max", "median"]
type CoverageType = Literal["longitude", "area", "population"]
SOLAR_METRICS: tuple[SolarMetric, ...] = ("min", "max", "median")
COVERAGE_TYPES: tuple[CoverageType, ...] = ("longitude", "area", "population")


class ConfigError(ValueError):
    """An experiment configuration value is missing, malformed, or out of range."""


MAJOR_MULTIZONE_COUNTRIES: frozendict[str, int] = frozendict(
    USA=1,  # Wide landmass longitude span; geographically separated territories.
    CAN=1,  # Wide landmass longitude span.
    RUS=1,  # Wide landmass longitude span; geographically separated territories.
    CHN=1,  # Wide landmass longitude span.
    AUS=1,  # Wide landmass longitude span.
    BRA=1,  # Wide landmass longitude span.
    IND=1,  # Wide landmass longitude span.
    IDN=1,  # Island groups span many longitudes.
    MEX=1,  # Wide landmass longitude span.
    KAZ=1,  # Wide landmass longitude span.
    MNG=1,  # Wide landmass longitude span.
    KIR=1,  # Island groups span many longitudes.
    GRL=1,  # Wide landmass longitude span.
)

MERGED_SUBDIVISION_LEVELS: frozendict[str, int] = frozendict(
    FRA=1,  # Geographically separated territories.
    NLD=1,  # Geographically separated territories.
)


# Settings that decide which territories exist and their boundaries; maps must use the same ones as the run.
TERRITORY_SETTINGS = (
    "countries",
    "default_admin_level",
    "subdivision_rules",
    "merged_subdivision_levels",
    "include_antarctica",
)


def _default_workers() -> int:
    available = os.process_cpu_count() or 1
    # ProcessPoolExecutor on Windows supports at most 61 workers.
    return min(available, 61) if sys.platform == "win32" else available


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentConfig:
    """Controls territory, solar-noon, coverage, and optimization choices.

    See the "Experiment settings" reference in README.md for field meanings,
    interactions, and supported values. examples/default.json is a runnable
    configuration containing the static defaults; workers are detected at runtime.

    Field types are trusted here (``from_mapping`` checks them for JSON input); ``__post_init__`` checks ranges,
    allowed values, and cross-field rules. Country-level mappings are stored as immutable ``frozendict``s, so a
    configuration is hashable and cannot change after validation.
    """

    # Solar-noon rule.
    minimum_noon: str | None = "12:00"
    maximum_noon: str | None = None
    solar_metric: SolarMetric = "median"
    year: int = 2026
    # How much of each territory must satisfy the rule.
    coverage_type: CoverageType = "area"
    coverage_threshold: float = 0.9
    # Candidate permanent offsets.
    utc_offsets: tuple[float, ...] = tuple(float(offset) for offset in range(-12, 15))
    # Territories: which countries, at which administrative levels.
    countries: tuple[str, ...] | None = None
    default_admin_level: int = 0
    subdivision_rules: Mapping[str, int] = MAJOR_MULTIZONE_COUNTRIES
    merged_subdivision_levels: Mapping[str, int] = MERGED_SUBDIVISION_LEVELS
    include_antarctica: bool = False
    # Accuracy and performance.
    sample_grid_size: int = 12
    population_cell_degrees: float = 1.0
    workers: int = field(default_factory=_default_workers)
    # Map exports.
    event_map_cell_degrees: float = 0.5
    map_geometry_simplification_degrees: float = DEFAULT_MAP_SIMPLIFICATION_DEGREES
    rasterize_static_map_fills: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values eagerly."""
        object.__setattr__(self, "countries", parse_countries(self.countries, "countries"))
        object.__setattr__(self, "subdivision_rules", parse_country_levels(self.subdivision_rules, "subdivision_rules"))
        object.__setattr__(
            self,
            "merged_subdivision_levels",
            parse_country_levels(self.merged_subdivision_levels, "merged_subdivision_levels", minimum=1),
        )
        validate_choice("solar_metric", self.solar_metric, SOLAR_METRICS, ConfigError)
        validate_choice("coverage_type", self.coverage_type, COVERAGE_TYPES, ConfigError)
        validate_choice("population_cell_degrees", self.population_cell_degrees, POPULATION_CELL_DEGREES, ConfigError)
        validate_choice("event_map_cell_degrees", self.event_map_cell_degrees, EVENT_MAP_CELL_DEGREES, ConfigError)
        validate_choice(
            "map_geometry_simplification_degrees",
            self.map_geometry_simplification_degrees,
            MAP_SIMPLIFICATION_DEGREES,
            ConfigError,
        )
        self._validate_ranges()
        self._validate_cross_field_rules()

    def _validate_ranges(self) -> None:
        if self.default_admin_level < 0:
            msg = "default_admin_level must be non-negative."
            raise ConfigError(msg)
        if self.sample_grid_size < 1:
            msg = "sample_grid_size must be positive."
            raise ConfigError(msg)
        if self.workers < 1:
            msg = "workers must be positive."
            raise ConfigError(msg)
        if not 0.0 <= self.coverage_threshold <= 1.0:
            msg = "coverage_threshold must be between 0 and 1."
            raise ConfigError(msg)
        if not self.utc_offsets:
            msg = "At least one UTC offset must be configured."
            raise ConfigError(msg)
        if any(not MIN_UTC_OFFSET_HOURS <= offset <= MAX_UTC_OFFSET_HOURS for offset in self.utc_offsets):
            msg = "UTC offsets must be between -14 and +14 hours."
            raise ConfigError(msg)
        if not MIN_SOLAR_YEAR <= self.year <= MAX_SOLAR_YEAR:
            msg = f"year must be between {MIN_SOLAR_YEAR} and {MAX_SOLAR_YEAR}: {self.year}."
            raise ConfigError(msg)

    def _validate_cross_field_rules(self) -> None:
        if self.countries is not None and "ATA" in self.countries and not self.include_antarctica:
            msg = "countries includes ATA (Antarctica); also set include_antarctica to true."
            raise ConfigError(msg)
        conflicts = {
            country for country in self.merged_subdivision_levels if self.subdivision_rules.get(country, 0) > 0
        }
        if conflicts:
            msg = (
                f"merged_subdivision_levels cannot overlap positive subdivision_rules: {', '.join(sorted(conflicts))}."
            )
            raise ConfigError(msg)
        minimum = _clock_minutes(self.minimum_noon)
        maximum = _clock_minutes(self.maximum_noon)
        if minimum is not None and maximum is not None and minimum > maximum:
            msg = "minimum_noon cannot be later than maximum_noon."
            raise ConfigError(msg)

    def to_mapping(self) -> dict[str, Any]:
        """Return the JSON-compatible settings, without ``workers`` so it is detected on each machine."""
        values = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "workers"}
        return {key: dict(value) if isinstance(value, Mapping) else value for key, value in values.items()}

    @property
    def noon_constraint(self) -> NoonConstraint:
        """Return the configured solar-noon bounds in minutes after midnight."""
        return NoonConstraint(_clock_minutes(self.minimum_noon), _clock_minutes(self.maximum_noon))

    def admin_level_for(self, country_code: str) -> int:
        """Return the configured subdivision level for a country."""
        country_code = country_code.upper()
        if country_code in self.merged_subdivision_levels:
            return 0
        return self.subdivision_rules.get(country_code, self.default_admin_level)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> ExperimentConfig:
        """Build a configuration from a JSON-compatible mapping; omitted settings keep their defaults."""
        unknown = values.keys() - {item.name for item in fields(cls)}
        if unknown:
            msg = f"Unknown experiment configuration fields: {', '.join(sorted(unknown))}."
            raise ConfigError(msg)
        return cls(**{key: _FIELD_PARSERS[key](key, value) for key, value in values.items()})


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load a configuration JSON, or use built-in defaults when omitted."""
    if path is None:
        return ExperimentConfig()
    try:
        with Path(path).open(encoding="utf-8") as file:
            values = json.load(file)
        if not isinstance(values, dict):
            msg = "Experiment configuration must be a JSON object."
            raise ConfigError(msg)
        return ExperimentConfig.from_mapping(values)
    except ValueError as error:  # Includes json.JSONDecodeError.
        msg = f"Invalid configuration {path}: {error}"
        raise ConfigError(msg) from error


def parse_country_levels(value: object, name: str, *, minimum: int = 0) -> frozendict[str, int]:
    """Validate a ``{country code: admin level}`` mapping, normalizing codes to upper case."""
    if not isinstance(value, Mapping):
        msg = f"{name} must map country codes to integer administrative levels."
        raise ConfigError(msg)
    levels: dict[str, int] = {}
    for country, level in value.items():
        if not isinstance(country, str) or not isinstance(level, int) or isinstance(level, bool):
            msg = f"{name} must map country codes to integer administrative levels."
            raise ConfigError(msg)
        if level < minimum:
            msg = f"{name} administrative levels must be {'positive' if minimum else 'non-negative'}."
            raise ConfigError(msg)
        try:
            levels[normalize_country_code(country)] = level
        except ValueError as error:
            raise ConfigError(str(error)) from error
    return frozendict(levels)


def parse_countries(value: object, name: str) -> tuple[str, ...] | None:
    """Validate an optional list of country codes, returning them upper-cased, sorted, and de-duplicated."""
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        msg = f"{name} must be a list of country codes, or null for all countries."
        raise ConfigError(msg)
    if not value:
        msg = f"{name} must list at least one country code; use null for all countries."
        raise ConfigError(msg)
    try:
        return tuple(sorted({normalize_country_code(country) for country in value}))
    except ValueError as error:
        raise ConfigError(str(error)) from error


def _integer(key: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer."
        raise ConfigError(msg)
    return value


def _number(key: str, value: object) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        msg = f"{key} must be a number."
        raise ConfigError(msg)
    return float(value)


def _boolean(key: str, value: object) -> bool:
    if not isinstance(value, bool):
        msg = f"{key} must be a boolean."
        raise ConfigError(msg)
    return value


def _optional_string(key: str, value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        msg = f"{key} must be a string or null."
        raise ConfigError(msg)
    return value


def _offsets(key: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        msg = f"{key} must be a sequence of numbers."
        raise ConfigError(msg)
    return tuple(_number(key, offset) for offset in value)


def _choice[T: str](choices: tuple[T, ...]) -> Callable[[str, object], T]:
    def parse(key: str, value: object) -> T:
        for choice in choices:
            if value == choice and isinstance(value, str):
                return choice
        msg = f"{key} must be one of {', '.join(choices)}: {value!r}."
        raise ConfigError(msg)

    return parse


_FIELD_PARSERS: dict[str, Callable[[str, object], Any]] = {
    "minimum_noon": _optional_string,
    "maximum_noon": _optional_string,
    "solar_metric": _choice(SOLAR_METRICS),
    "year": _integer,
    "coverage_type": _choice(COVERAGE_TYPES),
    "coverage_threshold": _number,
    "utc_offsets": _offsets,
    "countries": lambda key, value: parse_countries(value, key),
    "default_admin_level": _integer,
    "subdivision_rules": lambda key, value: parse_country_levels(value, key),
    "merged_subdivision_levels": lambda key, value: parse_country_levels(value, key, minimum=1),
    "include_antarctica": _boolean,
    "sample_grid_size": _integer,
    "population_cell_degrees": _number,
    "workers": _integer,
    "event_map_cell_degrees": _number,
    "map_geometry_simplification_degrees": _number,
    "rasterize_static_map_fills": _boolean,
}


def _clock_minutes(value: str | None) -> int | None:
    try:
        return parse_clock_minutes(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error
