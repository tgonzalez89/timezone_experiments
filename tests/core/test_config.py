"""Experiment configuration: defaults, parsing, validation, and the example files."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import pytest
from factories import (
    ROOT,
)

from timezone_experiments.core import config as config_module
from timezone_experiments.core.config import (
    MAJOR_MULTIZONE_COUNTRIES,
    ConfigError,
    ExperimentConfig,
    load_config,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("example", sorted((ROOT / "examples").glob("*.json")), ids=lambda path: path.name)
def test_example_configurations_keep_default_subdivision_rules(example: Path) -> None:
    with example.open(encoding="utf-8") as file:
        config = ExperimentConfig.from_mapping(json.load(file))
    assert config.admin_level_for("usa") == 1
    assert config.admin_level_for("KIR") == 1
    assert config.subdivision_rules == MAJOR_MULTIZONE_COUNTRIES


def test_default_example_matches_builtin_defaults() -> None:
    with (ROOT / "examples" / "default.json").open(encoding="utf-8") as file:
        values = json.load(file)
    assert set(values) == set(asdict(ExperimentConfig())) - {"workers"}
    assert ExperimentConfig.from_mapping(values) == ExperimentConfig()


def test_countries_are_normalized_sorted_and_deduplicated() -> None:
    assert ExperimentConfig().countries is None
    assert ExperimentConfig.from_mapping({"countries": ["usa", "MEX", "usa"]}).countries == ("MEX", "USA")
    assert ExperimentConfig(countries=("ATA",), include_antarctica=True).countries == ("ATA",)


@pytest.mark.parametrize(
    ("cpu_count", "platform", "expected"),
    [(8, "linux", 8), (20, "win32", 20), (128, "win32", 61), (None, "linux", 1)],
)
def test_automatic_workers_use_available_cpu_threads(
    monkeypatch: pytest.MonkeyPatch, cpu_count: int | None, platform: str, expected: int
) -> None:
    monkeypatch.setattr(config_module.os, "process_cpu_count", lambda: cpu_count)
    with monkeypatch.context() as context:
        context.setattr(config_module.sys, "platform", platform)
        assert ExperimentConfig().workers == expected
        assert ExperimentConfig.from_mapping({}).workers == expected
        assert ExperimentConfig.from_mapping({"workers": 2}).workers == 2


def test_default_rules_preserve_explicit_overrides() -> None:
    config = ExperimentConfig()
    assert config.solar_metric == "median"
    assert config.coverage_type == "area"
    assert config.coverage_threshold == 0.9
    assert config.year == 2026
    assert config.subdivision_rules == MAJOR_MULTIZONE_COUNTRIES
    assert ExperimentConfig.from_mapping({}) == config
    assert config.admin_level_for("USA") == 1
    assert config.admin_level_for("MNG") == 1
    assert config.admin_level_for("KIR") == 1
    assert config.admin_level_for("ARG") == 0
    assert config.admin_level_for("FRA") == 0
    assert config.admin_level_for("NLD") == 0
    assert config.admin_level_for("GRL") == 1
    overrides = ExperimentConfig.from_mapping({"subdivision_rules": {"USA": 0}, "workers": 1})
    assert overrides.subdivision_rules == {"USA": 0}
    assert overrides.workers == 1
    assert overrides.admin_level_for("CAN") == 0
    assert ExperimentConfig.from_mapping({"subdivision_rules": {}}).subdivision_rules == {}
    with pytest.raises(TypeError):
        config.subdivision_rules["USA"] = 0  # ty: ignore[invalid-assignment]
    assert hash(config) == hash(ExperimentConfig(workers=config.workers))
    assert ExperimentConfig().subdivision_rules == MAJOR_MULTIZONE_COUNTRIES
    assert ExperimentConfig.from_mapping({}).admin_level_for("USA") == 1


def test_merged_subdivision_levels_preserve_country_units() -> None:
    config = ExperimentConfig.from_mapping(
        {"default_admin_level": 2, "subdivision_rules": {}, "merged_subdivision_levels": {"fra": 1}}
    )
    assert config.merged_subdivision_levels == {"FRA": 1}
    assert config.admin_level_for("fra") == 0
    assert config.admin_level_for("NLD") == 2
    assert ExperimentConfig().merged_subdivision_levels == {"FRA": 1, "NLD": 1}


# Every validation branch of ExperimentConfig, as (settings, expected message).
_REJECTED: list[tuple[dict[str, object], str]] = [
    # Noon rule
    ({"minimum_noon": 12}, "minimum_noon must be a string or null"),
    ({"minimum_noon": "25:00"}, "Clock time"),
    ({"maximum_noon": "noon"}, "Clock time"),
    ({"minimum_noon": "14:00", "maximum_noon": "13:00"}, "minimum_noon cannot be later than maximum_noon"),
    ({"solar_metric": "mean"}, "solar_metric must be one of"),
    ({"solar_metric": ["percentile", 5]}, "solar_metric must be one of"),
    ({"year": "2025"}, "year must be an integer"),
    ({"year": True}, "year must be an integer"),
    ({"year": 1899}, "year must be between 1900 and 2200"),
    ({"year": 2201}, "year must be between 1900 and 2200"),
    # Coverage
    ({"coverage_type": "people"}, "coverage_type must be one of"),
    ({"coverage_threshold": 1.1}, "coverage_threshold must be between 0 and 1"),
    ({"coverage_threshold": -0.1}, "coverage_threshold must be between 0 and 1"),
    ({"coverage_threshold": "0.9"}, "coverage_threshold must be a number"),
    # Offsets
    ({"utc_offsets": []}, "At least one UTC offset"),
    ({"utc_offsets": "1,2"}, "utc_offsets must be a sequence of numbers"),
    ({"utc_offsets": [15]}, "between -14 and \\+14 hours"),
    # Territories
    ({"countries": []}, "at least one country code; use null"),
    ({"countries": "USA"}, "list of country codes"),
    ({"countries": [1]}, "list of country codes"),
    ({"countries": ["../x"]}, "three letters"),
    ({"countries": ["ATA"]}, "include_antarctica"),
    ({"default_admin_level": -1}, "default_admin_level must be non-negative"),
    ({"subdivision_rules": ["USA", 1]}, "subdivision_rules must map country codes"),
    ({"subdivision_rules": {"USA": 1.5}}, "subdivision_rules must map country codes"),
    ({"subdivision_rules": {"USA": -1}}, "subdivision_rules administrative levels must be non-negative"),
    ({"subdivision_rules": {"../X": 1}}, "Country codes must be three"),
    ({"subdivision_rules": {"fra": 1}}, "overlap.*FRA"),
    ({"merged_subdivision_levels": []}, "merged_subdivision_levels must map country codes"),
    ({"merged_subdivision_levels": {"FRA": True}}, "merged_subdivision_levels must map country codes"),
    ({"merged_subdivision_levels": {"FRA": 1.5}}, "merged_subdivision_levels must map country codes"),
    ({"merged_subdivision_levels": {"FRA": 0}}, "merged_subdivision_levels administrative levels must be positive"),
    ({"include_antarctica": 1}, "include_antarctica must be a boolean"),
    # Accuracy and performance
    ({"sample_grid_size": 0}, "sample_grid_size must be positive"),
    ({"sample_grid_size": 0.5}, "sample_grid_size must be an integer"),
    ({"population_cell_degrees": True}, "population_cell_degrees must be a number"),
    ({"population_cell_degrees": 0.2}, "population_cell_degrees must be one of"),
    ({"workers": 0}, "workers must be positive"),
    # Maps
    ({"event_map_cell_degrees": True}, "event_map_cell_degrees must be a number"),
    ({"event_map_cell_degrees": 0.75}, "event_map_cell_degrees must be one of"),
    ({"map_geometry_simplification_degrees": True}, "map_geometry_simplification_degrees must be a number"),
    ({"map_geometry_simplification_degrees": 0.001}, "map_geometry_simplification_degrees must be one of"),
    ({"rasterize_static_map_fills": 1}, "rasterize_static_map_fills must be a boolean"),
    # Structure
    ({"unrecognized": True, "other": 1}, "Unknown experiment configuration fields: other, unrecognized"),
]


@pytest.mark.parametrize(
    ("settings", "message"),
    _REJECTED,
    ids=[", ".join(f"{key}={value!r}" for key, value in s.items()) for s, _ in _REJECTED],
)
def test_invalid_settings_are_rejected_with_the_setting_named(settings: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        ExperimentConfig.from_mapping(settings)


def test_config_from_mapping_converts_json_values() -> None:
    config = ExperimentConfig.from_mapping(
        {"solar_metric": "median", "utc_offsets": [-1, 0.5], "coverage_threshold": 0.8, "subdivision_rules": {"usa": 1}}
    )
    assert config.solar_metric == "median"
    assert config.utc_offsets == (-1.0, 0.5)
    assert config.subdivision_rules == {"USA": 1}


@pytest.mark.parametrize(
    ("content", "message"),
    [("[]", "Experiment configuration must be a JSON object"), ("{", "Expecting property name")],
    ids=["not-an-object", "malformed-json"],
)
def test_configuration_loader_names_the_file(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=rf"invalid\.json: {message}"):
        load_config(path)


def test_config_round_trips_through_json_for_every_field() -> None:
    config = ExperimentConfig(
        subdivision_rules={"usa": 2},
        merged_subdivision_levels={},
        solar_metric="max",
        coverage_type="area",
        maximum_noon="14:00",
        utc_offsets=(-1, 0.5),
        event_map_cell_degrees=1.0,
        map_geometry_simplification_degrees=0.0,
        rasterize_static_map_fills=True,
        workers=3,
        population_cell_degrees=0.5,
        include_antarctica=True,
    )
    assert ExperimentConfig.from_mapping(json.loads(json.dumps(asdict(config)))) == config
