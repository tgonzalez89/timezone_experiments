"""Focused tests for experiment configuration, analysis, and outputs."""

from __future__ import annotations

import csv
import json
import logging
import math
import statistics
from copy import deepcopy
from dataclasses import asdict, fields, replace
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Self
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from plotly.colors import sample_colorscale, unlabel_rgb
from shapely import union_all
from shapely.geometry import LineString, MultiLineString, Polygon, box, mapping, shape

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from numpy.typing import NDArray

from timezone_experiments import config as config_module
from timezone_experiments import dashboard
from timezone_experiments.astronomy import solar as solar_module
from timezone_experiments.astronomy.solar import (
    winter_solstice_date,
)
from timezone_experiments.config import (
    MAJOR_MULTIZONE_COUNTRIES,
    ConfigError,
    CoverageType,
    ExperimentConfig,
    SolarMetric,
)
from timezone_experiments.data import boundaries as boundaries_module
from timezone_experiments.data import download_boundaries
from timezone_experiments.data import noon as noon_module
from timezone_experiments.data import timezones as timezones_module
from timezone_experiments.data.boundaries import load_territories, load_territory_geometries, load_timezone_boundaries
from timezone_experiments.data.geometry import point_in_geometry, sample_geometry
from timezone_experiments.data.models import Territory, TerritorySample
from timezone_experiments.data.noon import calculate_noon_statistics
from timezone_experiments.data.population import PopulationPoint, attach_population_weights, load_population_csv
from timezone_experiments.data.timezone_index import TimezoneBoundaryIndex, annotate_current_timezones
from timezone_experiments.data.timezones import (
    current_noon_ranges,
    describe_timezone,
    format_utc_offset,
    parse_utc_offset,
)
from timezone_experiments.experiments.runner import (
    read_results_csv,
    run_experiment,
    write_results_csv,
)
from timezone_experiments.optimization import optimizer as optimizer_module
from timezone_experiments.optimization.constraints import NoonConstraint
from timezone_experiments.optimization.optimizer import AssignmentResult, optimize_territory
from timezone_experiments.territory.coverage import evaluate_coverage
from timezone_experiments.visualization import maps as maps_module
from timezone_experiments.visualization.maps import (
    MAP_FIELDS,
    MapJob,
    MapKind,
    MapOptions,
    build_plotly_figure,
    export_maps,
    geometry_feature_collection,
    result_feature_collection,
)

ROOT = Path(__file__).parents[1]
TOY_BOUNDARIES = ROOT / "tests" / "data" / "toy_boundaries.geojson"
# Rule the dateline tests' expected offsets and noon values were derived with.
LONGITUDE_95_IN_2025 = ExperimentConfig(coverage_type="longitude", coverage_threshold=0.95, year=2025)


def _territory(
    territory_id: str = "TST:1",
    longitude: float = 0,
    *,
    current_timezone: str | None = "UTC+00:00",
    geometry: dict[str, object] | None = None,
    samples: tuple[TerritorySample, ...] | None = None,
) -> Territory:
    return Territory(
        territory_id=territory_id,
        country_code=territory_id.split(":", maxsplit=1)[0],
        name="Test",
        admin_level=0,
        samples=samples or (TerritorySample(latitude=0, longitude=longitude),),
        current_timezone=current_timezone,
        geometry=geometry,
    )


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


def test_countries_are_normalized_and_validated() -> None:
    assert ExperimentConfig().countries is None
    assert ExperimentConfig.from_mapping({"countries": ["usa", "MEX", "usa"]}).countries == ("MEX", "USA")
    for value, message in (
        ([], "at least one country code; use null"),
        ("USA", "list of country codes"),
        ([1], "list of country codes"),
        (["../x"], "three letters"),
        (["ATA"], "include_antarctica"),
    ):
        with pytest.raises(ConfigError, match=message):
            ExperimentConfig.from_mapping({"countries": value})
    assert ExperimentConfig(countries=("ATA",), include_antarctica=True).countries == ("ATA",)


def test_dashboard_countries_text_becomes_the_country_selection() -> None:
    config = ExperimentConfig(countries=("MEX", "USA"))
    inputs = dashboard._dashboard_inputs(_FakeStreamlit(has_upload=False, run=False, rules=None), config)
    assert inputs.countries == "MEX, USA"
    assert dashboard._build_config(inputs, config) == config
    assert dashboard._build_config(replace(inputs, countries=" ; "), config).countries is None
    assert dashboard._build_config(replace(inputs, countries="can;usa"), config).countries == ("CAN", "USA")


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


def test_merged_subdivision_levels_preserve_country_units_and_validate_conflicts() -> None:
    config = ExperimentConfig.from_mapping(
        {"default_admin_level": 2, "subdivision_rules": {}, "merged_subdivision_levels": {"fra": 1}}
    )
    assert config.merged_subdivision_levels == {"FRA": 1}
    assert config.admin_level_for("fra") == 0
    assert config.admin_level_for("NLD") == 2
    assert ExperimentConfig().merged_subdivision_levels == {"FRA": 1, "NLD": 1}
    with pytest.raises(ValueError, match=r"overlap.*FRA"):
        ExperimentConfig.from_mapping({"subdivision_rules": {"fra": 1}})


@pytest.mark.parametrize("levels", [[], {"FRA": True}, {"FRA": 1.5}, {"FRA": 0}, {"FRA": -1}])
def test_invalid_merged_subdivision_levels_are_rejected(levels: object) -> None:
    with pytest.raises(ValueError, match="merged_subdivision_levels"):
        ExperimentConfig.from_mapping({"merged_subdivision_levels": levels})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"coverage_threshold": 1.1}, "coverage_threshold"),
        ({"minimum_noon": "25:00"}, "Clock time"),
        ({"minimum_noon": "14:00", "maximum_noon": "13:00"}, "minimum_noon"),
        ({"utc_offsets": ()}, "At least one"),
        ({"workers": 0}, "workers must be positive"),
        ({"solar_metric": "percentile"}, "solar_metric must be one of"),
        ({"solar_metric": "mean"}, "solar_metric must be one of"),
        ({"solar_metric": ["percentile", 5]}, "solar_metric must be one of"),
        ({"solar_metric": ("percentile", 5)}, "solar_metric must be one of"),
        ({"event_map_cell_degrees": 0.75}, "event_map_cell_degrees"),
        ({"map_geometry_simplification_degrees": 0.001}, "map_geometry_simplification_degrees"),
    ],
)
def test_experiment_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.from_mapping(kwargs)


def test_config_rejects_non_numeric_event_map_resolution() -> None:
    with pytest.raises(ValueError, match="event_map_cell_degrees"):
        ExperimentConfig.from_mapping({"event_map_cell_degrees": True})
    with pytest.raises(ValueError, match="map_geometry_simplification_degrees"):
        ExperimentConfig.from_mapping({"map_geometry_simplification_degrees": True})
    with pytest.raises(ValueError, match="rasterize_static_map_fills"):
        ExperimentConfig.from_mapping({"rasterize_static_map_fills": 1})


def test_config_from_mapping_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown experiment"):
        ExperimentConfig.from_mapping({"unrecognized": True})


def test_config_from_mapping_converts_values_and_rejects_bad_types() -> None:
    config = ExperimentConfig.from_mapping(
        {"solar_metric": "median", "utc_offsets": [-1, 0.5], "coverage_threshold": 0.8},
    )
    assert config.solar_metric == "median"
    assert config.utc_offsets == (-1.0, 0.5)
    with pytest.raises(ValueError, match="utc_offsets"):
        ExperimentConfig.from_mapping({"utc_offsets": "1,2"})
    with pytest.raises(ValueError, match="integer"):
        ExperimentConfig.from_mapping({"year": "2025"})
    with pytest.raises(ValueError, match="subdivision_rules"):
        ExperimentConfig.from_mapping({"subdivision_rules": ["USA", 1]})
    with pytest.raises(ValueError, match="must be a string or null"):
        ExperimentConfig.from_mapping({"minimum_noon": 12})
    with pytest.raises(ValueError, match="must be an integer"):
        ExperimentConfig.from_mapping({"sample_grid_size": 0.5})


def _daily_noon(longitude: float, year: int, offset: float = 0.0) -> list[float]:
    """Return reference daily local solar-noon clock times (not wrapped to a day), straight from the definition."""
    return [720 - 4 * longitude - equation + 60 * offset for equation in solar_module.equation_of_time_minutes(year)]


def _statistic(values: Iterable[float], metric: SolarMetric) -> float:
    data = list(values)
    return min(data) if metric == "min" else max(data) if metric == "max" else float(statistics.median(data))


def test_winter_solstice_follows_the_hemisphere() -> None:
    assert winter_solstice_date(1, 2025) == date(2025, 12, 21)
    assert winter_solstice_date(-1, 2025) == date(2025, 6, 21)


@pytest.mark.parametrize("year", [2024, 2025])
def test_fast_annual_minimum_matches_daily_series_including_midnight_rounding(year: int) -> None:
    equations = solar_module.equation_of_time_minutes(year)
    midnight_longitude = (720 - max(equations)) / 4
    longitudes = [
        -180,
        -157,
        0,
        45,
        179.999,
        midnight_longitude,
        math.nextafter(midnight_longitude, -math.inf),
        math.nextafter(midnight_longitude, math.inf),
    ]
    for offset in (-12, -11.5, 0, 5.75, 14):
        expected = [min(value % 1440 for value in _daily_noon(longitude, year, offset)) for longitude in longitudes]
        actual = solar_module.annual_solar_noon_minutes(np.array(longitudes), year, offset, "min")
        assert actual.tolist() == expected


@pytest.mark.parametrize("year", [2024, 2025])
@pytest.mark.parametrize(
    "metric",
    [
        "median",
        "max",
    ],
)
def test_fast_annual_statistics_match_daily_series_at_clock_wraps(year: int, metric: SolarMetric) -> None:
    equations = solar_module.equation_of_time_minutes(year)
    longitudes = [-180, -157, 0, 45, 179.999]
    for equation in equations:
        longitude = (720 - equation) / 4
        if -180 <= longitude <= 180:
            longitudes.extend((longitude, math.nextafter(longitude, -math.inf), math.nextafter(longitude, math.inf)))
    for offset in (-12, -11.5, 0, 5.75, 14):
        expected = [
            _statistic((value % 1440 for value in _daily_noon(longitude, year, offset)), metric)
            for longitude in longitudes
        ]
        actual = solar_module.annual_solar_noon_minutes(np.array(longitudes), year, offset, metric)
        assert actual.tolist() == expected


@pytest.mark.parametrize("coverage", ["area", "longitude", "population"])
def test_optimizer_skips_only_weightless_samples(coverage: CoverageType, monkeypatch: pytest.MonkeyPatch) -> None:
    active = TerritorySample(0, 0, area_weight=2, longitude_weight=3, population_weight=10)
    padding = TerritorySample(1, 1, area_weight=0, longitude_weight=0, population_weight=0)
    minimal = _territory(samples=(active,))
    padded = replace(minimal, samples=(active, padding))
    config = ExperimentConfig(coverage_type=coverage, minimum_noon=None, workers=1)
    expected = optimize_territory(minimal, config)
    original = optimizer_module.annual_solar_noon_minutes
    evaluated: list[int] = []

    def count(longitudes: NDArray[np.float64], year: int, offset: float, metric: SolarMetric) -> object:
        evaluated.append(len(longitudes))
        return original(longitudes, year, offset, metric)

    monkeypatch.setattr(optimizer_module, "annual_solar_noon_minutes", count)
    result = optimize_territory(padded, config)
    assert replace(result, noon=expected.noon) == expected
    assert evaluated == [1]
    assert padded.samples == (active, padding)


def test_optimizer_keeps_incomplete_population_metadata_and_zero_area_fallback() -> None:
    config = ExperimentConfig(coverage_type="area", minimum_noon=None, workers=1)
    incomplete = _territory(
        samples=(
            TerritorySample(0, 0, population_weight=10),
            TerritorySample(1, 1, area_weight=0, longitude_weight=0),
        )
    )
    assert optimize_territory(incomplete, config).population_covered is None
    zero_area = replace(
        incomplete,
        samples=(
            TerritorySample(0, 0, area_weight=0, longitude_weight=1),
            TerritorySample(0, 15, area_weight=0, longitude_weight=1),
        ),
    )
    uniform_area = replace(zero_area, samples=tuple(replace(sample, area_weight=1) for sample in zero_area.samples))
    longitude_config = replace(config, coverage_type="longitude")
    assert optimize_territory(zero_area, longitude_config) == optimize_territory(uniform_area, longitude_config)


@pytest.mark.parametrize("coverage", ["longitude", "population"])
def test_optimizer_retains_positive_non_area_weights(coverage: CoverageType) -> None:
    territory = _territory(
        samples=(
            TerritorySample(0, 0, population_weight=1),
            TerritorySample(0, 15, area_weight=0, longitude_weight=1, population_weight=1),
        )
    )
    result = optimize_territory(
        territory,
        ExperimentConfig(
            coverage_type=coverage, coverage_threshold=0.5, minimum_noon="11:01", utc_offsets=(0,), workers=1
        ),
    )
    assert result.compliance_rate == 0.5
    assert result.population_covered == 0.5


def test_polygon_sampling_preserves_holes_and_dateline_crossings() -> None:
    holed_polygon = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
            [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]],
        ],
    }
    samples = sample_geometry(holed_polygon, 10)
    assert not any(
        4 <= sample.longitude <= 6 and 4 <= sample.latitude <= 6 and sample.area_weight > 0 for sample in samples
    )
    dateline = {
        "type": "Polygon",
        "coordinates": [[[170, -10], [-170, -10], [-170, 10], [170, 10], [170, -10]]],
    }
    assert point_in_geometry(179, 0, dateline)
    assert point_in_geometry(-179, 0, dateline)
    assert not point_in_geometry(0, 0, dateline)
    dateline_samples = sample_geometry(dateline, 4)
    assert max(sample.area_weight for sample in dateline_samples) < 100
    weighted_samples = sample_geometry(
        {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 2], [0, 2], [0, 0]]]},
        4,
    )
    longitude_weights: dict[float, float] = {}
    for sample in weighted_samples:
        if sample.area_weight > 0:
            longitude_weights[sample.longitude] = longitude_weights.get(sample.longitude, 0) + sample.longitude_weight
    assert len(set(longitude_weights.values())) == 1
    with pytest.raises(ValueError, match="Polygon or MultiPolygon"):
        sample_geometry({"type": "Point", "coordinates": [0, 0]}, 4)
    with pytest.raises(ValueError, match="grid_size"):
        sample_geometry(dateline, 0)
    multi = {"type": "MultiPolygon", "coordinates": [dateline["coordinates"], dateline["coordinates"]]}
    # Both components are sampled on their own 2x2 grid; boundary vertices are no longer samples.
    assert len(sample_geometry(multi, 2)) == 2 * 2 * 2
    assert all(sample.area_weight > 0 for sample in dateline_samples)


@pytest.mark.parametrize(
    "ring",
    [
        [[121.203, -6.7779], [121.2032, -6.7766], [121.203, -6.7779], [121.203, -6.7779]],
        [[0, 0], [1, 1], [2, 2], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
    ],
)
def test_polygon_sampling_skips_zero_area_components(ring: list[list[float]], caplog: pytest.LogCaptureFixture) -> None:
    valid = [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]]
    geometry = {"type": "MultiPolygon", "coordinates": [valid, [ring]]}
    samples = sample_geometry(geometry, 4)
    assert samples == sample_geometry({"type": "Polygon", "coordinates": valid}, 4)
    assert "zero area" in caplog.text
    with pytest.raises(ValueError, match="no non-zero-area"):
        sample_geometry({"type": "Polygon", "coordinates": [ring]}, 4)


def test_boundary_loader_preserves_valid_parts_of_degenerate_multipolygon(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    collection = json.loads(TOY_BOUNDARIES.read_text(encoding="utf-8"))
    valid = collection["features"][0]["geometry"]["coordinates"]
    collapsed = [[[121.203, -6.7779], [121.2032, -6.7766], [121.203, -6.7779], [121.203, -6.7779]]]
    collection["features"] = collection["features"][:1]
    feature = collection["features"][0]
    feature["geometry"] = {"type": "MultiPolygon", "coordinates": [collapsed, valid]}
    path = tmp_path / "boundaries.geojson"
    path.write_text(json.dumps(collection), encoding="utf-8")
    config = ExperimentConfig(sample_grid_size=2)
    territories = load_territories(path, config)
    assert len(territories) == 1
    assert all(0 <= sample.longitude <= 8 for sample in territories[0].samples)
    assert "zero area" in caplog.text
    feature["geometry"]["coordinates"] = [collapsed]
    path.write_text(json.dumps(collection), encoding="utf-8")
    with pytest.raises(ValueError, match=r"TST.*no non-zero-area"):
        load_territories(path, config)


def test_polygon_sampling_preserves_tiny_nonzero_area() -> None:
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [[121.203, -6.7779], [121.203000001, -6.7779], [121.203, -6.777899999], [121.203, -6.7779]],
        ],
    }
    samples = sample_geometry(geometry, 1)
    assert any(sample.area_weight > 0 for sample in samples)


def test_boundary_loader_builds_country_and_mixed_subdivision_units() -> None:
    country = load_territories(TOY_BOUNDARIES, ExperimentConfig(sample_grid_size=3))
    subdivided = load_territories(
        TOY_BOUNDARIES,
        ExperimentConfig(subdivision_rules={"TST": 1}, sample_grid_size=3),
    )
    assert len(country) == 1
    assert country[0].name == "Testland"
    assert country[0].geometry is not None
    assert len(subdivided) == 2
    assert {territory.name for territory in subdivided} == {"West Province", "East Province"}
    with pytest.raises(ValueError, match="only contains level"):
        load_territories(
            TOY_BOUNDARIES,
            ExperimentConfig(subdivision_rules={"TST": 2}),
        )


def test_load_territories_rejects_non_feature_collections(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text('{"solar_metric": "min"}', encoding="utf-8")
    with pytest.raises(ValueError, match="FeatureCollection"):
        load_territories(settings, ExperimentConfig())


def test_population_overlay_and_weighted_coverage(tmp_path: Path) -> None:
    territory = load_territories(TOY_BOUNDARIES, ExperimentConfig(sample_grid_size=3))[0]
    with_population = attach_population_weights(
        [territory],
        [PopulationPoint(latitude=4, longitude=4, population=90)],
    )[0]
    assert any(sample.population_weight == 90 for sample in with_population.samples)
    passes = [sample.population_weight == 90 for sample in with_population.samples]
    summary = evaluate_coverage(with_population, passes, "population")
    assert summary.compliance_rate == 1
    assert summary.population_covered == 1
    csv_path = tmp_path / "population_example.csv"
    csv_path.write_text("latitude,longitude,population\n4,4,90\n", encoding="utf-8")
    assert load_population_csv(csv_path)[0].population == 90


def test_population_csv_validation_and_coverage_errors(tmp_path: Path) -> None:
    invalid = tmp_path / "population.csv"
    invalid.write_text("lat,lon,count\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain"):
        load_population_csv(invalid)
    invalid.write_text("latitude,longitude,population\n1,2,-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"population.csv:2: population must be finite and non-negative"):
        load_population_csv(invalid)
    territory = _territory()
    with pytest.raises(ValueError, match="missing population"):
        evaluate_coverage(territory, [True], "population")
    with pytest.raises(ValueError, match="correspond one-to-one"):
        evaluate_coverage(territory, [], "area")
    with pytest.raises(ValueError, match="positive"):
        evaluate_coverage(
            _territory(samples=(TerritorySample(0, 0, area_weight=0, longitude_weight=0),)),
            [False],
            "longitude",
        )


def test_population_unavailable_uses_explicit_area_fallback(caplog: pytest.LogCaptureFixture) -> None:
    result = optimize_territory(
        _territory(),
        ExperimentConfig(coverage_type="population", utc_offsets=(1, 2)),
    )
    assert result.coverage_type_used == "area"
    assert "using area coverage" in caplog.text


def test_constraint_bounds() -> None:
    assert ExperimentConfig(minimum_noon="12:00", maximum_noon="13:00").noon_constraint.accepts(750)
    assert not ExperimentConfig(minimum_noon=None, maximum_noon="13:00").noon_constraint.accepts(781)
    assert ExperimentConfig(minimum_noon=None).noon_constraint == NoonConstraint(None, None)
    with pytest.raises(ValueError, match="minimum_noon"):
        ExperimentConfig(minimum_noon="14:00", maximum_noon="13:00")


def test_optimizer_selects_earliest_feasible_offset_and_compares_current_zone() -> None:
    config = ExperimentConfig(utc_offsets=(0, 1, 2), minimum_noon="12:00")
    result = optimize_territory(_territory(), config)
    assert result.proposed_utc_offset_hours == 0
    assert result.timezone_shift_hours == 0
    assert result.compliance_rate == 1


@pytest.mark.parametrize(("coverage_type", "expected"), [("population", 1), ("longitude", 1), ("area", 2)])
@pytest.mark.parametrize("reverse", [False, True])
def test_earliest_feasible_offset_uses_constraints_not_area_alignment(
    coverage_type: CoverageType, expected: float, *, reverse: bool
) -> None:
    samples = (
        TerritorySample(0, 0, area_weight=1, longitude_weight=9, population_weight=90),
        TerritorySample(0, 25, area_weight=100, longitude_weight=1, population_weight=10),
    )
    territory = _territory(samples=tuple(reversed(samples)) if reverse else samples)
    config = ExperimentConfig(
        solar_metric="median", coverage_type=coverage_type, coverage_threshold=0.9, utc_offsets=(4, 2, 1.5, 1)
    )
    result = optimize_territory(territory, config)
    assert result.proposed_utc_offset_hours == expected
    assert result.compliance_rate >= 0.9
    if coverage_type == "population":
        assert result.population_covered == pytest.approx(0.9)
        changed_areas = replace(
            territory, samples=tuple(replace(sample, area_weight=101 - sample.area_weight) for sample in samples)
        )
        assert optimize_territory(changed_areas, config).proposed_utc_offset_hours == expected


@pytest.mark.parametrize("maximum", [None, "12:00"])
def test_earliest_feasible_offset_handles_fractional_candidates_without_a_minimum(maximum: str | None) -> None:
    config = ExperimentConfig(solar_metric="median", minimum_noon=None, maximum_noon=maximum, utc_offsets=(1.5, 0.5))
    result = optimize_territory(_territory(longitude=15), config)
    assert result.proposed_utc_offset_hours == 0.5


@pytest.mark.parametrize("reverse", [False, True])
def test_earliest_feasible_noon_does_not_wrap_late_population_noon_to_midnight(*, reverse: bool) -> None:
    samples = (
        TerritorySample(0, 110, area_weight=100, population_weight=0),
        TerritorySample(0, 145, area_weight=1, population_weight=100),
    )
    territory = _territory(samples=tuple(reversed(samples)) if reverse else samples, current_timezone="UTC+10")
    config = ExperimentConfig(solar_metric="median", coverage_type="population", coverage_threshold=0.9)
    result = optimize_territory(territory, config)
    assert result.proposed_utc_offset_hours == 10
    assert result.population_covered == 1


def test_default_area_coverage_accepts_exactly_90_percent_and_preserves_overrides() -> None:
    territory = _territory(
        samples=(
            TerritorySample(0, 0, area_weight=9),
            TerritorySample(0, 30, area_weight=1),
        ),
    )
    config = ExperimentConfig(utc_offsets=(1, 3))
    result = optimize_territory(territory, config)
    assert result.proposed_utc_offset_hours == 1
    assert result.compliance_rate == pytest.approx(0.9)
    assert result.coverage_type_used == "area"
    assert optimize_territory(territory, replace(config, coverage_threshold=1)).proposed_utc_offset_hours == 3


@pytest.mark.parametrize(
    ("longitude", "expected_offset"),
    [(-82.563574, -5), (-77.195996, -5), (-74, -4), (139.7, 10), (-157, -10), (179, 12), (-179, -11)],
)
def test_default_offsets_follow_geography_and_keep_median_noon_near_midday(
    longitude: float, expected_offset: float
) -> None:
    config = ExperimentConfig()
    result = optimize_territory(_territory(longitude=longitude), config)
    assert result.proposed_utc_offset_hours == expected_offset
    assert 720 <= result.area_weighted_noon_minutes < 780
    assert result.compliance_rate == 1


def test_earliest_feasible_offset_unwraps_the_dateline() -> None:
    config = ExperimentConfig(coverage_type="area")
    dateline = _territory(samples=(TerritorySample(0, 179), TerritorySample(0, -179)))
    assert optimize_territory(dateline, config).proposed_utc_offset_hours == 13


@pytest.mark.parametrize("current_timezone", [None, "UTC-09:00"])
@pytest.mark.parametrize("reverse", [False, True])
def test_dateline_outlier_uses_full_territory_calendar_date(current_timezone: str | None, *, reverse: bool) -> None:
    samples = (
        TerritorySample(52, 179.7734, area_weight=0.01),
        TerritorySample(64, -152, area_weight=10),
        TerritorySample(55, -129.9942, area_weight=0),
    )
    territory = _territory(samples=tuple(reversed(samples)) if reverse else samples, current_timezone=current_timezone)
    result = optimize_territory(territory, LONGITUDE_95_IN_2025)
    assert result.proposed_utc_offset_hours == -8
    assert result.area_weighted_noon_minutes == pytest.approx(848.7334727188363)
    assert result.compliance_rate == 1
    if current_timezone is not None:
        assert result.timezone_shift_hours == 1


@pytest.mark.parametrize(
    ("samples", "current_timezone", "expected_offset"),
    [
        ((TerritorySample(0, 178, area_weight=10), TerritorySample(0, -178)), None, 13),
        ((TerritorySample(0, 178, area_weight=10), TerritorySample(0, -178)), "UTC+12", 13),
        ((TerritorySample(0, 174), TerritorySample(0, -157, area_weight=10)), "UTC+14", 14),
        ((TerritorySample(0, 179, area_weight=0), TerritorySample(0, -150, area_weight=0)), None, -10),
    ],
)
def test_dateline_calendar_reference_preserves_pacific_dates_and_handles_zero_area(
    samples: tuple[TerritorySample, ...], current_timezone: str | None, expected_offset: float
) -> None:
    result = optimize_territory(_territory(samples=samples, current_timezone=current_timezone), LONGITUDE_95_IN_2025)
    assert result.proposed_utc_offset_hours == expected_offset
    assert result.compliance_rate == 1


@pytest.mark.parametrize(
    ("longitude", "current_timezone", "expected_offset"),
    [
        (-175, "UTC+13", 13),
        (-178, "UTC+12", 13),
        (-172, "UTC+13", 13),
        (-157, "UTC+14", 14),
        (179, "UTC-11", -12),
    ],
)
def test_calendar_reference_preserves_civil_date_without_crossing_geometry(
    longitude: float, current_timezone: str, expected_offset: float
) -> None:
    config = ExperimentConfig()
    territory = _territory(longitude=longitude, current_timezone=current_timezone)
    unanchored = optimize_territory(replace(territory, current_timezone=None), config)
    result = optimize_territory(territory, config)
    assert result.proposed_utc_offset_hours == expected_offset
    assert result.timezone_shift_hours is not None
    assert abs(result.timezone_shift_hours) <= 1
    assert result.area_weighted_noon_minutes == pytest.approx(unanchored.area_weighted_noon_minutes)
    assert result.compliance_rate == unanchored.compliance_rate == 1


def test_optimizer_falls_back_to_maximum_coverage_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    territory = _territory(
        samples=(TerritorySample(0, 0), TerritorySample(0, 4)),
        longitude=0,
        current_timezone="UTC+00:00",
    )
    config = ExperimentConfig(
        minimum_noon="12:00",
        maximum_noon="12:30",
        coverage_threshold=1,
        utc_offsets=(1, 0),
    )
    result = optimize_territory(territory, config)
    assert result.proposed_utc_offset_hours == 0
    assert result.compliance_rate == 0.5
    assert result.fallback_used
    assert "No UTC offset meets 100.0% area coverage" in caplog.text
    assert "using UTC+00:00 with 50.0% maximum coverage" in caplog.text


def test_optimizer_fallback_ties_use_existing_earliest_noon_order(caplog: pytest.LogCaptureFixture) -> None:
    result = optimize_territory(
        _territory(),
        ExperimentConfig(minimum_noon="23:00", coverage_threshold=1, utc_offsets=(1, 0)),
    )
    assert result.proposed_utc_offset_hours == 0
    assert result.compliance_rate == 0
    assert result.fallback_used
    assert "50.0%" not in caplog.text


@pytest.mark.parametrize(
    ("timezone", "expected"),
    [
        ("America/New_York", "America/New_York (standard: UTC-05:00; DST: UTC-04:00)"),
        ("Australia/Lord_Howe", "Australia/Lord_Howe (standard: UTC+10:30; DST: UTC+11:00)"),
        ("Africa/Casablanca", "Africa/Casablanca (standard: UTC+01:00; negative DST: UTC+00:00)"),
        ("America/Costa_Rica", "America/Costa_Rica (UTC-06:00; no DST)"),
        ("Etc/UTC", "Etc/UTC (UTC+00:00; no DST)"),
        ("UTC+05:45", "UTC+05:45 (DST information unavailable)"),
        (None, "Not available"),
    ],
)
def test_timezone_descriptions_include_actual_seasonal_offsets(timezone: str | None, expected: str) -> None:
    assert describe_timezone(timezone, year=2025) == expected


@pytest.mark.parametrize("timezone", ["Not/A_Zone", "UTC+14:30"])
def test_timezone_descriptions_report_invalid_timezone_data(timezone: str) -> None:
    with pytest.raises(ValueError, match=r"Unknown IANA|Invalid UTC offset"):
        describe_timezone(timezone, year=2026)


def test_plotly_hover_shows_actual_standard_and_dst_offsets() -> None:
    territories, results = _map_inputs((0, None))
    for index, timezone in enumerate(("America/New_York", None)):
        results[index] = replace(
            results[index],
            current_timezone=timezone,
            noon=calculate_noon_statistics(
                replace(territories[index], current_timezone=timezone),
                results[index].proposed_utc_offset_hours,
                year=2025,
            ),
        )
    figure = build_plotly_figure(territories, results, MAP_FIELDS[0])
    trace = figure.data[0]
    year = 2025
    assert trace.customdata[0][0] == describe_timezone("America/New_York", year=year)
    assert "standard: UTC-05:00" in trace.customdata[0][0]
    assert "DST: UTC-04:00" in trace.customdata[0][0]
    assert trace.customdata[1][0] == "Not available"
    assert trace.customdata[0][1] == results[0].proposed_timezone
    assert trace.customdata[0][5] == year
    assert "Actual (%{customdata[5]}): %{customdata[0]}" in trace.hovertemplate
    assert "Proposed: %{customdata[1]}" in trace.hovertemplate
    details = trace.customdata[0][4]
    assert f"Annual noon extremes ({year})" in details
    assert "Standard time (UTC-05:00): earliest" in details
    assert "DST (UTC-04:00): earliest" in details
    assert "representative timezone across territory" in details
    assert "Current noon range unavailable" in trace.customdata[1][4]
    assert "%{customdata[4]}" in trace.hovertemplate


@pytest.mark.parametrize(
    ("timezone", "west", "east"),
    [
        ("America/New_York", -80, -70),
        ("Australia/Lord_Howe", 159, 160),
        ("Europe/Dublin", -10, -6),
        ("Pacific/Apia", -172, -171),
    ],
)
def test_current_noon_ranges_match_daily_events_in_each_season(timezone: str, west: float, east: float) -> None:
    year = 2024
    zone = ZoneInfo(timezone)
    expected: dict[str, list[float]] = {}
    start = datetime(year, 1, 1, tzinfo=UTC)
    for day in range(-2, 368):
        midnight = start + timedelta(days=day)
        index = midnight.timetuple().tm_yday - 1
        for longitude in (west, east):
            noon = _daily_noon(longitude, midnight.year)[index]
            local = (midnight + timedelta(minutes=noon)).astimezone(zone)
            if local.year != year:
                continue
            daylight, offset = local.dst(), local.utcoffset()
            assert daylight is not None
            assert offset is not None
            state = (
                "Standard time" if daylight == timedelta(0) else "DST" if daylight > timedelta(0) else "Negative DST"
            )
            key = f"{state} ({format_utc_offset(offset.total_seconds() / 3600)})"
            expected.setdefault(key, []).append(noon + offset.total_seconds() / 60)
    actual = current_noon_ranges(timezone, year=year, west=west, east=east)
    assert actual.keys() == expected.keys()
    for key, values in expected.items():
        assert actual[key] == pytest.approx((min(values), max(values)), abs=1e-10)
    if timezone == "America/New_York":
        fixed = next(iter(current_noon_ranges("UTC-04:00", year=year, west=west, east=east).values()))
        assert actual["DST (UTC-04:00)"][1] < fixed[1]


@pytest.mark.parametrize(
    ("west", "east", "offset"),
    [(-85, -82, -5), (170, -170, 12), (170, -170, -12), (-179, -178, 0)],
)
def test_noon_hover_uses_full_boundaries_and_dateline_frame(west: float, east: float, offset: float) -> None:
    geometry: dict[str, object] = {
        "type": "Polygon",
        "coordinates": [[[west, 0], [east, 0], [east, 1], [west, 1], [west, 0]]],
    }
    territory = _territory(geometry=geometry, samples=(TerritorySample(0, 0),))
    result = optimize_territory(territory, ExperimentConfig(minimum_noon=None, utc_offsets=(offset,)))
    result = replace(
        result,
        current_timezone="UTC+05:45",
        noon=calculate_noon_statistics(replace(territory, current_timezone="UTC+05:45"), offset, year=result.noon.year),
    )
    figure = build_plotly_figure([territory], [result], MAP_FIELDS[0])
    details = figure.data[0].customdata[0][4]
    year = datetime.now(UTC).year
    unwrapped = [longitude + 360 * round((offset * 15 - longitude) / 360) for longitude in (west, east)]
    eastern = _daily_noon((max(unwrapped) + 180) % 360 - 180, year, offset)
    western = _daily_noon((min(unwrapped) + 180) % 360 - 180, year, offset)
    east_adjustment = -4 * (max(unwrapped) - ((max(unwrapped) + 180) % 360 - 180))
    west_adjustment = -4 * (min(unwrapped) - ((min(unwrapped) + 180) % 360 - 180))
    expected_earliest = maps_module._noon_clock(min(eastern) + east_adjustment)
    expected_latest = maps_module._noon_clock(max(western) + west_adjustment)
    assert f"Proposed: earliest {expected_earliest}; latest {expected_latest}" in details
    assert "Fixed offset UTC+05:45 (DST information unavailable)" in details
    if offset == 0:
        assert "(+1 day)" in details


def test_timezone_offsets_and_boundary_lookup(tmp_path: Path) -> None:
    assert parse_utc_offset("UTC-05:30") == -5.5
    assert parse_utc_offset("UTC+14") == 14
    assert format_utc_offset(-5.5) == "UTC-05:30"
    with pytest.raises(ValueError, match="Invalid UTC offset"):
        parse_utc_offset("UTC+14:30")
    timezone_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"tzid": "Etc/UTC"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]],
                },
            },
        ],
    }
    path = tmp_path / "timezones.json"
    path.write_text(json.dumps(timezone_geojson), encoding="utf-8")
    index = TimezoneBoundaryIndex(load_timezone_boundaries(path))
    assert index.timezone_at(0, 0) == "Etc/UTC"
    annotated = annotate_current_timezones([_territory(current_timezone=None)], index)
    assert annotated[0].current_timezone == "Etc/UTC"
    assert annotate_current_timezones([_territory(longitude=20)], index)[0].current_timezone == "UTC+00:00"
    with pytest.raises(ValueError, match="At least one"):
        TimezoneBoundaryIndex([])
    with pytest.raises(ValueError, match="Unknown IANA"):
        parse_utc_offset("Not/AZone")


def test_runner_csv_and_progress(tmp_path: Path) -> None:
    territories = [
        _territory("AAA:0", -20, current_timezone="UTC+00:00"),
        _territory("BBB:0", 10, current_timezone="UTC+01:00"),
    ]
    config = ExperimentConfig(utc_offsets=(1, 2), minimum_noon=None, workers=1)
    progress: list[tuple[int, int, str]] = []
    results = run_experiment(config, territories, on_territory_done=lambda *values: progress.append(values))
    assert progress == [(1, 2, "AAA:0"), (2, 2, "BBB:0")]
    path = write_results_csv(results, tmp_path / "results.csv")
    assert path.read_text(encoding="utf-8").splitlines()[0].split(",") == [
        "territory_id",
        "country_code",
        "territory_name",
        "admin_level",
        "current_timezone",
        "proposed_timezone",
        "proposed_utc_offset_hours",
        "timezone_shift_hours",
        "area_weighted_noon_minutes",
        "compliance_rate",
        "population_covered",
        "coverage_type_used",
        "fallback_used",
        "solar_year",
        "noon_year",
        "proposed_earliest_noon_minutes",
        "proposed_latest_noon_minutes",
        "current_timezone_description",
        "current_noon_ranges",
        "median_equation_of_time_minutes",
    ]
    assert run_experiment(config, territories[:1])[0].country_code == "AAA"


def test_runner_rejects_empty_or_duplicate_territories() -> None:
    config = ExperimentConfig(utc_offsets=(1,), minimum_noon=None)
    with pytest.raises(ValueError, match="At least one"):
        run_experiment(config, [])
    territory = _territory()
    with pytest.raises(ValueError, match="unique"):
        run_experiment(config, [territory, territory])


def test_runner_progress_does_not_log_each_optimization(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    territories = [_territory("AAA:0"), _territory("BBB:0")]
    results = run_experiment(ExperimentConfig(utc_offsets=(1,), minimum_noon=None, workers=1), territories)
    assert len(results) == 2
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    assert "0/2" in lines[0]
    assert "2/2" in lines[1]
    assert sum(record.message.startswith("Optimizing ") for record in caplog.records) == 1


def test_assignment_result_has_expected_output_shape() -> None:
    result = optimize_territory(_territory(), ExperimentConfig(utc_offsets=(1,), minimum_noon=None))
    assert isinstance(result, AssignmentResult)
    assert result.country_code == "TST"
    assert result.proposed_timezone == "UTC+01:00"


def test_download_script_supported_sources_and_url_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b'{"type":"FeatureCollection","features":[]}'
    original_download = download_boundaries.download_bytes
    monkeypatch.setattr(download_boundaries, "download_bytes", lambda url: payload)
    result = download_boundaries.download("gadm", "usa", 1, tmp_path)
    assert result.read_bytes() == payload
    with pytest.raises(ValueError, match="Unsupported"):
        download_boundaries.download("unknown", "USA", 0, tmp_path)
    with pytest.raises(ValueError, match="must use HTTPS"):
        original_download("http://example.invalid/data.json")
    calls = iter((b'{"gjDownloadURL":"https://raw.githubusercontent.com/data.geojson"}', payload))
    monkeypatch.setattr(download_boundaries, "download_bytes", lambda url: next(calls))
    assert download_boundaries.download("geoboundaries", "USA", 1, tmp_path).read_bytes() == payload
    with pytest.raises(ValueError, match="Administrative level"):
        download_boundaries.download("gadm", "USA", -1, tmp_path)


@pytest.mark.parametrize("admin_level", [0, 1])
def test_geometry_only_loading_preserves_grouping_without_sampling(
    admin_level: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig(subdivision_rules={"TST": admin_level}, sample_grid_size=2)
    expected = {territory.territory_id: territory.geometry for territory in load_territories(TOY_BOUNDARIES, config)}

    def no_sampling(*_args: object) -> None:
        pytest.fail("Geometry-only loading must not sample territories.")

    monkeypatch.setattr(boundaries_module, "sample_geometry", no_sampling)
    assert load_territory_geometries(TOY_BOUNDARIES, config) == expected


def test_saved_result_csv_round_trip(tmp_path: Path) -> None:
    assignment = optimize_territory(
        _territory(current_timezone="America/New_York"), ExperimentConfig(minimum_noon=None, utc_offsets=(1,))
    )
    results = [
        replace(assignment, proposed_utc_offset_hours=5.75, timezone_shift_hours=1.25, population_covered=0),
        replace(
            assignment,
            territory_id="TST:2",
            territory_name="North, coast",
            current_timezone=None,
            timezone_shift_hours=None,
            noon=calculate_noon_statistics(_territory(current_timezone=None), 1, year=2024),
        ),
    ]
    path = write_results_csv(results, tmp_path / "results.csv")
    assert read_results_csv(path) == results
    path.write_text(dashboard._csv_text(results), encoding="utf-8")
    assert read_results_csv(path) == results


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("columns", "missing columns"),
        ("empty", "no territory assignments"),
        ("duplicate", "line 3.*unique"),
        ("number", "line 2.*could not convert"),
        ("nonfinite", "line 2.*must be finite"),
        ("short", "line 2.*Incorrect number"),
        ("noon_year", "line 2.*valid year"),
        ("noon_nonfinite", "line 2.*finite and ordered"),
        ("noon_json", "line 2.*JSON object"),
        ("noon_bounds", "line 2.*minute pairs"),
        ("noon_missing_current", "line 2.*exactly when"),
        ("noon_median", "line 2.*Median equation of time must be finite"),
    ],
)
def test_saved_result_csv_rejects_invalid_data(tmp_path: Path, problem: str, message: str) -> None:
    assignment = optimize_territory(_territory(), ExperimentConfig(minimum_noon=None, utc_offsets=(1,)))
    path = write_results_csv([assignment], tmp_path / "results.csv")
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    if problem == "columns":
        rows = [["territory_id"], ["TST:1"]]
    elif problem == "empty":
        rows = rows[:1]
    elif problem == "duplicate":
        rows.append(rows[1])
    elif problem == "short":
        rows[1].pop()
    elif problem.startswith("noon_"):
        column, value = {
            "noon_year": ("noon_year", "0"),
            "noon_nonfinite": ("proposed_earliest_noon_minutes", "nan"),
            "noon_json": ("current_noon_ranges", "[]"),
            "noon_bounds": ("current_noon_ranges", '{"Standard time": [720, 700]}'),
            "noon_missing_current": ("current_noon_ranges", "{}"),
            "noon_median": ("median_equation_of_time_minutes", "inf"),
        }[problem]
        rows[1][rows[0].index(column)] = value
    else:
        rows[1][rows[0].index("proposed_utc_offset_hours")] = "nan" if problem == "nonfinite" else "invalid"
    with path.open("w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(rows)
    with pytest.raises(ValueError, match=message):
        read_results_csv(path)


def test_saved_result_maps_match_experiment_maps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ExperimentConfig(subdivision_rules={"TST": 1}, sample_grid_size=2, workers=1)
    territories = load_territories(TOY_BOUNDARIES, config)
    results = run_experiment(config, territories, on_territory_done=lambda *_: None)
    csv_path = write_results_csv(results, tmp_path / "results.csv")
    expected = export_maps(result_feature_collection(territories, results), tmp_path / "original")

    def no_calculation(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Saved-result maps must use persisted noon statistics, not recompute astronomy.")

    monkeypatch.setattr(solar_module, "equation_of_time_minutes", no_calculation)
    monkeypatch.setattr(noon_module, "equation_of_time_minutes", no_calculation)
    monkeypatch.setattr(timezones_module, "current_noon_ranges", no_calculation)
    monkeypatch.setattr(timezones_module, "describe_timezone", no_calculation)
    monkeypatch.setattr(optimizer_module, "calculate_noon_statistics", no_calculation)
    # Also exercise rendering in this process, where the guards are installed.
    saved_results = read_results_csv(csv_path)
    saved_figure = build_plotly_figure(territories, saved_results, MAP_FIELDS[0])
    assert str(results[0].noon.year) in saved_figure.data[0].customdata[0][4]
    saved_median_figure = build_plotly_figure(territories, saved_results, "median_solar_noon")
    assert len(set(saved_median_figure.data[0].z)) > 1
    actual = export_maps(
        geometry_feature_collection(load_territory_geometries(TOY_BOUNDARIES, config), saved_results),
        tmp_path / "saved",
        MapOptions(workers=2),
    )
    assert [path.name for path in actual] == [path.name for path in expected]
    for original, saved in zip(expected, actual, strict=True):
        if saved.suffix == ".png":
            assert saved.read_bytes() == original.read_bytes()
        elif saved.suffix == ".html":
            assert _html_map_data(saved) == _html_map_data(original)
        else:
            assert b"<svg" in saved.read_bytes()


def test_saved_result_maps_reject_unmatched_territories(tmp_path: Path) -> None:
    territories, results = _map_inputs((1,))
    geometries = {
        territory.territory_id: territory.geometry for territory in territories if territory.geometry is not None
    }
    with pytest.raises(ValueError, match=r"No boundary geometry matches.*unmapped"):
        export_maps(geometry_feature_collection(geometries, results), tmp_path / "maps")
    assert not (tmp_path / "maps").exists()


@pytest.mark.parametrize("clockwise", [False, True])
def test_plotly_winding_preserves_polygon_holes_without_mutating_sources(*, clockwise: bool) -> None:

    outer = [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]
    hole = [[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]
    if clockwise:
        outer.reverse()
        hole.reverse()
    geometry: dict[str, object] = {"type": "Polygon", "coordinates": [outer, hole]}
    original = deepcopy(geometry)
    territory = _territory(geometry=geometry)
    result = optimize_territory(territory, ExperimentConfig(minimum_noon=None, utc_offsets=(1,)))
    figure = build_plotly_figure([territory], [result], MAP_FIELDS[0])
    rendered = shape(figure.data[0].geojson["features"][0]["geometry"])
    assert isinstance(rendered, Polygon)
    assert not rendered.exterior.is_ccw
    assert rendered.interiors[0].is_ccw
    assert rendered.equals(shape(original))
    assert geometry == original
    assert len(figure.data) == 5


@pytest.mark.parametrize("dateline", [False, True])
def test_plotly_removes_degenerate_components_and_handles_dateline_winding(
    caplog: pytest.LogCaptureFixture, *, dateline: bool
) -> None:

    west, east = (170, -170) if dateline else (0, 20)
    outer = [[west, 0], [east, 0], [east, 10], [west, 10], [west, 0]]
    degenerate = [[west, 0]] * 4
    geometry: dict[str, object] = {"type": "MultiPolygon", "coordinates": [[degenerate], [outer, degenerate]]}
    original = deepcopy(geometry)
    territory = _territory(geometry=geometry)
    result = optimize_territory(territory, ExperimentConfig(minimum_noon=None, utc_offsets=(1,)))
    figure = build_plotly_figure([territory], [result], MAP_FIELDS[0])
    coordinates = figure.data[0].geojson["features"][0]["geometry"]["coordinates"]
    assert len(coordinates) == len(coordinates[0]) == 1
    ring = [(lon + 360 if dateline and lon < 0 else lon, lat) for lon, lat in coordinates[0][0]]
    polygon = Polygon(ring)
    assert not polygon.exterior.is_ccw
    assert polygon.area == 200
    assert geometry == original
    assert "Skipped 2 zero-area rings" in caplog.text


def test_plotly_rejects_entirely_degenerate_geometry() -> None:
    territory = _territory(geometry={"type": "Polygon", "coordinates": [[[1, 1]] * 4]})
    result = optimize_territory(_territory(), ExperimentConfig(minimum_noon=None, utc_offsets=(1,)))
    with pytest.raises(ValueError, match="nonzero-area polygon"):
        build_plotly_figure([territory], [result], MAP_FIELDS[0])


def test_map_exports(tmp_path: Path) -> None:
    territories = load_territories(
        TOY_BOUNDARIES,
        ExperimentConfig(subdivision_rules={"TST": 1}, sample_grid_size=2),
    )
    comparison = run_experiment(
        ExperimentConfig(minimum_noon=None, utc_offsets=(1,)),
        territories,
        on_territory_done=lambda *_: None,
    )
    figure = build_plotly_figure(territories, comparison, MAP_FIELDS[0])
    assert figure.data[0].type == "choropleth"
    assert figure.layout.geo.fitbounds is False
    meridians, boundaries, labels, country_borders = figure.data[1:]
    assert country_borders.name == "Country borders"
    assert country_borders.line.width > max(figure.data[0].marker.line.width)
    assert country_borders.line.color != figure.data[0].marker.line.color
    assert {longitude for longitude in meridians.lon if longitude is not None} == set(range(-180, 181, 15))
    assert {longitude for longitude in boundaries.lon if longitude is not None} == {
        longitude + 7.5 for longitude in range(-180, 180, 15)
    }
    assert meridians.line.dash == "dot"
    assert boundaries.line.dash == "dash"
    assert list(labels.lon) == list(range(-180, 180, 15))
    assert list(labels.text) == ["UTC-12/+12"] + [f"UTC{offset:+d}" for offset in range(-11, 12)]
    assert all(trace.hoverinfo == "skip" and not trace.showlegend for trace in figure.data[1:])
    static_paths = export_maps(
        result_feature_collection(territories, comparison), tmp_path / "static", kinds=("static",)
    )
    assert {path.name for path in static_paths} == {
        f"{field}.{suffix}" for field in MAP_FIELDS for suffix in ("png", "svg")
    }
    export_maps(result_feature_collection(territories, comparison), tmp_path / "interactive", kinds=("html",))
    assert (tmp_path / "interactive" / f"{MAP_FIELDS[0]}.html").exists()
    assert (tmp_path / "interactive" / "plotly.min.js").exists()
    with pytest.raises(ValueError, match="Unsupported map statistic"):
        build_plotly_figure(territories, comparison, "unknown")


def test_static_rasterization_applies_to_every_map_fill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    territories, results = _map_inputs((0,))
    rasterized_fields: list[str] = []

    def capture(figure: Figure, path: Path, **_kwargs: object) -> None:
        assert figure.axes[0].collections[0].get_rasterized()
        rasterized_fields.append(path.stem)

    monkeypatch.setattr(Figure, "savefig", capture)
    export_maps(
        result_feature_collection(territories, results),
        tmp_path,
        MapOptions(rasterize_static_map_fills=True),
        kinds=("static",),
    )
    assert set(rasterized_fields) == set(MAP_FIELDS)


def test_final_render_cache_reuses_exact_outputs_and_rejects_changed_or_corrupt_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "TST:1",
                "properties": {
                    "territory_id": "TST:1",
                    "country_code": "TST",
                    "proposed_utc_offset_hours": 1,
                    "fallback_used": False,
                },
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            }
        ],
    }
    calls = 0
    # Only output reuse is under test here; layers are covered by the feature-cache test.
    monkeypatch.setattr(maps_module, "_with_map_layers", lambda collection, *_args: collection)

    def fake_results(
        inputs: dict[str, Any], output: Path, requested: list[MapJob], _workers: int
    ) -> Iterator[tuple[int, list[Path]]]:
        nonlocal calls
        calls += 1
        output.mkdir(parents=True, exist_ok=True)
        for position, (field, kind) in enumerate(requested):
            assert kind == "html"
            path = output / f"{field}.html"
            path.write_bytes(json.dumps(inputs["features"]).encode())
            yield position, [path]

    def export(source: dict[str, Any], workers: int) -> list[Path]:
        return export_maps(source, tmp_path, MapOptions(workers=workers), kinds=("html",))

    monkeypatch.setattr(maps_module, "_map_results", fake_results)
    first = export(collection, 2)
    assert calls == 1
    assert (tmp_path / maps_module._MAP_CACHE_MANIFEST).is_file()
    assert export(collection, 4) == first
    assert calls == 1

    first[0].write_text("corrupt", encoding="utf-8")
    export(collection, 1)
    assert calls == 2

    changed = deepcopy(collection)
    changed["features"][0]["properties"]["proposed_utc_offset_hours"] = 2
    export(changed, 1)
    assert calls == 3


def test_map_feature_cache_reuses_borders_and_values_for_similar_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "TST:1",
                "properties": {
                    "territory_id": "TST:1",
                    "country_code": "TST",
                    "proposed_utc_offset_hours": 1,
                    "solar_year": 2025,
                    "compliance_rate": 1,
                    "noon": {"median_equation_of_time_minutes": -0.5},
                },
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            }
        ],
    }
    calls: dict[str, int] = {"borders": 0, "median_noon": 0, "solar_events": 0}
    monkeypatch.setattr(
        maps_module,
        "_country_borders",
        lambda *_args, **_kwargs: calls.__setitem__("borders", calls["borders"] + 1) or [],
    )

    def builder(kind: str) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
        def build(_collection: dict[str, Any]) -> list[dict[str, Any]]:
            calls[kind] += 1
            return [
                {
                    "type": "Feature",
                    "id": kind,
                    "properties": {"value": calls[kind]},
                    "geometry": collection["features"][0]["geometry"],
                }
            ]

        return build

    for kind in ("median_noon", "solar_events"):
        monkeypatch.setitem(maps_module._GRADIENT_BUILDERS, kind, builder(kind))
    monkeypatch.setattr(
        maps_module,
        "_map_results",
        lambda _collection, output, requested, _workers: iter(
            (index, [output / f"{field}.{kind}"]) for index, (field, kind) in enumerate(requested)
        ),
    )
    cache = tmp_path / "cache"

    def with_options(source: dict[str, Any], *, cell: float = 0.5, simplify: float = 0.01) -> dict[str, Any]:
        options = MapOptions(
            cache_directory=cache, event_map_cell_degrees=cell, map_geometry_simplification_degrees=simplify
        )
        return {**source, "options": options}

    export_maps(collection, tmp_path / "first", MapOptions(cache_directory=cache), kinds=("html",))
    similar = deepcopy(collection)
    similar["features"][0]["properties"]["compliance_rate"] = 0.75
    export_maps(similar, tmp_path / "second", MapOptions(cache_directory=cache), kinds=("html",))
    # Country and territory outlines once each; display-only changes reuse every layer.
    assert calls == {"borders": 2, "median_noon": 1, "solar_events": 1}

    first = maps_module._cached_gradient_features(with_options(collection), "median_noon")
    changed_display_only = deepcopy(collection)
    changed_display_only["features"][0]["properties"]["compliance_rate"] = 0.5
    assert maps_module._cached_gradient_features(with_options(changed_display_only), "median_noon") == first
    assert calls["median_noon"] == 1
    median_cache = next((cache / "map-features" / "median_noon").glob("*.json.zlib"))
    median_cache.write_bytes(b"corrupt")
    maps_module._cached_gradient_features(with_options(collection), "median_noon")
    assert calls["median_noon"] == 2
    assert maps_module._cached_gradient_features(with_options(collection), "solar_events")
    maps_module._cached_gradient_features(with_options(collection, cell=1.0), "solar_events")
    assert calls["solar_events"] == 2
    maps_module._cached_gradient_features(with_options(collection, simplify=0.02), "solar_events")
    assert calls["solar_events"] == 3
    changed_year = deepcopy(collection)
    changed_year["features"][0]["properties"]["solar_year"] = 2026
    maps_module._cached_gradient_features(with_options(changed_year), "solar_events")
    assert calls["solar_events"] == 4


@pytest.mark.parametrize("workers", [1, 2])
def test_country_borders_dissolve_internal_edges_but_keep_international_edges(workers: int) -> None:

    territories = [
        _territory(
            identifier,
            geometry={
                "type": "Polygon",
                "coordinates": [[[west, 0], [east, 0], [east, 2], [west, 2], [west, 0]]],
            },
        )
        for identifier, west, east in (("AAA:1", 0, 1), ("AAA:2", 1, 2), ("BBB:1", 2, 3))
    ]
    config = ExperimentConfig(minimum_noon=None, utc_offsets=(0,))
    results = [optimize_territory(territory, config) for territory in territories]
    original = deepcopy([territory.geometry for territory in territories])
    collection = result_feature_collection(territories, results)
    lines = maps_module._country_borders(collection, workers=workers)
    boundary = MultiLineString(lines)
    assert boundary.intersection(LineString([(1, 0.1), (1, 1.9)])).is_empty
    assert boundary.intersection(LineString([(2, 0.1), (2, 1.9)])).length == pytest.approx(1.8)
    assert [territory.geometry for territory in territories] == original


def test_country_borders_join_dateline_subdivisions_without_cross_map_segments() -> None:
    geometries = [
        {
            "type": "Polygon",
            "coordinates": [[[west, -5], [east, -5], [east, 5], [west, 5], [west, -5]]],
        }
        for west, east in ((170, 180), (-180, -170))
    ]
    lines = maps_module._country_lines(geometries)
    assert lines
    for line in lines:
        assert all(-180 <= longitude <= 180 for longitude, _ in line)
        for first, second in pairwise(line):
            assert abs(second[0] - first[0]) <= 10
            assert not (abs(first[0]) == abs(second[0]) == 180 and first[1] != second[1])


def test_country_borders_remove_overlaps_and_preserve_holes() -> None:

    geometries = [
        {
            "type": "Polygon",
            "coordinates": [
                [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
                [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]],
            ],
        },
        {"type": "Polygon", "coordinates": [[[3, 0], [5, 0], [5, 4], [3, 4], [3, 0]]]},
    ]
    boundary = MultiLineString(maps_module._country_lines(geometries))
    assert boundary.intersection(LineString([(3, 0.1), (3, 3.9)])).is_empty
    assert boundary.intersection(LineString([(4, 0.1), (4, 3.9)])).is_empty
    assert boundary.intersection(LineString([(1, 1), (2, 1)])).length == 1


@pytest.mark.parametrize("gap", [0.001, 1e-9])
def test_country_borders_remove_enclosed_subdivision_seams_without_removing_small_islands(gap: float) -> None:

    hole = [(0.2, 1.5), (0.4, 1.5), (0.4, 2), (0.2, 2), (0.2, 1.5)]
    geometries = [
        {
            "type": "Polygon",
            "coordinates": [
                [[0, 0], [1, 0], [1, 4], [0, 4], [0, 0]],
                hole,
            ],
        },
        mapping(box(1 + gap, 0, 3, 4)),
        mapping(box(0, 0, 3, 1)),
        mapping(box(0, 3, 3, 4)),
        mapping(box(5, 0, 5 + gap, gap)),
    ]
    original = deepcopy(geometries)
    boundary = MultiLineString(maps_module._country_lines(geometries))
    expected = MultiLineString(
        [
            [(0, 0), (3, 0), (3, 4), (0, 4), (0, 0)],
            hole,
            list(box(5, 0, 5 + gap, gap).exterior.coords),
        ],
    )
    assert boundary.equals(expected)
    assert geometries == original


@pytest.mark.parametrize(("offset", "year"), [(0, 2024), (5.75, 2025), (-3.5, 2024)])
def test_median_noon_gradient_matches_daily_median_and_preserves_geometry(offset: float, year: int) -> None:

    geometry: dict[str, object] = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [[0, 0], [2, 0], [2, 3], [0, 3], [0, 0]],
                [[0.5, 1], [1.5, 1], [1.5, 2], [0.5, 2], [0.5, 1]],
            ],
            [[[3.01, 0], [3.02, 0], [3.02, 0.01], [3.01, 0.01], [3.01, 0]]],
        ],
    }
    original = deepcopy(geometry)
    territory = _territory(geometry=geometry)
    result = optimize_territory(territory, ExperimentConfig(minimum_noon=None, utc_offsets=(offset,)))
    result = replace(result, noon=calculate_noon_statistics(territory, offset, year=year))
    collection = result_feature_collection([territory], [result])
    bands = maps_module._median_noon_features({**collection, "options": MapOptions(workers=2)})
    assert union_all([shape(band["geometry"]) for band in bands]).equals(shape(geometry))
    assert len({band["id"] for band in bands}) == len(bands)
    for band in bands:
        west, _, east, _ = shape(band["geometry"]).bounds
        longitude = band["properties"]["longitude"]
        value = band["properties"]["median_solar_noon"]
        assert value == pytest.approx(_statistic(_daily_noon(longitude, year, offset), "median"))
        for endpoint in (west, east):
            exact = _statistic(_daily_noon(endpoint, year, offset), "median")
            assert abs(value - exact) <= 0.5 + 1e-10
    assert geometry == original
    figure = build_plotly_figure([territory], [result], "median_solar_noon")
    trace = figure.data[0]
    assert len(set(trace.z)) > 1
    assert trace.marker.line.width == 0
    assert trace.zmin % 60 == trace.zmax % 60 == 0
    assert 540 <= trace.zmin < trace.zmax <= 900
    assert list(trace.z) == [band["properties"]["median_solar_noon"] for band in bands]
    assert all(trace.zmin <= value <= trace.zmax for value in trace.colorbar.tickvals)
    assert "%{customdata[3]}" in trace.hovertemplate
    assert trace.customdata[0][3] == maps_module._noon_clock(trace.z[0])
    assert trace.customdata[0][5] == year
    assert [feature["properties"]["median_solar_noon"] for feature in trace.geojson["features"]] == list(trace.z)
    assert figure.data[1].name == "Territory borders"
    assert figure.data[-1].name == "Country borders"


@pytest.mark.parametrize("offset", [-12, 12, 14])
def test_median_noon_gradient_splits_dateline_without_clock_or_color_seams(offset: float) -> None:

    geometry: dict[str, object] = {
        "type": "Polygon",
        "coordinates": [[[179, 0], [-179, 0], [-179, 1], [179, 1], [179, 0]]],
    }
    territory = _territory(geometry=geometry)
    result = optimize_territory(territory, ExperimentConfig(minimum_noon=None, utc_offsets=(offset,)))
    collection = result_feature_collection([territory], [result])
    bands = maps_module._median_noon_features({**collection, "options": MapOptions(workers=1)})
    expected = union_all([box(179, 0, 180, 1), box(-180, 0, -179, 1)])
    assert union_all([shape(band["geometry"]) for band in bands]).equals(expected)
    medians = []
    for band in bands:
        west, _, east, _ = shape(band["geometry"]).bounds
        assert -180 <= west < east <= 180
        assert east - west <= 0.25
        medians.append(band["properties"]["median_solar_noon"])
    assert max(medians) - min(medians) == pytest.approx(7)
    assert all(710 < value < 850 for value in medians)


@pytest.mark.parametrize("offset", [-4, 0, 0.05, 1, 4])
def test_static_median_noon_uses_same_values_scale_and_clock_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offset: float
) -> None:

    territories, results = _map_inputs((0,))
    results[0] = replace(
        results[0],
        proposed_utc_offset_hours=offset,
        noon=calculate_noon_statistics(territories[0], offset, year=2025),
    )
    interactive = build_plotly_figure(territories, results, "median_solar_noon").data[0]
    assert interactive.zmin % 60 == interactive.zmax % 60 == 0
    captured = []

    def capture(figure: Figure, path: Path, **_kwargs: object) -> None:
        if not path.name.startswith("median_solar_noon"):
            return
        axis = figure.axes[0]
        fill = axis.collections[0]
        observed = fill.get_array()
        assert observed is not None
        np.testing.assert_allclose(sorted(observed), sorted(interactive.z))
        assert fill.norm.vmin == interactive.zmin
        assert fill.norm.vmax == interactive.zmax
        samples = np.linspace(interactive.zmin, interactive.zmax, 9)
        positions = (samples - interactive.zmin) / (interactive.zmax - interactive.zmin)
        scale = [list(stop) for stop in interactive.colorscale]
        expected = np.array([unlabel_rgb(color) for color in sample_colorscale(scale, positions)]) / 255
        np.testing.assert_allclose(np.asarray(fill.cmap(fill.norm(samples)))[:, :3], expected, atol=1 / 128)
        outside = np.array([interactive.zmin - 120, interactive.zmax + 120])
        endpoints = np.array([interactive.zmin, interactive.zmax])
        np.testing.assert_allclose(fill.cmap(fill.norm(outside)), fill.cmap(fill.norm(endpoints)))
        if interactive.zmin < 720 < interactive.zmax:
            np.testing.assert_array_equal(fill.cmap(fill.norm(np.array([720])))[0], (1, 1, 1, 1))
        assert list(figure.axes[1].get_yticks()) == list(interactive.colorbar.tickvals)
        assert [tick.get_text() for tick in figure.axes[1].get_yticklabels()] == list(interactive.colorbar.ticktext)
        assert np.asarray(fill.get_edgecolor()).size == 0
        np.testing.assert_allclose(axis.collections[-2].get_linewidth(), 0.25)
        captured.append(path.suffix)

    monkeypatch.setattr(Figure, "savefig", capture)
    export_maps(result_feature_collection(territories, results), tmp_path, kinds=("static",))
    assert captured == [".png", ".svg"]


@pytest.mark.parametrize(
    ("values", "expected_limits"),
    [
        ([583, 909], (540, 900)),
        ([600, 720, 900], (600, 900)),
        ([750, 780], (720, 780)),
        ([615, 650], (600, 660)),
        ([720, 720], (660, 780)),
        ([780, 780], (720, 840)),
        ([735, 735], (720, 780)),
        ([600, 600], (540, 660)),
        ([1445, 1500], (840, 900)),
        ([-17, 9], (540, 600)),
        ([520, 800], (540, 840)),
        ([480, 1020], (540, 900)),
        ([540, 540], (540, 600)),
        ([900, 900], (840, 900)),
    ],
)
def test_median_noon_scale_rounds_outward_to_hours_and_keeps_white_at_noon(
    values: list[float], expected_limits: tuple[float, float]
) -> None:
    minimum, maximum, colors, bar = maps_module._plotly_scale("median_solar_noon", values)
    assert (minimum, maximum) == expected_limits
    assert bar["tickvals"][0] == minimum
    assert bar["tickvals"][-1] == maximum
    assert all(minimum <= tick <= maximum for tick in bar["tickvals"])
    assert bar["ticktext"] == [maps_module._noon_clock(tick) for tick in bar["tickvals"]]
    if minimum < 720 < maximum:
        white = next(position for position, color in colors if color == "#ffffff")
        assert isinstance(white, (int, float))
        assert minimum + white * (maximum - minimum) == pytest.approx(720)
        assert colors[0][1] == "#2166ac"
        assert colors[-1][1] == "#b35806"
        assert "12:00" in bar["ticktext"]
    elif maximum <= 720:
        assert colors[0][1] == "#2166ac"
        assert colors[-1][1] != "#b35806"
    else:
        assert colors[0][1] != "#2166ac"
        assert colors[-1][1] == "#b35806"


def test_combined_country_maps_hide_internal_subdivision_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    config = ExperimentConfig(subdivision_rules={}, workers=1, sample_grid_size=2)
    territories = load_territories(TOY_BOUNDARIES, config)
    results = run_experiment(config, territories)
    assert len(territories) == len(results) == 1
    for field in maps_module.MAP_FIELDS:
        figure = build_plotly_figure(territories, results, field)
        widths = figure.data[0].marker.line.width
        assert widths == 0 if field not in ("proposed_utc_offset_hours", "timezone_shift_hours") else set(widths) == {0}
        borders = figure.data[-1]
        assert not any(
            first[0] == second[0] == 8 and first[1] != second[1]
            for first, second in pairwise(zip(borders.lon, borders.lat, strict=True))
            if None not in (*first, *second)
        )
    saved = []

    def capture(figure: Figure, path: Path, **_kwargs: object) -> None:
        assert not np.asarray(figure.axes[0].collections[0].get_edgecolor())[:, 3].any()
        saved.append(path.name)

    monkeypatch.setattr(Figure, "savefig", capture)
    export_maps(result_feature_collection(territories, results), tmp_path, kinds=("static",))
    assert len(saved) == 2 * len(MAP_FIELDS)


def _map_inputs(shifts: tuple[float | None, ...]) -> tuple[list[Territory], list[AssignmentResult]]:
    geometry: dict[str, object] = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    }
    territories = [_territory(f"TST:{index}", geometry=geometry) for index in range(len(shifts))]
    config = ExperimentConfig(utc_offsets=(1,), minimum_noon=None)
    results = [
        replace(optimize_territory(territory, config), timezone_shift_hours=shift)
        for territory, shift in zip(territories, shifts, strict=True)
    ]
    results.append(replace(results[0], territory_id="unmapped", timezone_shift_hours=12))
    return territories, results


@pytest.mark.parametrize("interactive", [False, True])
def test_parallel_map_exports_match_serial_outputs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str], *, interactive: bool
) -> None:
    caplog.set_level(logging.INFO)
    territories, results = _map_inputs((-2, 1.25, None))
    kinds: tuple[MapKind, ...] = ("static", "html") if interactive else ("static",)
    jobs = len(MAP_FIELDS) * len(kinds)
    collection = result_feature_collection(territories, results)
    serial = export_maps(collection, tmp_path / "serial", MapOptions(workers=1), kinds=kinds)
    capsys.readouterr()
    parallel = export_maps(collection, tmp_path / "parallel", MapOptions(workers=2), kinds=kinds)
    assert [path.name for path in parallel] == [path.name for path in serial]
    assert len(parallel) == len(MAP_FIELDS) * (3 if interactive else 2)
    for expected, actual in zip(serial, parallel, strict=True):
        assert actual.is_file()
        if actual.suffix == ".png":
            assert actual.read_bytes() == expected.read_bytes()
        elif actual.suffix == ".html":
            assert _html_map_data(actual) == _html_map_data(expected)
        else:
            assert b"<svg" in actual.read_bytes()
    assert f"Exporting {jobs} map jobs with 2 workers" in caplog.text
    lines = capsys.readouterr().err.splitlines()
    assert f"Exporting map jobs: 0/{jobs}" in lines[-2]
    assert f"Exporting map jobs: {jobs}/{jobs} (100.0%; 0 left)" in lines[-1]


def _html_map_data(path: Path) -> tuple[object, object]:
    arguments = path.read_text(encoding="utf-8").rsplit("Plotly.newPlot(", maxsplit=1)[1]
    payload = arguments.split(",", maxsplit=1)[1].lstrip()
    data, end = json.JSONDecoder().raw_decode(payload)
    layout, _ = json.JSONDecoder().raw_decode(payload[end:].lstrip().removeprefix(",").lstrip())
    return data, layout


def test_parallel_map_export_propagates_failures_and_caps_workers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    territories, results = _map_inputs((-2, 1.25))
    (tmp_path / "proposed_utc_offset_hours.png").mkdir()
    with pytest.raises(OSError, match=r"proposed_utc_offset_hours\.png"):
        export_maps(result_feature_collection(territories, results), tmp_path, MapOptions(workers=128))
    output = capsys.readouterr().err
    assert "Exporting map jobs [failed]:" in output
    assert "Exporting map jobs: 14/14 (100.0%" not in output
    assert "Exporting 14 map jobs with 14 workers" in caplog.text
    with pytest.raises(ValueError, match="Map export workers must be positive"):
        export_maps(result_feature_collection(territories, results), tmp_path, MapOptions(workers=0))


@pytest.mark.parametrize(
    ("shifts", "limit"),
    [
        ((-24, -2, 1.25, 1.25, None), 24),
        ((-2, -1, 0, 0.5, 1, 1.25, 1.5, 2, None), 2),
        ((1, 1, None), 1),
        ((-2, -1, None), 2),
        ((0, 0, None), 1),
        ((None, None), 1),
    ],
)
def test_plotly_shift_scale_is_continuous_and_centered_on_white(shifts: tuple[float | None, ...], limit: float) -> None:
    territories, results = _map_inputs(shifts)
    figure = build_plotly_figure(territories, results, "timezone_shift_hours")
    trace = figure.data[0]
    assert tuple(trace.z) == shifts
    assert trace.colorbar.tickvals is None
    assert trace.zmin == -limit
    assert trace.zmax == limit
    assert trace.colorscale == ((0, "#2166ac"), (0.5, "#ffffff"), (1, "#b35806"))
    assert [row[3] for row in trace.customdata] == list(shifts)
    assert "%{customdata[3]}" in trace.hovertemplate
    if any(shift is not None for shift in shifts):
        assert trace.showscale
    else:
        assert not trace.showscale
        assert figure.layout.annotations[0].text == "No data available"


def test_proposed_offset_scale_remains_discrete() -> None:
    territories, results = _map_inputs((0, 0, 0))
    for index, value in enumerate((-2, 0.5, 1.25)):
        results[index] = replace(results[index], proposed_utc_offset_hours=value)
    trace = build_plotly_figure(territories, results, "proposed_utc_offset_hours").data[0]
    assert tuple(trace.z) == (0, 1, 2)
    assert tuple(trace.colorbar.ticktext) == ("-2", "0.5", "1.25")
    assert tuple(trace.colorbar.tickvals) == (0, 1, 2)
    assert [stop[0] for stop in trace.colorscale] == [0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1]


@pytest.mark.parametrize(
    ("shifts", "expected_values"),
    [
        ((-24, -2, 1.25, None), [-24, -2, 1.25]),
        ((-2, 0, 0.5, 1, 1.25, 1.5, 2), [-2, 0, 0.5, 1, 1.25, 1.5, 2]),
        ((1, 1, None), [1]),
        ((0, 0, None), [0]),
        ((None, None), []),
    ],
)
def test_static_shift_scale_is_continuous_while_offsets_stay_discrete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shifts: tuple[float | None, ...], expected_values: list[float]
) -> None:

    legends: dict[str, list[float]] = {}

    def capture(figure: Figure, path: Path, **kwargs: object) -> None:
        if path.stem not in ("proposed_utc_offset_hours", "timezone_shift_hours"):
            return
        axis = figure.axes[0]
        assert {np.asarray(line.get_xdata())[0] for line in axis.lines if line.get_linestyle() == "--"} == {
            longitude + 7.5 for longitude in range(-180, 180, 15)
        }
        assert {np.asarray(line.get_xdata())[0] for line in axis.lines if line.get_linestyle() == ":"} == set(
            range(-180, 181, 15)
        )
        assert {text.get_text() for text in axis.texts} >= {f"UTC{offset:+d}" for offset in range(-12, 13)}
        assert axis.get_xlim()[1] - axis.get_xlim()[0] < 2
        outlines = axis.collections[-1]
        assert isinstance(outlines, LineCollection)
        np.testing.assert_allclose(outlines.get_linewidth(), 0.6)
        legend = axis.get_legend()
        if path.name.startswith("timezone_shift_hours") and expected_values:
            collection = axis.collections[0]
            limit = max(map(abs, expected_values)) or 1
            assert collection.norm.vmin == -limit
            assert collection.norm.vmax == limit
            np.testing.assert_array_equal(collection.cmap(collection.norm(np.array([0])))[0], (1, 1, 1, 1))
            assert len(figure.axes) == 2
            assert figure.axes[1].get_ylabel() == "Timezone shift (hours)"
            observed = collection.get_array()
            assert observed is not None
            legends[path.name] = sorted(set(observed))
            if limit == 2:
                colors = collection.cmap(collection.norm(np.array([1, 1.25, 1.5])))
                np.testing.assert_allclose(colors[1], (colors[0] + colors[2]) / 2)
        elif legend is None:
            legends[path.name] = []
            assert "No data available" in [text.get_text() for text in axis.texts]
        else:
            legends[path.name] = [float(text.get_text()) for text in legend.get_texts() if text.get_text() != "No data"]
            if path.name.startswith("proposed_utc_offset_hours"):
                assert legends[path.name] == [1]

    monkeypatch.setattr(Figure, "savefig", capture)
    territories, results = _map_inputs(shifts)
    export_maps(result_feature_collection(territories, results), tmp_path, kinds=("static",))
    assert legends["timezone_shift_hours.png"] == expected_values
    assert legends["timezone_shift_hours.svg"] == expected_values


class _UploadedTimezones:
    name = "toy_timezones.geojson"

    @staticmethod
    def getvalue() -> bytes:
        return TOY_BOUNDARIES.read_bytes()


class _FakeStreamlit(ModuleType):
    def __init__(self, *, has_upload: bool, run: bool, rules: str | None) -> None:
        super().__init__("streamlit")
        self.has_upload = has_upload
        self.run = run
        self.rules = rules
        self.errors: list[str] = []
        self.downloads: list[str] = []
        self.warnings: list[str] = []
        self.number_inputs: dict[str, dict[str, object]] = {}
        self.session_state: dict[str, object] = {}

    def set_page_config(self, **kwargs: object) -> None:
        return None

    def title(self, value: str) -> None:
        return None

    def caption(self, value: str) -> None:
        return None

    def file_uploader(self, label: str, **kwargs: object) -> _UploadedTimezones | None:
        return _UploadedTimezones() if "timezone" in label and self.has_upload else None

    def subheader(self, value: str) -> None:
        return None

    def selectbox(self, label: str, options: tuple[object, ...], *, index: int = 0, **kwargs: object) -> object:
        return options[index]

    def checkbox(self, label: str, *, value: bool, **kwargs: object) -> bool:
        return value

    def number_input(self, label: str, **kwargs: object) -> int:
        self.number_inputs[label] = kwargs
        value = kwargs.get("value", 0)
        return value if isinstance(value, int) else 0

    def text_area(self, label: str, **kwargs: object) -> str:
        return (
            self.rules
            if self.rules is not None and label.startswith("Subdivision rules")
            else str(kwargs.get("value", ""))
        )

    def slider(self, label: str, **kwargs: object) -> float:
        value = kwargs.get("value", 1.0)
        return float(value) if isinstance(value, (int, float)) else 1.0

    def text_input(self, label: str, **kwargs: object) -> str:
        return str(kwargs.get("value", ""))

    def columns(self, count: int) -> tuple[_FakeStreamlit, ...]:
        return (self,) * count

    def expander(self, label: str, *, expanded: bool) -> _FakeStreamlit:
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def info(self, message: str) -> None:
        return None

    def button(self, label: str, **kwargs: object) -> bool:
        return self.run

    def progress(self, value: float, **kwargs: object) -> _FakeStreamlit:
        return self

    def dataframe(self, data: object, **kwargs: object) -> None:
        return None

    def plotly_chart(self, figure: object, **kwargs: object) -> None:
        return None

    def download_button(self, label: str, **kwargs: str) -> None:
        self.downloads.append(label)

    def success(self, message: str) -> None:
        return None

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def test_streamlit_dashboard_upload_run_and_input_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toy_data_directory: Path
) -> None:
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"countries": ["TST"]}), encoding="utf-8")
    (toy_data_directory / "downloads" / "world_timezones.geojson").unlink()  # Must come from the upload.
    arguments = ["--config", str(config), "--data-directory", str(toy_data_directory)]
    fake = _FakeStreamlit(has_upload=False, run=False, rules="{}")
    monkeypatch.setattr(dashboard, "st", fake)
    dashboard.main(arguments)
    assert not fake.errors
    fake = _FakeStreamlit(has_upload=True, run=True, rules='{"USA": 1}')
    monkeypatch.setattr(dashboard, "st", fake)
    dashboard.main(arguments)
    assert not fake.errors
    assert fake.downloads == ["Download results CSV"]
    # A rerun without pressing the button (e.g. after picking another map field) keeps showing the last results.
    fake.run = False
    dashboard.main(arguments)
    assert fake.downloads == ["Download results CSV"] * 2
    fake = _FakeStreamlit(has_upload=True, run=True, rules="not json")
    monkeypatch.setattr(dashboard, "st", fake)
    dashboard.main(arguments)
    assert fake.errors


def test_dashboard_refuses_concurrent_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeStreamlit(has_upload=True, run=True, rules="{}")
    monkeypatch.setattr(dashboard, "st", fake)
    with dashboard._RUN_LOCK:
        dashboard.main(["--data-directory", str(tmp_path)])
    assert fake.warnings == ["Another experiment is running on this server; try again when it finishes."]
    assert not fake.downloads
    assert not dashboard._RUN_LOCK.locked()


def test_dashboard_has_a_control_for_every_configuration_setting() -> None:
    controls = {field.name for field in fields(dashboard.DashboardInputs)}
    settings = {field.name for field in fields(ExperimentConfig)}
    assert settings >= dashboard.NOT_IN_DASHBOARD
    assert settings - dashboard.NOT_IN_DASHBOARD <= controls


def test_dashboard_admin_level_is_the_fallback_for_default_subdivision_rules() -> None:
    config = ExperimentConfig(default_admin_level=2)
    fake = _FakeStreamlit(has_upload=False, run=False, rules=None)
    inputs = dashboard._dashboard_inputs(fake, config)
    result = dashboard._build_config(inputs, config)
    assert result == config
    assert result.admin_level_for("USA") == 1
    assert result.admin_level_for("FRA") == 0
    assert result.admin_level_for("ARG") == 2
    widget = fake.number_inputs["Default administrative level"]
    assert "default_admin_level" in str(widget["help"])


def test_dashboard_runs_without_uploads_and_preserves_base_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig(
        subdivision_rules={"TST": 1},
        merged_subdivision_levels={},
        utc_offsets=(1, 2),
        sample_grid_size=2,
        minimum_noon=None,
        solar_metric="median",
    )
    fake = _FakeStreamlit(has_upload=False, run=True, rules='{"TST": 1}')
    inputs = dashboard._dashboard_inputs(fake, config)
    assert dashboard._build_config(inputs, config) == config
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    (tmp_path / "downloads").mkdir()
    (tmp_path / "downloads" / "world_countries.geojson").write_bytes(TOY_BOUNDARIES.read_bytes())
    (tmp_path / "downloads" / "boundaries_gadm_TST_1.geojson").write_bytes(TOY_BOUNDARIES.read_bytes())
    monkeypatch.setattr(dashboard, "st", fake)
    dashboard.main(["--config", str(path), "--data-directory", str(tmp_path)])
    assert not fake.errors
    assert fake.downloads == ["Download results CSV"]
