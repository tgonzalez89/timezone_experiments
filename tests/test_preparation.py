"""Tests for automatic configuration-driven input resolution and caching."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from email.message import Message
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import HTTPError

import pytest

from timezone_experiments.config import ExperimentConfig, load_config
from timezone_experiments.data import download_data, preparation
from timezone_experiments.data.boundaries import load_territory_geometries
from timezone_experiments.data.noon import calculate_noon_statistics
from timezone_experiments.data.preparation import InputOptions, load_experiment_territories, prepare_inputs
from timezone_experiments.experiments.runner import result_csv_rows, run_experiment

TOY = Path(__file__).parents[1] / "examples" / "toy_boundaries.geojson"

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from timezone_experiments.data.models import Territory
    from timezone_experiments.data.population import PopulationPoint
    from timezone_experiments.data.timezone_index import TimezoneBoundaryIndex


def test_country_inputs_download_once_and_refresh_explicitly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return TOY.read_bytes()

    monkeypatch.setattr(download_data, "_download_bytes", fetch)
    options = InputOptions(data_directory=tmp_path)
    config = ExperimentConfig(subdivision_rules={}, country_boundary_levels={})
    first = prepare_inputs(config, options)
    assert first.population is None
    assert first.timezones is None
    assert prepare_inputs(config, options) == first
    assert len(calls) == 1
    assert prepare_inputs(config, replace(options, refresh=True)) == first
    assert len(calls) == 2


def test_required_overlays_are_downloaded_cached_and_refreshed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level(logging.INFO)
    population_calls: list[tuple[float, bool, int]] = []
    timezone_calls: list[Path] = []

    def population(directory: Path, cell_degrees: float, *, refresh: bool, workers: int) -> Path:
        population_calls.append((cell_degrees, refresh, workers))
        path = directory / "population_2025.csv"
        path.write_text("latitude,longitude,population\n4,4,50\n", encoding="utf-8")
        path.with_suffix(".source.json").write_text(json.dumps({"cell_degrees": cell_degrees}), encoding="utf-8")
        return path

    def timezones(directory: Path) -> Path:
        timezone_calls.append(directory)
        path = directory / "world_timezones.geojson"
        path.write_bytes(TOY.read_bytes())
        return path

    monkeypatch.setattr(preparation, "prepare_population", population)
    monkeypatch.setattr(preparation, "download_timezones", timezones)
    config = ExperimentConfig(coverage_type="population", compare_current_timezones=True)
    options = InputOptions(data_directory=tmp_path, boundaries_path=TOY)
    territories = load_experiment_territories(config, options)
    assert territories[0].current_timezone == "Etc/UTC"
    assert sum(sample.population_weight or 0 for sample in territories[0].samples) == 50
    for message in (
        "Input 1/3: administrative boundaries",
        "Input 2/3: population grid",
        "Input 3/3: current timezone boundaries",
        "Input processing complete: 1 territories ready",
    ):
        assert message in caplog.text
    output = capsys.readouterr().err.splitlines()
    for label in ("Sampling territory", "Timezone overlay", "Population overlay"):
        lines = [line for line in output if line.startswith(f"{label}:")]
        assert len(lines) == 2
        assert "0/1" in lines[0]
        assert "1/1 (100.0%; 0 left)" in lines[1]
        assert "finished TST:TST (Testland)" in lines[1]
        assert f"{label} 1/1:" not in caplog.text
    prepare_inputs(config, options)
    assert population_calls == [(1.0, False, config.workers)]
    assert len(timezone_calls) == 1
    prepare_inputs(replace(config, population_cell_degrees=0.5), options)
    assert population_calls[-1] == (0.5, False, config.workers)
    prepare_inputs(config, replace(options, refresh=True))
    assert population_calls[-1] == (1.0, True, config.workers)
    assert len(timezone_calls) == 2


def test_explicit_overrides_require_no_downloads_even_on_refresh(tmp_path: Path) -> None:
    population = tmp_path / "custom.csv"
    population.write_text("latitude,longitude,population\n4,4,100\n", encoding="utf-8")
    options = InputOptions(
        data_directory=tmp_path / "unused",
        boundaries_path=TOY,
        population_path=population,
        timezones_path=TOY,
        refresh=True,
    )
    territories = load_experiment_territories(ExperimentConfig(coverage_type="population"), options)
    assert len(territories) == 1
    assert not options.data_directory.exists()
    assert sum(sample.population_weight or 0 for sample in territories[0].samples) == 100


def test_prepared_base_cache_reuses_across_optimizer_changes_and_invalidates_spatial_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = 0
    original = preparation.load_territories

    caplog.set_level(logging.INFO)

    def counted(
        path: str | Path,
        config: ExperimentConfig,
        *,
        source_admin_level: int | None = None,
        grid_size: int | None = None,
    ) -> list[Territory]:
        nonlocal calls
        calls += 1
        return original(path, config, source_admin_level=source_admin_level, grid_size=grid_size)

    monkeypatch.setattr(preparation, "load_territories", counted)
    options = InputOptions(data_directory=tmp_path, boundaries_path=TOY)
    initial = ExperimentConfig(sample_grid_size=3, workers=1)
    first = load_experiment_territories(initial, options)
    first_cache = next((tmp_path / "cache" / "prepared" / "base").glob("*.sqlite3"))
    similar = replace(
        initial,
        solar_metric="min",
        minimum_noon="10:30",
        maximum_noon="14:00",
        coverage_threshold=0.5,
        utc_offsets=(1, 2),
        year=2026,
    )
    assert load_experiment_territories(similar, options) == first
    assert calls == 1
    assert "Using cached prepared territory samples" in caplog.text

    load_experiment_territories(replace(initial, sample_grid_size=4), options)
    assert calls == 2
    first_cache.write_bytes(b"corrupt")
    load_experiment_territories(replace(initial, sample_grid_size=3), options)
    assert calls == 3
    assert "Ignoring unusable prepared territory cache" in caplog.text


def test_timezone_and_population_layers_cache_independently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    population = tmp_path / "population.csv"
    population.write_text("latitude,longitude,population\n4,4,100\n", encoding="utf-8")
    timezone = tmp_path / "timezone.geojson"
    timezone.write_bytes(TOY.read_bytes())
    options = InputOptions(
        data_directory=tmp_path / "data",
        boundaries_path=TOY,
        population_path=population,
        timezones_path=timezone,
    )
    config = ExperimentConfig(
        coverage_type="population",
        compare_current_timezones=True,
        sample_grid_size=3,
        workers=1,
    )
    timezone_calls = population_calls = 0
    original_timezone = preparation.annotate_current_timezones
    original_population = preparation.attach_population_weights

    def counted_timezone(
        territories: Sequence[Territory], index: TimezoneBoundaryIndex, *, workers: int = 1
    ) -> list[Territory]:
        nonlocal timezone_calls
        timezone_calls += 1
        return original_timezone(territories, index, workers=workers)

    def counted_population(
        territories: Iterable[Territory], population_points: Iterable[PopulationPoint], *, workers: int = 1
    ) -> list[Territory]:
        nonlocal population_calls
        population_calls += 1
        return original_population(territories, population_points, workers=workers)

    monkeypatch.setattr(preparation, "annotate_current_timezones", counted_timezone)
    monkeypatch.setattr(preparation, "attach_population_weights", counted_population)
    first = load_experiment_territories(config, options)
    assert load_experiment_territories(replace(config, coverage_threshold=0.5), options) == first
    assert (timezone_calls, population_calls) == (1, 1)

    population.write_text("latitude,longitude,population\n4,4,200\n", encoding="utf-8")
    second = load_experiment_territories(config, options)
    assert timezone_calls == 1
    assert population_calls == 2
    assert sum(sample.population_weight or 0 for sample in second[0].samples) == 200

    timezone.write_bytes(TOY.read_bytes() + b"\n")
    third = load_experiment_territories(config, options)
    assert timezone_calls == 2
    assert population_calls == 2
    assert sum(sample.population_weight or 0 for sample in third[0].samples) == 200


@pytest.mark.parametrize("field", ["boundaries_path", "population_path", "timezones_path"])
def test_missing_explicit_file_is_not_replaced_by_download(tmp_path: Path, field: str) -> None:
    options = replace(InputOptions(data_directory=tmp_path), **{field: tmp_path / "missing"})
    with pytest.raises(FileNotFoundError, match="Explicit input file"):
        prepare_inputs(ExperimentConfig(), options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "config",
    [
        ExperimentConfig(subdivision_rules={"TST": 1}, country_boundary_levels={}),
        ExperimentConfig(default_admin_level=1, subdivision_rules={"OTH": 0}, country_boundary_levels={}),
    ],
)
def test_mixed_world_downloads_only_configured_subdivisions_and_keys_cache_by_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: ExperimentConfig,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level(logging.INFO)
    world = json.loads(TOY.read_text(encoding="utf-8"))
    world["features"][0]["properties"] = {"ISO_A3": "TST", "NAME_0": "Testland"}
    world["features"][1]["properties"] = {"ISO_A3": "OTH", "NAME_0": "Other"}
    monkeypatch.setattr(download_data, "_download_bytes", lambda url: json.dumps(world).encode())
    calls: list[tuple[str, int]] = []

    def subdivisions(source: str, country: str, level: int, directory: Path) -> Path:
        calls.append((country, level))
        data = json.loads(TOY.read_text(encoding="utf-8"))
        for index, feature in enumerate(data["features"]):
            feature["properties"] = {
                "GID_0": country,
                f"GID_{level}": f"{country}.{index}",
                f"NAME_{level}": str(index),
            }
        path = directory / f"boundaries_{source}_{country}_{level}.geojson"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    monkeypatch.setattr(preparation, "download", subdivisions)
    options = InputOptions(data_directory=tmp_path)
    territories = load_experiment_territories(config, options)
    assert [(unit.country_code, unit.admin_level) for unit in territories] == [("OTH", 0), ("TST", 1), ("TST", 1)]
    assert "Subdivision dataset 1/1: TST ADM1" in caplog.text
    output = capsys.readouterr().err
    assert "Sampling territory: 3/3 (100.0%; 0 left)" in output
    assert "Sampling territory 3/3:" not in caplog.text
    first = prepare_inputs(config, options)
    assert calls == [("TST", 1)]
    second = prepare_inputs(replace(config, subdivision_rules={"TST": 2, "OTH": 0}), options)
    assert second.boundaries != first.boundaries
    assert calls == [("TST", 1), ("TST", 2)]


@pytest.mark.parametrize("invalid_world", [False, True])
def test_unavailable_subdivisions_fail_without_falling_back_to_countries(
    tmp_path: Path, *, invalid_world: bool
) -> None:
    world = json.loads(TOY.read_text(encoding="utf-8"))
    if invalid_world:
        world["features"][0]["properties"] = {}
    (tmp_path / "world_countries.geojson").write_text(json.dumps(world), encoding="utf-8")
    (tmp_path / "boundaries_gadm_TST_1.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"no subdivision features|no recognized country code"):
        prepare_inputs(
            ExperimentConfig(subdivision_rules={"TST": 1}, country_boundary_levels={}),
            InputOptions(data_directory=tmp_path),
        )
    assert not list(tmp_path.glob("world_mixed_*.geojson"))


@pytest.mark.parametrize(
    ("gadm_status", "alternative_status"),
    [
        (HTTPStatus.NOT_FOUND, None),
        (HTTPStatus.SERVICE_UNAVAILABLE, None),
        (HTTPStatus.NOT_FOUND, HTTPStatus.NOT_FOUND),
    ],
)
def test_missing_gadm_uses_same_level_geoboundaries_and_caches_without_hiding_other_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    gadm_status: HTTPStatus,
    alternative_status: HTTPStatus | None,
) -> None:
    world = json.loads(TOY.read_text(encoding="utf-8"))
    world["features"][0]["properties"] = {"ISO_A3": "TST", "NAME_0": "Testland"}
    world["features"][1]["properties"] = {"ISO_A3": "OTH", "NAME_0": "Other"}
    monkeypatch.setattr(download_data, "_download_bytes", lambda url: json.dumps(world).encode())
    calls: list[str] = []

    def subdivisions(source: str, country: str, level: int, directory: Path) -> Path:
        calls.append(source)
        status = gadm_status if source == "gadm" else alternative_status
        if status is not None:
            url = "https://example.invalid"
            raise HTTPError(url, status, status.phrase, Message(), None)
        data = json.loads(TOY.read_text(encoding="utf-8"))
        for index, feature in enumerate(data["features"]):
            feature["properties"] = {
                "shapeGroup": country,
                "shapeType": f"ADM{level}",
                "shapeID": f"unit-{index}",
                "shapeName": f"Subdivision {index}",
            }
        path = directory / f"boundaries_{source}_{country}_{level}.geojson"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    monkeypatch.setattr(preparation, "download", subdivisions)
    config = ExperimentConfig(subdivision_rules={"TST": 1}, country_boundary_levels={})
    options = InputOptions(data_directory=tmp_path)
    if gadm_status != HTTPStatus.NOT_FOUND or alternative_status is not None:
        with pytest.raises(HTTPError) as error:
            prepare_inputs(config, options)
        assert error.value.code == (alternative_status or gadm_status)
        assert calls == (["gadm", "geoboundaries"] if gadm_status == HTTPStatus.NOT_FOUND else ["gadm"])
        assert not list(tmp_path.glob("world_mixed_*.geojson"))
        return
    first = prepare_inputs(config, options)
    assert "trying GeoBoundaries at the same administrative level" in caplog.text
    territories = load_experiment_territories(config, options)
    assert [(unit.country_code, unit.admin_level) for unit in territories] == [("OTH", 0), ("TST", 1), ("TST", 1)]
    first.boundaries.unlink()
    assert prepare_inputs(config, options) == first
    assert calls == ["gadm", "geoboundaries"]
    prepare_inputs(config, replace(options, refresh=True))
    assert calls == ["gadm", "geoboundaries", "gadm", "geoboundaries"]


def test_missing_country_rule_and_invalid_metadata_are_explicit_errors(tmp_path: Path) -> None:
    (tmp_path / "world_countries.geojson").write_bytes(TOY.read_bytes())
    options = InputOptions(data_directory=tmp_path)
    with pytest.raises(ValueError, match="absent from the world dataset: XYZ"):
        prepare_inputs(ExperimentConfig(subdivision_rules={"XYZ": 1}, country_boundary_levels={}), options)
    (tmp_path / "population_2025.csv").write_text("cached", encoding="utf-8")
    (tmp_path / "population_2025.source.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid population cache metadata"):
        prepare_inputs(
            ExperimentConfig(coverage_type="population", subdivision_rules={}, country_boundary_levels={}), options
        )
    with pytest.raises(ValueError, match="non-negative"):
        prepare_inputs(ExperimentConfig(), replace(options, source_admin_level=-1))


@pytest.mark.parametrize(
    "settings",
    [
        {"compare_current_timezones": "false"},
        {"include_antarctica": 1},
        {"population_cell_degrees": True},
        {"population_cell_degrees": 0.2},
    ],
)
def test_invalid_download_configuration_is_rejected(settings: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ExperimentConfig.from_mapping(settings)


def test_configuration_loader_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        load_config(path)


@pytest.mark.parametrize("provider", ["gadm", "geoboundaries"])
def test_country_boundary_levels_combine_subdivisions_and_preserve_country_identity(
    tmp_path: Path, provider: str, caplog: pytest.LogCaptureFixture
) -> None:
    source = json.loads(TOY.read_text(encoding="utf-8"))
    for index, feature in enumerate(source["features"]):
        feature["properties"] = (
            {"GID_0": "TST", "GID_1": f"TST.{index}", "NAME_1": f"Province {index}"}
            if provider == "gadm"
            else {"shapeGroup": "TST", "shapeType": "ADM1", "shapeID": str(index), "shapeName": f"Province {index}"}
        )
    (tmp_path / f"boundaries_{provider}_TST_1.geojson").write_text(json.dumps(source), encoding="utf-8")
    world = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"ISO_A3": "TST", "NAME_0": "Testland"},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        *(feature["geometry"]["coordinates"] for feature in source["features"]),
                        [[[120, 0], [121, 0], [121, 1], [120, 1], [120, 0]]],
                    ],
                },
            }
        ],
    }
    (tmp_path / "world_countries.geojson").write_text(json.dumps(world), encoding="utf-8")
    config = ExperimentConfig(subdivision_rules={}, country_boundary_levels={"TST": 1}, workers=1, sample_grid_size=2)
    options = InputOptions(data_directory=tmp_path)
    territories = load_experiment_territories(config, options)
    assert len(territories) == 1
    territory = territories[0]
    assert (territory.territory_id, territory.name, territory.admin_level) == ("TST:TST", "Testland", 0)
    assert max(sample.longitude for sample in territory.samples) <= 16
    expected_geometry = {
        "type": "MultiPolygon",
        "coordinates": [feature["geometry"]["coordinates"] for feature in source["features"]],
    }
    assert territory.geometry == expected_geometry
    prepared = prepare_inputs(config, options)
    assert load_territory_geometries(prepared.boundaries, config) == {"TST:TST": expected_geometry}
    noon = calculate_noon_statistics(territory, 0, year=2025)
    assert noon.proposed_earliest_minutes > 600
    results = run_experiment(config, territories)
    assert len(results) == len(result_csv_rows(results)) == 1
    assert results[0].region == "Testland"
    assert results[0].noon == calculate_noon_statistics(
        territory, results[0].proposed_utc_offset, year=results[0].noon.year
    )
    assert "areas absent" in caplog.text
    unmodified = prepare_inputs(replace(config, country_boundary_levels={}), options)
    assert unmodified.boundaries != prepared.boundaries
    assert json.loads(unmodified.boundaries.read_text(encoding="utf-8")) == world


def test_country_boundary_levels_fail_for_missing_countries_and_wrong_source_codes(tmp_path: Path) -> None:
    (tmp_path / "world_countries.geojson").write_bytes(TOY.read_bytes())
    options = InputOptions(data_directory=tmp_path)
    config = ExperimentConfig(subdivision_rules={}, country_boundary_levels={"XYZ": 1})
    with pytest.raises(ValueError, match="absent from the world dataset: XYZ"):
        prepare_inputs(config, options)
    source = json.loads(TOY.read_text(encoding="utf-8"))
    for feature in source["features"]:
        feature["properties"]["GID_0"] = "OTH"
    (tmp_path / "boundaries_gadm_TST_1.geojson").write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="no features belonging to TST"):
        prepare_inputs(replace(config, country_boundary_levels={"TST": 1}), options)
