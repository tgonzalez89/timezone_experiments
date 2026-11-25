"""Input preparation: downloads, caching, subdivisions, and fallbacks."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest
from factories import (
    TOY_BOUNDARIES,
)

from timezone_experiments.core.config import ExperimentConfig
from timezone_experiments.data import preparation
from timezone_experiments.data.boundaries import load_territory_geometries
from timezone_experiments.data.download import catalog
from timezone_experiments.data.download import sources as download_data
from timezone_experiments.data.download.catalog import metadata_path, read_source_metadata
from timezone_experiments.data.download.http import DownloadError
from timezone_experiments.data.paths import boundary_filename, population_filename, timezones_filename, world_filename
from timezone_experiments.data.preparation import InputOptions, load_experiment_territories, prepare_inputs
from timezone_experiments.data.prepared_cache import read_base_territories
from timezone_experiments.experiments.results_io import result_csv_rows
from timezone_experiments.experiments.runner import run_experiment
from timezone_experiments.optimization.assessment import calculate_noon_statistics

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from timezone_experiments.core.models import Territory
    from timezone_experiments.data.population import PopulationPoint
    from timezone_experiments.data.timezone_index import TimezoneBoundaryIndex


def _downloads(data_directory: Path) -> Path:
    """Create and return the downloads directory of a test data directory."""
    downloads = data_directory / "downloads"
    downloads.mkdir(exist_ok=True)
    return downloads


def test_country_inputs_download_once_and_refresh_explicitly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    timezone_calls = 0

    def fetch(url: str) -> bytes:
        calls.append(url)
        return TOY_BOUNDARIES.read_bytes()

    monkeypatch.setattr(download_data, "download_bytes", fetch)

    def fetch_timezones(directory: Path) -> Path:
        nonlocal timezone_calls
        timezone_calls += 1
        path = directory / timezones_filename()
        path.write_bytes(TOY_BOUNDARIES.read_bytes())
        return path

    monkeypatch.setattr(preparation, "download_timezones", fetch_timezones)
    options = InputOptions(data_directory=tmp_path)
    config = ExperimentConfig(subdivision_rules={}, merged_subdivision_levels={})
    first = prepare_inputs(config, options)
    assert first.population is None
    assert first.timezones == tmp_path / "downloads" / timezones_filename()
    assert prepare_inputs(config, options) == first
    assert len(calls) == 1
    assert timezone_calls == 1
    assert prepare_inputs(config, replace(options, refresh=True)) == first
    assert len(calls) == 2
    assert timezone_calls == 2


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
        path = directory / population_filename()
        path.write_text("latitude,longitude,population\n4,4,50\n", encoding="utf-8")
        path.with_suffix(".source.json").write_text(json.dumps({"cell_degrees": cell_degrees}), encoding="utf-8")
        return path

    def timezones(directory: Path) -> Path:
        timezone_calls.append(directory)
        path = directory / timezones_filename()
        path.write_bytes(TOY_BOUNDARIES.read_bytes())
        return path

    monkeypatch.setattr(preparation, "prepare_population", population)
    monkeypatch.setattr(preparation, "download_timezones", timezones)
    monkeypatch.setattr(download_data, "download_bytes", lambda url: TOY_BOUNDARIES.read_bytes())
    config = ExperimentConfig(coverage_type="population", countries=("TST",))
    options = InputOptions(data_directory=tmp_path)
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
    prepare_inputs(config, options)
    assert population_calls == [(1.0, False, config.workers)]
    assert len(timezone_calls) == 1
    prepare_inputs(replace(config, population_cell_degrees=0.5), options)
    assert population_calls[-1] == (0.5, False, config.workers)
    prepare_inputs(config, replace(options, refresh=True))
    assert population_calls[-1] == (1.0, True, config.workers)
    assert len(timezone_calls) == 2


def test_explicit_population_and_timezones_are_not_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toy_data_directory: Path
) -> None:
    population = tmp_path / "custom.csv"
    population.write_text("latitude,longitude,population\n4,4,100\n", encoding="utf-8")
    (toy_data_directory / "downloads" / timezones_filename()).unlink()

    def no_download(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Explicit population and timezone files must not be downloaded.")

    monkeypatch.setattr(preparation, "download_timezones", no_download)
    monkeypatch.setattr(preparation, "prepare_population", no_download)
    options = InputOptions(data_directory=toy_data_directory, population_path=population, timezones_path=TOY_BOUNDARIES)
    config = ExperimentConfig(coverage_type="population", countries=("TST",))
    territories = load_experiment_territories(config, options)
    assert len(territories) == 1
    assert sum(sample.population_weight or 0 for sample in territories[0].samples) == 100


def test_prepared_base_cache_reuses_across_optimizer_changes_and_invalidates_spatial_changes(
    toy_data_directory: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = 0
    original = preparation.load_territories

    caplog.set_level(logging.INFO)

    def counted(
        path: str | Path,
        config: ExperimentConfig,
        *,
        grid_size: int | None = None,
    ) -> list[Territory]:
        nonlocal calls
        calls += 1
        return original(path, config, grid_size=grid_size)

    monkeypatch.setattr(preparation, "load_territories", counted)
    options = InputOptions(data_directory=toy_data_directory)
    initial = ExperimentConfig(sample_grid_size=3, workers=1, countries=("TST",))
    first = load_experiment_territories(initial, options)
    first_cache = next((toy_data_directory / "cache" / "prepared" / "base").glob("*.sqlite3"))
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


def test_timezone_and_population_layers_cache_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toy_data_directory: Path
) -> None:
    population = tmp_path / "population.csv"
    population.write_text("latitude,longitude,population\n4,4,100\n", encoding="utf-8")
    timezone = tmp_path / "timezone.geojson"
    timezone.write_bytes(TOY_BOUNDARIES.read_bytes())
    options = InputOptions(
        data_directory=toy_data_directory,
        population_path=population,
        timezones_path=timezone,
    )
    config = ExperimentConfig(
        countries=("TST",),
        coverage_type="population",
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

    timezone.write_bytes(TOY_BOUNDARIES.read_bytes() + b"\n")
    third = load_experiment_territories(config, options)
    assert timezone_calls == 2
    assert population_calls == 2
    assert sum(sample.population_weight or 0 for sample in third[0].samples) == 200


@pytest.mark.parametrize("field", ["population_path", "timezones_path"])
def test_missing_explicit_file_is_not_replaced_by_download(tmp_path: Path, field: str) -> None:
    options = replace(InputOptions(data_directory=tmp_path), **{field: tmp_path / "missing"})
    with pytest.raises(FileNotFoundError, match="Explicit input file"):
        prepare_inputs(ExperimentConfig(), options)
    assert not list(tmp_path.iterdir())


@pytest.mark.slow
@pytest.mark.parametrize(
    "config",
    [
        ExperimentConfig(subdivision_rules={"TST": 1}, merged_subdivision_levels={}),
        ExperimentConfig(default_admin_level=1, subdivision_rules={"OTH": 0}, merged_subdivision_levels={}),
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
    world = json.loads(TOY_BOUNDARIES.read_text(encoding="utf-8"))
    world["features"][0]["properties"] = {"ISO_A3": "TST", "NAME_0": "Testland"}
    world["features"][1]["properties"] = {"ISO_A3": "OTH", "NAME_0": "Other"}
    monkeypatch.setattr(download_data, "download_bytes", lambda url: json.dumps(world).encode())
    calls: list[tuple[str, int]] = []

    def subdivisions(source: str, country: str, level: int, directory: Path) -> Path:
        calls.append((country, level))
        data = json.loads(TOY_BOUNDARIES.read_text(encoding="utf-8"))
        for index, feature in enumerate(data["features"]):
            feature["properties"] = {
                "GID_0": country,
                f"GID_{level}": f"{country}.{index}",
                f"NAME_{level}": str(index),
            }
        path = directory / boundary_filename(source, country, level)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    monkeypatch.setattr(preparation, "download", subdivisions)
    options = InputOptions(data_directory=tmp_path)
    territories = load_experiment_territories(config, options)
    assert [(unit.country_code, unit.admin_level) for unit in territories] == [("OTH", 0), ("TST", 1), ("TST", 1)]
    first = prepare_inputs(config, options)
    assert calls == [("TST", 1)]
    second = prepare_inputs(replace(config, subdivision_rules={"TST": 2, "OTH": 0}), options)
    assert second.boundaries != first.boundaries
    assert calls == [("TST", 1), ("TST", 2)]


@pytest.mark.parametrize("invalid_world", [False, True])
def test_unavailable_subdivisions_fail_without_falling_back_to_countries(
    tmp_path: Path, *, invalid_world: bool
) -> None:
    world = json.loads(TOY_BOUNDARIES.read_text(encoding="utf-8"))
    if invalid_world:
        world["features"][0]["properties"] = {}
    (_downloads(tmp_path) / world_filename(include_antarctica=False)).write_text(json.dumps(world), encoding="utf-8")
    (_downloads(tmp_path) / boundary_filename("gadm", "TST", 1)).write_text(
        json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"no subdivision features|no recognized country code"):
        prepare_inputs(
            ExperimentConfig(subdivision_rules={"TST": 1}, merged_subdivision_levels={}),
            InputOptions(data_directory=tmp_path),
        )
    assert not list((tmp_path / "downloads").glob("world_mixed_*.geojson"))


def _fake_subdivision_sources(
    monkeypatch: pytest.MonkeyPatch, gadm_status: HTTPStatus | None, alternative_status: HTTPStatus | None
) -> list[str]:
    """Serve a two-country world and fake subdivision downloads; returns the sources requested, in order."""
    world = json.loads(TOY_BOUNDARIES.read_text(encoding="utf-8"))
    world["features"][0]["properties"] = {"ISO_A3": "TST", "NAME_0": "Testland"}
    world["features"][1]["properties"] = {"ISO_A3": "OTH", "NAME_0": "Other"}
    monkeypatch.setattr(download_data, "download_bytes", lambda url: json.dumps(world).encode())
    calls: list[str] = []

    def subdivisions(source: str, country: str, level: int, directory: Path) -> Path:
        calls.append(source)
        status = gadm_status if source == "gadm" else alternative_status
        if status is not None:
            url = "https://example.invalid"
            raise DownloadError(url, status.phrase, status)
        data = json.loads(TOY_BOUNDARIES.read_text(encoding="utf-8"))
        for index, feature in enumerate(data["features"]):
            feature["properties"] = {
                "shapeGroup": country,
                "shapeType": f"ADM{level}",
                "shapeID": f"unit-{index}",
                "shapeName": f"Subdivision {index}",
            }
        path = directory / boundary_filename(source, country, level)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    monkeypatch.setattr(preparation, "download", subdivisions)
    return calls


_SUBDIVIDED_TST = ExperimentConfig(subdivision_rules={"TST": 1}, merged_subdivision_levels={})


@pytest.mark.parametrize(
    ("gadm_status", "alternative_status", "expected_calls"),
    [
        (HTTPStatus.SERVICE_UNAVAILABLE, None, ["gadm"]),
        (HTTPStatus.NOT_FOUND, HTTPStatus.NOT_FOUND, ["gadm", "geoboundaries"]),
    ],
    ids=["gadm-unavailable", "both-missing"],
)
def test_subdivision_download_errors_other_than_a_missing_gadm_file_are_raised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gadm_status: HTTPStatus,
    alternative_status: HTTPStatus | None,
    expected_calls: list[str],
) -> None:
    calls = _fake_subdivision_sources(monkeypatch, gadm_status, alternative_status)
    with pytest.raises(DownloadError) as error:
        prepare_inputs(_SUBDIVIDED_TST, InputOptions(data_directory=tmp_path))
    assert error.value.status == (alternative_status or gadm_status)
    assert calls == expected_calls
    assert not list((tmp_path / "downloads").glob("world_mixed_*.geojson"))


@pytest.mark.slow
def test_missing_gadm_uses_same_level_geoboundaries_and_caches_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = _fake_subdivision_sources(monkeypatch, HTTPStatus.NOT_FOUND, None)
    options = InputOptions(data_directory=tmp_path)
    first = prepare_inputs(_SUBDIVIDED_TST, options)
    assert "trying GeoBoundaries at the same administrative level" in caplog.text
    territories = load_experiment_territories(_SUBDIVIDED_TST, options)
    assert [(unit.country_code, unit.admin_level) for unit in territories] == [("OTH", 0), ("TST", 1), ("TST", 1)]
    first.boundaries.unlink()
    assert prepare_inputs(_SUBDIVIDED_TST, options) == first
    assert calls == ["gadm", "geoboundaries"]
    prepare_inputs(_SUBDIVIDED_TST, replace(options, refresh=True))
    assert calls == ["gadm", "geoboundaries", "gadm", "geoboundaries"]


def test_countries_absent_from_the_world_dataset_are_explicit_errors(toy_data_directory: Path) -> None:
    options = InputOptions(data_directory=toy_data_directory)
    with pytest.raises(ValueError, match="absent from the world dataset: XYZ"):
        prepare_inputs(ExperimentConfig(subdivision_rules={"XYZ": 1}, merged_subdivision_levels={}), options)
    with pytest.raises(ValueError, match="countries lists codes absent from the world dataset: XYZ"):
        prepare_inputs(ExperimentConfig(countries=("TST", "XYZ")), options)


def test_countries_limit_boundaries_and_subdivision_downloads(
    toy_data_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = json.loads(TOY_BOUNDARIES.read_text(encoding="utf-8"))
    world["features"][0]["properties"] = {"ISO_A3": "TST", "NAME_0": "Testland"}
    world["features"][1]["properties"] = {"ISO_A3": "OTH", "NAME_0": "Other"}
    (toy_data_directory / "downloads" / world_filename(include_antarctica=False)).write_text(
        json.dumps(world), encoding="utf-8"
    )

    def no_download(*_args: object) -> None:
        pytest.fail("Countries outside the selection must not be downloaded.")

    monkeypatch.setattr(preparation, "download", no_download)
    options = InputOptions(data_directory=toy_data_directory)
    # OTH's subdivision rule is ignored because only TST is selected; TST comes from the cached GADM file.
    config = ExperimentConfig(countries=("TST",), subdivision_rules={"TST": 1, "OTH": 1}, merged_subdivision_levels={})
    boundaries = prepare_inputs(config, options).boundaries
    assert boundaries.name.startswith("world_mixed_")
    territories = load_experiment_territories(config, options)
    assert [(unit.country_code, unit.admin_level) for unit in territories] == [("TST", 1), ("TST", 1)]
    other = replace(config, countries=("OTH",), subdivision_rules={"TST": 1})
    assert prepare_inputs(other, options).boundaries != boundaries
    assert [(unit.country_code, unit.admin_level) for unit in load_experiment_territories(other, options)] == [
        ("OTH", 0)
    ]


@pytest.mark.parametrize("metadata", ["[]", "{corrupt", '{"cell_degrees": 0.5}'])
def test_unusable_population_metadata_regenerates_the_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: str
) -> None:
    (_downloads(tmp_path) / population_filename()).write_text("cached", encoding="utf-8")
    metadata_path(_downloads(tmp_path) / population_filename()).write_text(metadata, encoding="utf-8")
    regenerated: list[float] = []

    def prepare(directory: Path, cell_degrees: float, **_options: object) -> Path:
        regenerated.append(cell_degrees)
        return directory / population_filename()

    monkeypatch.setattr(preparation, "prepare_population", prepare)
    preparation._prepare_population(ExperimentConfig(coverage_type="population"), InputOptions(data_directory=tmp_path))
    assert regenerated == [1.0]


@pytest.mark.slow
@pytest.mark.parametrize("provider", ["gadm", "geoboundaries"])
def test_merged_subdivision_levels_combine_subdivisions_and_preserve_country_identity(
    tmp_path: Path, provider: str, caplog: pytest.LogCaptureFixture
) -> None:
    source = json.loads(TOY_BOUNDARIES.read_text(encoding="utf-8"))
    for index, feature in enumerate(source["features"]):
        feature["properties"] = (
            {"GID_0": "TST", "GID_1": f"TST.{index}", "NAME_1": f"Province {index}"}
            if provider == "gadm"
            else {"shapeGroup": "TST", "shapeType": "ADM1", "shapeID": str(index), "shapeName": f"Province {index}"}
        )
    (_downloads(tmp_path) / boundary_filename(provider, "TST", 1)).write_text(json.dumps(source), encoding="utf-8")
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
    (_downloads(tmp_path) / world_filename(include_antarctica=False)).write_text(json.dumps(world), encoding="utf-8")
    config = ExperimentConfig(subdivision_rules={}, merged_subdivision_levels={"TST": 1}, workers=1, sample_grid_size=2)
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
    assert results[0].territory_name == "Testland"
    assert results[0].noon == calculate_noon_statistics(
        territory, results[0].proposed_utc_offset_hours, year=results[0].noon.year
    )
    assert "areas absent" in caplog.text
    unmodified = prepare_inputs(replace(config, merged_subdivision_levels={}), options)
    assert unmodified.boundaries != prepared.boundaries
    assert json.loads(unmodified.boundaries.read_text(encoding="utf-8")) == world


def test_merged_subdivision_levels_fail_for_missing_countries_and_wrong_source_codes(tmp_path: Path) -> None:
    (_downloads(tmp_path) / world_filename(include_antarctica=False)).write_bytes(TOY_BOUNDARIES.read_bytes())
    options = InputOptions(data_directory=tmp_path)
    config = ExperimentConfig(subdivision_rules={}, merged_subdivision_levels={"XYZ": 1})
    with pytest.raises(ValueError, match="absent from the world dataset: XYZ"):
        prepare_inputs(config, options)
    source = json.loads(TOY_BOUNDARIES.read_text(encoding="utf-8"))
    for feature in source["features"]:
        feature["properties"]["GID_0"] = "OTH"
    (_downloads(tmp_path) / boundary_filename("gadm", "TST", 1)).write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="no features belonging to TST"):
        prepare_inputs(replace(config, merged_subdivision_levels={"TST": 1}), options)


def test_damaged_prepared_caches_are_rebuilt(toy_data_directory: Path) -> None:
    config = ExperimentConfig(subdivision_rules={}, merged_subdivision_levels={}, sample_grid_size=2, workers=1)
    options = InputOptions(data_directory=toy_data_directory)
    expected = load_experiment_territories(config, options)
    caches = sorted((toy_data_directory / "cache" / "prepared").rglob("*.sqlite3"))
    assert {path.parent.name for path in caches} == {"base", "timezones"}
    for path in caches:
        path.write_bytes(b"damaged")
    assert load_experiment_territories(config, options) == expected
    base = next(path for path in caches if path.parent.name == "base")
    assert read_base_territories(base) is not None


def test_assembled_boundaries_record_their_sources_and_change_with_the_pins(
    toy_data_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig(subdivision_rules={"TST": 1}, merged_subdivision_levels={})
    options = InputOptions(data_directory=toy_data_directory)
    assembled = preparation.prepare_boundaries(config, options)
    metadata = read_source_metadata(assembled)
    assert metadata is not None
    assert [component["file"] for component in metadata["derived_from"]] == [
        world_filename(include_antarctica=False),
        boundary_filename("gadm", "TST", 1),
    ]
    # A new Natural Earth pin assembles a new file instead of reusing one built from the old release.
    old_world = toy_data_directory / "downloads" / world_filename(include_antarctica=False)
    monkeypatch.setitem(catalog.SOURCES, "natural_earth", replace(catalog.SOURCES["natural_earth"], version="v99"))
    old_world.replace(toy_data_directory / "downloads" / world_filename(include_antarctica=False))
    assert preparation.prepare_boundaries(config, options) != assembled


def test_areas_left_out_of_merged_countries_are_kept_for_drawing_only() -> None:
    mainland = [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]
    overseas = [[[10, 10], [11, 10], [11, 11], [10, 11], [10, 10]]]
    properties = {"ISO_A3_EH": "FRA", "ADM0_A3": "FRA"}
    original = [
        {
            "type": "Feature",
            "properties": properties,
            "geometry": {"type": "MultiPolygon", "coordinates": [[mainland[0]], [overseas[0]]]},
        }
    ]
    rebuilt = [{"type": "Feature", "properties": properties, "geometry": {"type": "Polygon", "coordinates": mainland}}]
    config = ExperimentConfig(merged_subdivision_levels={"FRA": 1})
    (area,) = preparation._excluded_areas(original, rebuilt, config)
    assert area["type"] == "Polygon"
    assert area["coordinates"][0][0] == (10.0, 10.0)
    # A country that is not merged contributes nothing.
    assert preparation._excluded_areas(original, rebuilt, ExperimentConfig(merged_subdivision_levels={})) == []
