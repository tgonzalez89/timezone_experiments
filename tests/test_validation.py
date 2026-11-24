"""Verify untrusted inputs (CSV rows, samples, config, country codes) are rejected with context."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING
from zipfile import ZipFile

import pytest

from timezone_experiments import cli, fingerprint
from timezone_experiments.config import ExperimentConfig
from timezone_experiments.data import download_boundaries, download_population, prepared_cache
from timezone_experiments.data.models import TerritorySample
from timezone_experiments.data.population import load_population_csv
from timezone_experiments.data.timezones import is_known_timezone, parse_utc_offset

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("1,2,nan", "population must be finite"),
        ("1,2,inf", "population must be finite"),
        ("1,2,-1", "population must be finite"),
        ("91,2,3", "latitude must be between"),
        ("1,-181,3", "longitude must be between"),
        ("1,2,abc", "could not convert"),
        ("1,2", "missing columns"),
    ],
)
def test_population_csv_rows_are_validated_with_file_and_line(tmp_path: Path, row: str, message: str) -> None:
    path = tmp_path / "population.csv"
    path.write_text(f"latitude,longitude,population\n0,0,1\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"population\.csv:3: .*{message}"):
        load_population_csv(path)


@pytest.mark.parametrize(
    "weights",
    [
        {"area_weight": float("nan")},
        {"longitude_weight": float("inf")},
        {"area_weight": -1.0},
        {"population_weight": float("nan")},
        {"population_weight": -1.0},
    ],
)
def test_samples_reject_non_finite_or_negative_weights(weights: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        TerritorySample(0, 0, **weights)


@pytest.mark.parametrize("year", [1899, 2201])
def test_config_rejects_years_outside_the_supported_range(year: int) -> None:
    with pytest.raises(ValueError, match="year must be between 1900 and 2200"):
        ExperimentConfig(year=year)


def test_config_normalizes_and_validates_country_codes() -> None:
    assert ExperimentConfig.from_mapping({"subdivision_rules": {"usa": 1}}).subdivision_rules == {"USA": 1}
    with pytest.raises(ValueError, match="Country codes must be three"):
        ExperimentConfig.from_mapping({"subdivision_rules": {"../X": 1}})


@pytest.mark.parametrize("country", ["../x", "US", "USA/1", ""])
def test_boundary_download_rejects_unsafe_country_codes_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, country: str
) -> None:
    monkeypatch.setattr(download_boundaries, "download_bytes", pytest.fail)
    with pytest.raises(ValueError, match="Country codes must be three"):
        download_boundaries.download("gadm", country, 1, tmp_path)
    assert not any(tmp_path.iterdir())


def test_cli_rejects_unsafe_country_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "settings.json"
    config.write_text('{"countries": ["../x"]}', encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["download", "--config", str(config), "--data-directory", str(tmp_path)])
    assert exit_info.value.code == 2
    assert "Country codes must be three" in capsys.readouterr().err


def test_known_timezone_detection() -> None:
    assert is_known_timezone("Europe/Madrid")
    assert is_known_timezone("UTC+05:30")
    assert not is_known_timezone("UTC+15")
    assert not is_known_timezone("Mars/Olympus_Mons")


def test_aware_instants_are_converted_not_relabelled() -> None:
    # 05:30 at UTC+05:00 is 00:30 UTC, before Madrid's 01:00 UTC switch to summer time on 2025-03-30.
    instant = datetime(2025, 3, 30, 5, 30, tzinfo=timezone(timedelta(hours=5)))
    assert parse_utc_offset("Europe/Madrid", at=instant) == 1
    assert parse_utc_offset("Europe/Madrid", at=instant.astimezone(UTC).replace(tzinfo=None)) == 1


@pytest.mark.parametrize(
    ("cache_path", "changed_source"),
    [
        ("base", "constants.py"),
        ("timezone", "constants.py"),
        ("timezone", "boundaries.py"),
        ("population", "constants.py"),
    ],
)
def test_prepared_cache_keys_track_the_source_files_they_depend_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_path: str, changed_source: str
) -> None:
    data_file = tmp_path / "input.geojson"
    data_file.write_text("{}", encoding="utf-8")

    def key() -> Path:
        if cache_path == "base":
            return prepared_cache.base_cache_path(tmp_path, data_file, ExperimentConfig())
        if cache_path == "timezone":
            return prepared_cache.timezone_cache_path(tmp_path, data_file, data_file)
        return prepared_cache.population_cache_path(tmp_path, data_file, data_file)

    before = key()
    original = fingerprint.file_digest
    monkeypatch.setattr(
        fingerprint, "file_digest", lambda path: "edited" if path.name == changed_source else original(path)
    )
    assert key() != before


def test_population_metadata_never_outlives_a_replaced_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "population_2025.csv"
    metadata_path = csv_path.with_suffix(".source.json")
    csv_path.write_text("latitude,longitude,population\n0.5,0.5,1\n", encoding="utf-8")
    metadata_path.write_text(json.dumps({"cell_degrees": 1.0}), encoding="utf-8")
    with ZipFile(tmp_path / "ghsl_population_2025.zip", "w") as archive:
        archive.writestr("population.tif", b"raster")

    def convert(_raster: Path, target: Path, **_options: object) -> tuple[int, float]:
        target.write_text("latitude,longitude,population\n0.25,0.25,1\n", encoding="utf-8")
        return 1, 1.0

    def crash(*_args: object, **_kwargs: object) -> str:
        msg = "simulated crash before metadata is written"
        raise RuntimeError(msg)

    monkeypatch.setattr(download_population, "population_raster_to_csv", convert)
    monkeypatch.setattr(download_population.json, "dumps", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        download_population.download_population(tmp_path, cell_degrees=0.5)
    assert "0.25,0.25" in csv_path.read_text(encoding="utf-8")
    assert not metadata_path.exists()


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
