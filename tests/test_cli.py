"""Tests for the timezone_experiments CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from subprocess import CompletedProcess
from typing import TYPE_CHECKING, Any

import pytest

from timezone_experiments.app import cli
from timezone_experiments.app.cli import build_parser, main
from timezone_experiments.core.config import ExperimentConfig, load_config
from timezone_experiments.data import preparation as preparation_module
from timezone_experiments.data.boundaries import load_territories
from timezone_experiments.data.download import sources as download_data
from timezone_experiments.data.preparation import InputOptions, prepare_boundaries
from timezone_experiments.experiments.results_io import read_results_config, results_config_path, write_results_csv
from timezone_experiments.experiments.runner import run_experiment
from timezone_experiments.maps import geometry_feature_collection

if TYPE_CHECKING:
    from timezone_experiments.maps import MapOptions


def test_main_no_args_returns_zero() -> None:
    assert main([]) == 0


def test_parser_exposes_unified_commands() -> None:
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    assert subparsers.choices is not None
    assert set(subparsers.choices) == {"run", "maps", "download", "clean", "dashboard"}


def test_run_selects_configured_countries_from_cached_inputs(tmp_path: Path, toy_data_directory: Path) -> None:
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"countries": ["tst"]}), encoding="utf-8")
    output = tmp_path / "results.csv"
    args = ["--config", str(config), "--data-directory", str(toy_data_directory), "--results", str(output)]
    assert main(["run", *args]) == 0
    assert read_results_config(output) == ExperimentConfig(countries=("TST",))
    assert "workers" not in json.loads(results_config_path(output).read_text(encoding="utf-8"))
    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["coverage_type_used"] == "area"
    assert float(rows[0]["compliance_rate"]) == pytest.approx(23 / 24)


def test_run_exports_all_maps_with_configured_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toy_data_directory: Path
) -> None:
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"workers": 3, "countries": ["TST"]}), encoding="utf-8")
    calls: list[tuple[Path, int]] = []

    def export(collection: dict[str, Any], directory: Path, options: MapOptions) -> list[Path]:
        assert len(collection["features"]) == 1
        calls.append((directory, options.workers))
        return []

    monkeypatch.setattr(cli, "export_maps", export)
    output = tmp_path / "csv" / "results.csv"
    assert (
        main(
            [
                "run",
                "--data-directory",
                str(toy_data_directory),
                "--config",
                str(config),
                "--results",
                str(output),
                "--maps",
                "--maps-dir",
                str(tmp_path / "maps"),
            ]
        )
        == 0
    )
    assert calls == [(tmp_path / "maps", 3)]


@pytest.mark.parametrize("cached", [False, True])
def test_maps_reuses_saved_results_without_calculations_or_overlays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toy_data_directory: Path, *, cached: bool
) -> None:
    boundaries = Path(__file__).parent / "data" / "toy_boundaries.geojson"
    rules = {} if cached else {"TST": 1}
    config = ExperimentConfig(subdivision_rules=rules, merged_subdivision_levels={}, workers=1, sample_grid_size=2)
    territories = load_territories(boundaries, config)
    results = run_experiment(config, territories, on_territory_done=lambda *_: None)
    csv_path = write_results_csv(results, tmp_path / "results.csv")
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "workers": 3,
                "countries": ["TST"],
                "subdivision_rules": rules,
                "merged_subdivision_levels": {},
                "coverage_type": "population",
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[Path, int]] = []

    def no_calculations(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Maps must not run sampling, optimization or population/timezone preparation.")

    def export(collection: dict[str, Any], directory: Path, options: MapOptions) -> list[Path]:
        assert collection == geometry_feature_collection(
            {territory.territory_id: territory.geometry for territory in territories if territory.geometry is not None},
            results,
        )
        calls.append((directory, options.workers))
        return []

    monkeypatch.setattr(cli, "run_experiment", no_calculations)
    monkeypatch.setattr(cli, "load_experiment_territories", no_calculations)
    monkeypatch.setattr(preparation_module, "prepare_inputs", no_calculations)
    monkeypatch.setattr(preparation_module, "_prepare_population", no_calculations)
    monkeypatch.setattr(preparation_module, "download_timezones", no_calculations)
    monkeypatch.setattr(preparation_module, "download_world_boundaries", no_calculations)
    monkeypatch.setattr(cli, "export_maps", export)
    if cached:
        prepare_boundaries(load_config(settings), InputOptions(data_directory=toy_data_directory))
        monkeypatch.setattr(preparation_module, "_mixed_features", no_calculations)
    output = tmp_path / "maps"
    assert (
        main(
            [
                "maps",
                "--results",
                str(csv_path),
                "--config",
                str(settings),
                "--data-directory",
                str(toy_data_directory),
                "--maps-dir",
                str(output),
            ]
        )
        == 0
    )
    assert calls == [(output, 3)]


def test_maps_validates_saved_results_before_preparing_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_boundaries(*_args: object) -> None:
        pytest.fail("Invalid results must be reported before preparing boundaries.")

    monkeypatch.setattr(cli, "prepare_boundaries", no_boundaries)
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text("territory_id\nTST:1\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["maps", "--results", str(csv_path)])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "missing columns" in message
    assert "Traceback" not in message


@pytest.mark.parametrize("command", ["run", "maps", "download", "clean", "dashboard"])
def test_options_with_a_default_document_it(command: str) -> None:
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    assert isinstance(subparsers.choices, dict)
    selected = subparsers.choices[command]
    assert selected.description
    for action in selected._actions:
        assert action.help, action.dest
        if action.dest != "help" and action.default not in {None, False}:
            assert "default:" in action.help, action.dest
    help_text = selected.format_help()
    assert "default: None" not in help_text
    assert "default: False" not in help_text


def test_cli_runs_with_automatically_downloaded_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toy = Path(__file__).parent / "data" / "toy_boundaries.geojson"
    monkeypatch.setattr(download_data, "download_bytes", lambda url: toy.read_bytes())
    config = tmp_path / "country_config.json"
    config.write_text(json.dumps({"subdivision_rules": {}, "merged_subdivision_levels": {}}), encoding="utf-8")
    output = tmp_path / "results.csv"
    args = ["--data-directory", str(tmp_path), "--config", str(config)]
    assert main(["run", *args, "--results", str(output)]) == 0
    assert output.is_file()


def test_cli_reports_input_errors_without_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--population", str(tmp_path / "missing.csv")])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "Explicit input file does not exist" in message
    assert "Traceback" not in message


def test_dashboard_command_forwards_configuration_and_propagates_exit_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    config = tmp_path / "settings.json"
    config.write_text("{}", encoding="utf-8")

    def launch(command: list[str], *, check: bool) -> CompletedProcess[str]:
        assert not check
        calls.append(command)
        return CompletedProcess(command, 7)

    monkeypatch.setattr(cli.subprocess, "run", launch)
    assert main(["dashboard", "--config", str(config), "--port", "8510", "--headless"]) == 7
    command = calls[0]
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert command[command.index("--server.port") + 1] == "8510"
    assert command[command.index("--server.headless") + 1] == "true"
    forwarded = command[command.index("--") + 1 :]
    assert "--refresh" not in forwarded
    assert forwarded[forwarded.index("--config") + 1] == str(config)


def test_clean_command_selects_targets_and_supports_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    downloads = tmp_path / "data" / "downloads"
    cache = tmp_path / "data" / "cache"
    csv = tmp_path / "outputs" / "csv"
    maps = tmp_path / "outputs" / "maps"
    for directory in (downloads, cache, csv, maps):
        directory.mkdir(parents=True)
        (directory / "artifact.txt").write_text("data", encoding="utf-8")
    assert (
        main(
            [
                "clean",
                "--downloads",
                "--csv",
                "--dry-run",
                "--data-directory",
                str(tmp_path / "data"),
                "--output-directory",
                str(tmp_path / "outputs"),
            ]
        )
        == 0
    )
    assert downloads.exists()
    assert csv.exists()
    assert cache.exists()
    assert maps.exists()
    assert "Would remove (downloads)" in capsys.readouterr().out
    assert (
        main(
            [
                "clean",
                "--downloads",
                "--cache",
                "--csv",
                "--data-directory",
                str(tmp_path / "data"),
                "--output-directory",
                str(tmp_path / "outputs"),
            ]
        )
        == 0
    )
    assert not downloads.exists()
    assert not cache.exists()
    assert not csv.exists()
    assert maps.exists()


def test_clean_outputs_and_all_require_a_target(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "outputs"
    for directory in (output / "csv", output / "maps"):
        directory.mkdir(parents=True)
    assert main(["clean", "--csv", "--maps", "--output-directory", str(output)]) == 0
    assert output.is_dir()
    assert not any(output.iterdir())
    downloads = tmp_path / "data" / "downloads"
    cache = tmp_path / "data" / "cache"
    for directory in (downloads, cache, output / "csv", output / "maps"):
        directory.mkdir(parents=True)
    assert (
        main(
            [
                "clean",
                "--all",
                "--data-directory",
                str(tmp_path / "data"),
                "--output-directory",
                str(output),
            ]
        )
        == 0
    )
    assert not downloads.exists()
    assert not cache.exists()
    assert (tmp_path / "data").is_dir()  # Only downloads/ and cache/ are removed from the data directory.
    assert not output.exists()
    with pytest.raises(SystemExit):
        main(["clean", "--data-directory", str(tmp_path / "data")])
    assert "Select at least one clean target" in capsys.readouterr().err


def test_run_maps_dir_requires_maps(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["run", "--maps-dir", str(tmp_path / "maps"), "--data-directory", str(tmp_path)])
    assert "add --maps" in capsys.readouterr().err
    assert not tmp_path.joinpath("downloads").exists()


def test_maps_uses_the_saved_configuration_and_rejects_different_territories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toy_data_directory: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_config = tmp_path / "run.json"
    run_config.write_text(json.dumps({"countries": ["TST"], "sample_grid_size": 2}), encoding="utf-8")
    results = tmp_path / "results.csv"
    data = ["--data-directory", str(toy_data_directory)]
    assert main(["run", "--config", str(run_config), "--results", str(results), *data]) == 0
    exported: list[MapOptions] = []

    def export(collection: dict[str, Any], directory: Path, options: MapOptions) -> list[Path]:
        assert len(collection["features"]) == 1
        exported.append(options)
        return []

    monkeypatch.setattr(cli, "export_maps", export)
    maps = ["maps", "--results", str(results), "--maps-dir", str(tmp_path / "maps"), *data]
    # The built-in defaults would ask for countries absent from the toy data, so success means the saved one was used.
    assert main(maps) == 0

    map_settings = tmp_path / "map_settings.json"
    map_settings.write_text(json.dumps({"countries": ["TST"], "event_map_cell_degrees": 1.0}), encoding="utf-8")
    assert main([*maps, "--config", str(map_settings)]) == 0
    assert [options.event_map_cell_degrees for options in exported] == [0.5, 1.0]

    other = tmp_path / "other.json"
    other.write_text(json.dumps({"countries": ["TST"], "subdivision_rules": {"TST": 1}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main([*maps, "--config", str(other)])
    error = capsys.readouterr().err
    assert 'subdivision_rules ({"TST": 1} instead of {' in error
    assert "countries" not in error.split("defines different territories")[1].split("Omit")[0]
    assert len(exported) == 2


@pytest.mark.parametrize(("saved", "message"), [("[]", "must be a JSON object"), ('{"year": 1}', "Invalid saved")])
def test_maps_reports_an_unusable_saved_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], saved: str, message: str
) -> None:
    results = tmp_path / "results.csv"
    territories = load_territories(Path(__file__).parent / "data" / "toy_boundaries.geojson", ExperimentConfig())
    write_results_csv(run_experiment(ExperimentConfig(), territories, on_territory_done=lambda *_: None), results)
    results_config_path(results).write_text(saved, encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["maps", "--results", str(results), "--data-directory", str(tmp_path)])
    assert message in capsys.readouterr().err
