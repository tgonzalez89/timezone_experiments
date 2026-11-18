"""Tests for the timezone_experiments CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from subprocess import CompletedProcess
from typing import TYPE_CHECKING

import pytest

from timezone_experiments import cli
from timezone_experiments.cli import build_parser, main
from timezone_experiments.config import ExperimentConfig
from timezone_experiments.data import download_boundaries, download_data
from timezone_experiments.data import preparation as preparation_module
from timezone_experiments.data.boundaries import load_territories
from timezone_experiments.experiments.runner import run_experiment, write_results_csv

if TYPE_CHECKING:
    from timezone_experiments.data.models import Territory
    from timezone_experiments.optimization.optimizer import AssignmentResult


def test_main_no_args_returns_zero() -> None:
    assert main([]) == 0


def test_parser_exposes_unified_commands() -> None:
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    assert subparsers.choices is not None
    assert set(subparsers.choices) == {"run", "maps", "download", "download-boundaries", "dashboard"}


def test_run_uses_default_config_without_config_argument(tmp_path: Path) -> None:
    boundaries = Path(__file__).parents[1] / "examples" / "toy_boundaries.geojson"
    output = tmp_path / "results.csv"
    assert main(["run", "--boundaries", str(boundaries), "--output", str(output)]) == 0
    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["coverage_type_used"] == "longitude"
    assert float(rows[0]["compliance_rate"]) == 1


def test_run_exports_all_maps_with_configured_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    boundaries = Path(__file__).parents[1] / "examples" / "toy_boundaries.geojson"
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"workers": 3}), encoding="utf-8")
    calls: list[tuple[Path, int]] = []

    def export(
        territories: list[Territory], results: list[AssignmentResult], directory: Path, *, workers: int
    ) -> list[Path]:
        assert len(territories) == len(results) == 1
        calls.append((directory, workers))
        return []

    monkeypatch.setattr(cli, "export_maps", export)
    output = tmp_path / "csv" / "results.csv"
    assert (
        main(["run", "--boundaries", str(boundaries), "--config", str(config), "--output", str(output), "--maps"]) == 0
    )
    assert calls == [(tmp_path / "maps", 3)]


@pytest.mark.parametrize("cached", [False, True])
def test_maps_reuses_saved_results_without_calculations_or_overlays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cached: bool
) -> None:
    boundaries = Path(__file__).parents[1] / "examples" / "toy_boundaries.geojson"
    rules = {} if cached else {"TST": 1}
    config = ExperimentConfig(subdivision_rules=rules, country_boundary_levels={}, workers=1, sample_grid_size=2)
    territories = load_territories(boundaries, config)
    results = run_experiment(config, territories, progress=lambda *_: None)
    csv_path = write_results_csv(results, tmp_path / "results.csv")
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "workers": 3,
                "subdivision_rules": rules,
                "country_boundary_levels": {},
                "coverage_type": "population",
                "compare_current_timezones": True,
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[Path, int]] = []

    def no_calculations(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Maps must not run sampling, optimization or population/timezone preparation.")

    def export(
        geometries: dict[str, object], saved: list[AssignmentResult], directory: Path, *, workers: int
    ) -> list[Path]:
        assert saved == results
        assert geometries == {territory.territory_id: territory.geometry for territory in territories}
        calls.append((directory, workers))
        return []

    monkeypatch.setattr(cli, "run_experiment", no_calculations)
    monkeypatch.setattr(cli, "load_experiment_territories", no_calculations)
    monkeypatch.setattr(preparation_module, "prepare_inputs", no_calculations)
    monkeypatch.setattr(preparation_module, "_prepare_population", no_calculations)
    monkeypatch.setattr(preparation_module, "download_timezones", no_calculations)
    monkeypatch.setattr(preparation_module, "download_world_boundaries", no_calculations)
    monkeypatch.setattr(cli, "export_result_maps", export)
    if cached:
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "world_countries.geojson").write_bytes(boundaries.read_bytes())
        inputs = ["--data-directory", str(cache)]
    else:
        inputs = ["--boundaries", str(boundaries)]
    output = tmp_path / "maps"
    assert (
        main(
            [
                "maps",
                "--results",
                str(csv_path),
                "--config",
                str(settings),
                *inputs,
                "--output",
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


@pytest.mark.parametrize("command", ["run", "maps", "download", "download-boundaries", "dashboard"])
def test_every_option_documents_its_default(command: str) -> None:
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    assert isinstance(subparsers.choices, dict)
    selected = subparsers.choices[command]
    help_text = selected.format_help()
    options = [action for action in selected._actions if action.dest != "help"]
    assert help_text.count("(default:") == len(options)
    assert "default: None" not in help_text


def test_cli_runs_with_automatically_downloaded_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toy = Path(__file__).parents[1] / "examples" / "toy_boundaries.geojson"
    monkeypatch.setattr(download_data, "_download_bytes", lambda url: toy.read_bytes())
    config = tmp_path / "country_config.json"
    config.write_text(json.dumps({"subdivision_rules": {}, "country_boundary_levels": {}}), encoding="utf-8")
    output = tmp_path / "results.csv"
    args = ["--data-directory", str(tmp_path), "--config", str(config)]
    assert main(["run", *args, "--output", str(output)]) == 0
    assert output.is_file()


def test_cli_reports_input_errors_without_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--boundaries", str(tmp_path / "missing.geojson")])
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

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli.subprocess, "run", launch)
    assert main(["dashboard", "--config", str(config), "--port", "8510", "--headless", "--refresh"]) == 7
    command = calls[0]
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert command[command.index("--server.port") + 1] == "8510"
    assert command[command.index("--server.headless") + 1] == "true"
    forwarded = command[command.index("--") + 1 :]
    assert forwarded[-1] == "--refresh"
    assert forwarded[forwarded.index("--config") + 1] == str(config)


def test_dashboard_missing_extra_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(SystemExit):
        main(["dashboard"])
    assert "uv sync --extra visualization" in capsys.readouterr().err


def test_download_boundaries_reuses_cache_unless_refreshed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def payload(url: str) -> bytes:
        calls.append(url)
        return b'{"type":"FeatureCollection","features":[]}'

    monkeypatch.setattr(download_boundaries, "_download_bytes", payload)
    args = ["download-boundaries", "--data-directory", str(tmp_path)]
    assert main(args) == 0
    assert main(args) == 0
    assert len(calls) == 1
    assert main([*args, "--refresh"]) == 0
    assert len(calls) == 2
