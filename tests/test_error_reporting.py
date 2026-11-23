"""Verify failures name the territory involved and reach users as messages (or tracebacks with --debug)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from timezone_experiments import cli
from timezone_experiments.config import ExperimentConfig
from timezone_experiments.data.models import Territory, TerritorySample
from timezone_experiments.data.population import PopulationPoint, attach_population_weights
from timezone_experiments.errors import describe_error
from timezone_experiments.experiments.runner import run_experiment

if TYPE_CHECKING:
    from pathlib import Path

SQUARE: dict[str, Any] = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}


def _territory(identifier: str, geometry: dict[str, Any] | None) -> Territory:
    return Territory(
        f"TST:{identifier}", "TST", f"Place {identifier}", 1, (TerritorySample(0.5, 0.5),), geometry=geometry
    )


def test_optimization_failures_name_the_territory() -> None:
    territories = [_territory("good", SQUARE), _territory("bad", {"type": "Point", "coordinates": [0, 0]})]
    with pytest.raises(ValueError, match="Expected Polygon or MultiPolygon") as error:
        run_experiment(ExperimentConfig(), territories)
    assert "While optimizing territory TST:bad (Place bad)." in describe_error(error.value)


@pytest.mark.parametrize("workers", [1, 2])
def test_population_overlay_failures_name_the_territory(workers: int) -> None:
    territories = [_territory("good", SQUARE), _territory("bad", None)]
    with pytest.raises(ValueError, match="requires geometry for 'TST:bad'"):
        attach_population_weights(territories, [PopulationPoint(0.5, 0.5, 10)], workers=workers)


def test_cli_prints_user_errors_with_notes_and_exit_code_2(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"year": 1066}', encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "--config", str(config)])
    assert exit_info.value.code == 2
    assert "config.json: year must be between 1900 and 2200" in capsys.readouterr().err


def test_cli_debug_flag_shows_the_traceback(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        cli.main(["--debug", "run", "--config", str(missing)])


def test_describe_error_appends_notes() -> None:
    error = ValueError("broken")
    error.add_note("While doing something.")
    assert describe_error(error) == "broken\nWhile doing something."
