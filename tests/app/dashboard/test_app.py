"""Dashboard script: running experiments from the controls and showing results."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import pytest
from factories import (
    TOY_BOUNDARIES,
    FakeStreamlit,
)

from timezone_experiments.app.dashboard import app as dashboard
from timezone_experiments.app.dashboard.inputs import dashboard_inputs
from timezone_experiments.app.dashboard.settings import build_config
from timezone_experiments.core.config import (
    ExperimentConfig,
)
from timezone_experiments.data.download.catalog import SOURCES, write_source_metadata
from timezone_experiments.data.paths import boundary_filename, timezones_filename, world_filename

if TYPE_CHECKING:
    from pathlib import Path


def test_streamlit_dashboard_upload_run_and_input_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toy_data_directory: Path
) -> None:
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"countries": ["TST"]}), encoding="utf-8")
    (toy_data_directory / "downloads" / timezones_filename()).unlink()  # Must come from the upload.
    world = toy_data_directory / "downloads" / world_filename(include_antarctica=False)
    write_source_metadata(world, SOURCES["natural_earth"], SOURCES["natural_earth"].resolve())
    arguments = ["--config", str(config), "--data-directory", str(toy_data_directory)]
    fake = FakeStreamlit(has_upload=False, run=False, rules="{}")
    monkeypatch.setattr(dashboard, "st", fake)
    dashboard.main(arguments)
    assert not fake.errors
    fake = FakeStreamlit(has_upload=True, run=True, rules='{"USA": 1}')
    monkeypatch.setattr(dashboard, "st", fake)
    dashboard.main(arguments)
    assert not fake.errors
    assert fake.downloads == ["Download results CSV"]
    assert f"Data: Natural Earth {SOURCES['natural_earth'].version}." in fake.captions
    # A rerun without pressing the button (e.g. after picking another map field) keeps showing the last results.
    fake.run = False
    dashboard.main(arguments)
    assert fake.downloads == ["Download results CSV"] * 2
    fake = FakeStreamlit(has_upload=True, run=True, rules="not json")
    monkeypatch.setattr(dashboard, "st", fake)
    dashboard.main(arguments)
    assert fake.errors


def test_dashboard_refuses_concurrent_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeStreamlit(has_upload=True, run=True, rules="{}")
    monkeypatch.setattr(dashboard, "st", fake)
    with dashboard._RUN_LOCK:
        dashboard.main(["--data-directory", str(tmp_path)])
    assert fake.warnings == ["Another experiment is running on this server; try again when it finishes."]
    assert not fake.downloads
    assert not dashboard._RUN_LOCK.locked()


@pytest.mark.slow
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
    fake = FakeStreamlit(has_upload=False, run=True, rules='{"TST": 1}')
    inputs = dashboard_inputs(fake, config)
    assert build_config(inputs, config) == config
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    (tmp_path / "downloads").mkdir()
    (tmp_path / "downloads" / world_filename(include_antarctica=False)).write_bytes(TOY_BOUNDARIES.read_bytes())
    (tmp_path / "downloads" / boundary_filename("gadm", "TST", 1)).write_bytes(TOY_BOUNDARIES.read_bytes())
    monkeypatch.setattr(dashboard, "st", fake)
    dashboard.main(["--config", str(path), "--data-directory", str(tmp_path)])
    assert not fake.errors
    assert fake.downloads == ["Download results CSV"]
