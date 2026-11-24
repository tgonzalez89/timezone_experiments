"""Shared pytest fixtures for timezone_experiments tests."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest
from factories import TOY_BOUNDARIES

from timezone_experiments.data.download.versions import UPDATE_CHECK_ENV
from timezone_experiments.data.paths import boundary_filename, downloads_directory, timezones_filename, world_filename

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _no_release_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep downloads from asking GitHub for newer dataset releases; the check has its own tests."""
    monkeypatch.setenv(UPDATE_CHECK_ENV, "1")


@pytest.fixture
def toy_data_directory(tmp_path: Path) -> Path:
    """Return a data directory whose downloads are the offline toy dataset, so no download happens.

    The toy file stands in for the Natural Earth world countries, the GADM ``TST`` level-1 subdivisions, and the
    timezone boundaries.
    """
    data = tmp_path / "data"
    downloads = downloads_directory(data)
    downloads.mkdir(parents=True)
    for name in (world_filename(include_antarctica=False), boundary_filename("gadm", "TST", 1), timezones_filename()):
        shutil.copyfile(TOY_BOUNDARIES, downloads / name)
    return data


@pytest.fixture
def world_payload() -> bytes:
    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"ADM0_A3": "USA", "ISO_A3": "USA"}},
                {"type": "Feature", "properties": {"ADM0_A3": "ATA", "ISO_A3": "ATA"}},
            ],
        },
    ).encode()
