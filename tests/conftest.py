"""Shared pytest fixtures for timezone_experiments tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from timezone_experiments.data.paths import TIMEZONES_FILENAME, boundary_filename, downloads_directory, world_filename

TOY_BOUNDARIES = Path(__file__).parent / "data" / "toy_boundaries.geojson"


@pytest.fixture
def toy_data_directory(tmp_path: Path) -> Path:
    """Return a data directory whose downloads are the offline toy dataset, so no download happens.

    The toy file stands in for the Natural Earth world countries, the GADM ``TST`` level-1 subdivisions, and the
    timezone boundaries.
    """
    data = tmp_path / "data"
    downloads = downloads_directory(data)
    downloads.mkdir(parents=True)
    for name in (world_filename(include_antarctica=False), boundary_filename("gadm", "TST", 1), TIMEZONES_FILENAME):
        shutil.copyfile(TOY_BOUNDARIES, downloads / name)
    return data
