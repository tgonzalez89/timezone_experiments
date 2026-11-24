"""Developer script that reports newer dataset releases and rewrites the pins."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import update_data_sources

from timezone_experiments.data.download.catalog import SOURCES_PATH
from timezone_experiments.data.download.http import DownloadError

if TYPE_CHECKING:
    from pathlib import Path

    from timezone_experiments.data.download.catalog import DataSource

_NEWER = {"natural_earth": "v6.0.0", "timezone_boundary_builder": "2030a"}


@pytest.fixture
def pins(tmp_path: Path) -> Path:
    path = tmp_path / "sources.json"
    path.write_bytes(SOURCES_PATH.read_bytes())
    return path


def _latest(source: DataSource) -> str | None:
    if source.key == "gadm":
        url = "https://api.github.com"
        raise DownloadError(url, "offline")
    return None if source.version is None else _NEWER.get(source.key, source.version)


def test_newer_releases_are_reported_without_changing_the_pins(
    pins: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    before = pins.read_bytes()
    monkeypatch.setattr(update_data_sources, "latest_version", _latest)
    monkeypatch.setattr(update_data_sources, "url_exists", lambda url: True)
    assert update_data_sources.main(["--sources", str(pins)]) == 1
    output = capsys.readouterr().out
    assert "Natural Earth 1:50m admin-0 countries: newer release v6.0.0 (pinned: v5.1.2)" in output
    assert "GADM administrative areas: check failed (Download failed for https://api.github.com: offline)" in output
    assert "geoBoundaries gbOpen: not pinnable" in output
    assert "GHS-POP population counts, epoch 2025, WGS84 30 arcseconds: up to date (R2023A)" in output
    assert "run with --write" in output
    assert pins.read_bytes() == before


def test_write_updates_only_releases_whose_download_url_exists(
    pins: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(update_data_sources, "latest_version", _latest)
    monkeypatch.setattr(update_data_sources, "url_exists", lambda url: "natural-earth" in url)
    assert update_data_sources.main(["--sources", str(pins), "--write"]) == 1  # GADM and the TBB URL failed.
    assert "2030a is released but" in capsys.readouterr().out
    written = json.loads(pins.read_text(encoding="utf-8"))
    assert written["natural_earth"]["version"] == "v6.0.0"
    assert (
        written["timezone_boundary_builder"]["version"]
        == json.loads(SOURCES_PATH.read_bytes())["timezone_boundary_builder"]["version"]
    )
    assert b"\r\n" not in pins.read_bytes()
    assert "©" in pins.read_text(encoding="utf-8")


def test_current_pins_succeed(pins: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update_data_sources, "latest_version", lambda source: source.version)
    assert update_data_sources.main(["--sources", str(pins), "--write"]) == 0
    assert pins.read_bytes() == SOURCES_PATH.read_bytes()
