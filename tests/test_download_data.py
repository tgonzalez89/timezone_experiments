"""Tests for preparing global experiment inputs without live network access."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING
from zipfile import ZipFile

import pytest

from timezone_experiments.app import cli
from timezone_experiments.data import preparation
from timezone_experiments.data.download import http as http_module
from timezone_experiments.data.download import sources as download_data

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


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


@pytest.fixture
def country_config(tmp_path: Path) -> Path:
    path = tmp_path / "country_config.json"
    path.write_text(json.dumps({"subdivision_rules": {}, "merged_subdivision_levels": {}}), encoding="utf-8")
    return path


def test_world_download_excludes_antarctica_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world_payload: bytes,
) -> None:
    monkeypatch.setattr(download_data, "download_bytes", lambda url: world_payload)
    path = download_data.download_world_boundaries(tmp_path)
    collection = json.loads(path.read_text(encoding="utf-8"))
    assert [feature["properties"]["ADM0_A3"] for feature in collection["features"]] == ["USA"]
    path = download_data.download_world_boundaries(tmp_path, include_antarctica=True)
    assert len(json.loads(path.read_text(encoding="utf-8"))["features"]) == 2


def test_world_download_rejects_invalid_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(download_data, "download_bytes", lambda url: b'{"type":"FeatureCollection","features":[]}')
    with pytest.raises(ValueError, match="non-empty"):
        download_data.download_world_boundaries(tmp_path)
    assert not (tmp_path / "world_countries.geojson").exists()


class DatasetResponse(io.BytesIO):
    def __init__(self, content: bytes, *, known_length: bool = True) -> None:
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content))} if known_length else {}
        self.status = 200


def _archive_response(files: dict[str, bytes]) -> Callable[..., DatasetResponse]:
    archive_bytes = io.BytesIO()
    with ZipFile(archive_bytes, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return lambda *args, **kwargs: DatasetResponse(archive_bytes.getvalue())


@pytest.mark.parametrize("to_file", [False, True])
@pytest.mark.parametrize("known_length", [False, True])
def test_download_progress_preserves_chunked_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    to_file: bool,
    known_length: bool,
) -> None:
    payload = b"GeoJSON payload\n" * 100_000
    monkeypatch.setattr(
        http_module,
        "_urlopen",
        lambda *args, **kwargs: DatasetResponse(payload, known_length=known_length),
    )
    url = "https://geodata.ucdavis.edu/test.geojson"
    if to_file:
        path = tmp_path / "test.geojson"
        http_module.download_file(url, path)
        actual = path.read_bytes()
    else:
        actual = http_module.download_bytes(url)
    assert actual == payload
    output = capsys.readouterr().err
    assert ("100.0%" in output) == known_length
    assert ("total unknown" in output) != known_length
    assert "MiB" in output


def test_timezone_download_unpacks_only_geojson(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b'{"type":"FeatureCollection","features":[]}'
    monkeypatch.setattr(
        http_module,
        "_urlopen",
        _archive_response({"combined-now.json": payload, "README.txt": b"Source information"}),
    )
    path = download_data.download_timezones(tmp_path)
    assert path.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [path]


def test_timezone_download_preserves_existing_file_on_invalid_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "world_timezones.geojson"
    path.write_bytes(b"existing dataset")
    monkeypatch.setattr(http_module, "_urlopen", _archive_response({"README.txt": b"no geometry"}))
    with pytest.raises(ValueError, match="exactly one"):
        download_data.download_timezones(tmp_path)
    assert path.read_bytes() == b"existing dataset"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("with_antarctica", [False, True])
def test_download_command_prepares_and_reports_requested_inputs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    world_payload: bytes,
    country_config: Path,
    *,
    with_antarctica: bool,
) -> None:
    tmp_path = country_config.parent
    monkeypatch.setattr(download_data, "download_bytes", lambda url: world_payload)
    monkeypatch.setattr(preparation, "download_timezones", lambda directory: directory / "world_timezones.geojson")
    settings = json.loads(country_config.read_text(encoding="utf-8"))
    settings["include_antarctica"] = with_antarctica
    country_config.write_text(json.dumps(settings), encoding="utf-8")
    assert cli.main(["download", "--data-directory", str(tmp_path), "--config", str(country_config)]) == 0
    output = capsys.readouterr().out
    filename = "world_countries_with_antarctica.geojson" if with_antarctica else "world_countries.geojson"
    assert f"Boundaries: {tmp_path / 'downloads' / filename}" in output
    assert "Timezones:" in output
    assert "Population:" not in output


def test_population_download_command_uses_requested_resolution_and_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    world_payload: bytes,
    country_config: Path,
) -> None:
    monkeypatch.setattr(download_data, "download_bytes", lambda url: world_payload)
    settings = json.loads(country_config.read_text(encoding="utf-8"))
    settings |= {"workers": 3, "coverage_type": "population", "population_cell_degrees": 0.5}
    country_config.write_text(json.dumps(settings), encoding="utf-8")
    calls: list[tuple[float, int]] = []

    def prepare(directory: Path, cell_degrees: float, *, refresh: bool, workers: int) -> Path:
        assert not refresh
        calls.append((cell_degrees, workers))
        return directory / "population_2025.csv"

    monkeypatch.setattr(preparation, "prepare_population", prepare)
    assert cli.main(["download", "--data-directory", str(tmp_path), "--config", str(country_config)]) == 0
    assert calls == [(0.5, 3)]
    output = capsys.readouterr().out
    assert f"Population: {tmp_path / 'downloads' / 'population_2025.csv'}" in output
