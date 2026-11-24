"""Dataset sources: world boundaries and timezone archives."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING
from zipfile import ZipFile

import pytest
from factories import (
    DatasetResponse,
)

from timezone_experiments.data.download import http as http_module
from timezone_experiments.data.download import sources as download_data

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _archive_response(files: dict[str, bytes]) -> Callable[..., DatasetResponse]:
    archive_bytes = io.BytesIO()
    with ZipFile(archive_bytes, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return lambda *args, **kwargs: DatasetResponse(archive_bytes.getvalue())


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
