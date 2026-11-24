"""Subdivision boundary downloads."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from timezone_experiments.data.download import boundaries as download_boundaries
from timezone_experiments.data.download.catalog import SOURCES, read_source_metadata

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("country", ["../x", "US", "USA/1", ""])
def test_boundary_download_rejects_unsafe_country_codes_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, country: str
) -> None:
    monkeypatch.setattr(download_boundaries, "download_bytes", pytest.fail)
    with pytest.raises(ValueError, match="Country codes must be three"):
        download_boundaries.download("gadm", country, 1, tmp_path)
    assert not any(tmp_path.iterdir())


_EMPTY_COLLECTION = b'{"type":"FeatureCollection","features":[]}'


def test_gadm_boundaries_are_saved_under_the_normalized_country_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download_boundaries, "download_bytes", lambda url: _EMPTY_COLLECTION)
    path = download_boundaries.download("gadm", "usa", 1, tmp_path)
    assert path.read_bytes() == _EMPTY_COLLECTION
    assert "USA" in path.name
    assert f"gadm-{SOURCES['gadm'].version}" in path.name
    metadata = read_source_metadata(path)
    assert metadata is not None
    assert metadata["url"] == SOURCES["gadm"].resolve(country="USA", level=1)


def test_geoboundaries_follow_the_metadata_download_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []
    answer = {
        "gjDownloadURL": "https://raw.githubusercontent.com/data.geojson",
        "boundaryID": "USA-ADM1-123",
        "buildDate": "Dec 12, 2023",
        "boundaryLicense": "Public Domain",
        "unrelated": "not recorded",
    }
    responses = iter((json.dumps(answer).encode(), _EMPTY_COLLECTION))

    def fetch(url: str) -> bytes:
        requested.append(url)
        return next(responses)

    monkeypatch.setattr(download_boundaries, "download_bytes", fetch)
    path = download_boundaries.download("geoboundaries", "USA", 1, tmp_path)
    assert path.read_bytes() == _EMPTY_COLLECTION
    assert requested == [
        "https://www.geoboundaries.org/api/current/gbOpen/USA/ADM1/",
        "https://raw.githubusercontent.com/data.geojson",
    ]
    metadata = read_source_metadata(path)
    assert metadata is not None
    assert metadata["version"] is None
    assert metadata["url"] == "https://raw.githubusercontent.com/data.geojson"
    assert metadata["api_url"] == requested[0]
    assert metadata["boundaryID"] == "USA-ADM1-123"
    assert metadata["buildDate"] == "Dec 12, 2023"
    assert metadata["sourceDataUpdateDate"] is None
    assert "unrelated" not in metadata
    assert (metadata["country"], metadata["level"]) == ("USA", 1)


@pytest.mark.parametrize(
    ("source", "level", "message"),
    [("unknown", 0, "Unsupported"), ("gadm", -1, "Administrative level")],
    ids=["unknown-source", "negative-level"],
)
def test_invalid_boundary_requests_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, level: int, message: str
) -> None:
    monkeypatch.setattr(download_boundaries, "download_bytes", pytest.fail)
    with pytest.raises(ValueError, match=message):
        download_boundaries.download(source, "USA", level, tmp_path)


@pytest.mark.parametrize("metadata", [b"{}", b'{"gjDownloadURL": ""}', b'{"gjDownloadURL": 1}', b"[]"])
def test_geoboundaries_metadata_without_a_download_url_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: bytes
) -> None:
    monkeypatch.setattr(download_boundaries, "download_bytes", lambda url: metadata)
    with pytest.raises(ValueError, match="did not return a GeoJSON download URL for USA ADM1"):
        download_boundaries.download("geoboundaries", "USA", 1, tmp_path)
    assert not any(tmp_path.iterdir())
