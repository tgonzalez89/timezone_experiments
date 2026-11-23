"""Verify dataset downloads: retries, error context, redirect checks, and bounded extraction."""

from __future__ import annotations

import http.client
import io
import urllib.request
from email.message import Message
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError
from zipfile import ZipFile

import pytest

from timezone_experiments.data.download import http as http_module

if TYPE_CHECKING:
    from pathlib import Path

URL = "https://geodata.ucdavis.edu/data.json"


class _Response(io.BytesIO):
    def __init__(self, content: bytes, *, declared_length: int | None = None) -> None:
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content) if declared_length is None else declared_length)}


def _http_error(code: int) -> HTTPError:
    return HTTPError(URL, code, "error", Message(), None)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(http_module.time, "sleep", delays.append)
    return delays


def _serve(monkeypatch: pytest.MonkeyPatch, *outcomes: Exception | bytes | _Response) -> list[str]:
    """Answer successive requests with the given errors or payloads, recording requested URLs."""
    pending = list(outcomes)
    requested: list[str] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        del timeout
        requested.append(request.full_url)
        outcome = pending.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if isinstance(outcome, _Response) else _Response(outcome)

    monkeypatch.setattr(http_module, "_urlopen", urlopen)
    return requested


@pytest.mark.parametrize(
    "transient",
    [
        URLError("connection reset"),
        TimeoutError("read timed out"),
        http.client.IncompleteRead(b"par", 10),
        _http_error(503),
        _http_error(429),
        _Response(b"partial", declared_length=100),
    ],
    ids=["url-error", "timeout", "incomplete-read", "http-503", "http-429", "short-body"],
)
def test_transient_failures_are_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], transient: Exception | _Response
) -> None:
    requested = _serve(monkeypatch, transient, b"payload")
    assert http_module.download_bytes(URL) == b"payload"
    assert requested == [URL, URL]
    assert sleeps == [2.0]


def test_client_errors_fail_immediately_with_url_and_status(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    requested = _serve(monkeypatch, _http_error(404))
    with pytest.raises(
        http_module.DownloadError, match=r"Download failed for https://geodata\.ucdavis\.edu/data\.json: HTTP 404"
    ) as error:
        http_module.download_bytes(URL)
    assert error.value.status == 404
    assert requested == [URL]
    assert sleeps == []


def test_retries_give_up_after_three_attempts(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    _serve(monkeypatch, _http_error(503), _http_error(503), _http_error(503))
    with pytest.raises(http_module.DownloadError) as error:
        http_module.download_bytes(URL)
    assert error.value.status == 503
    assert sleeps == [2.0, 8.0]


def test_failed_file_download_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    del sleeps
    target = tmp_path / "data.json"
    target.write_bytes(b"previous")
    _serve(monkeypatch, *(_Response(b"partial", declared_length=100) for _ in range(3)))
    with pytest.raises(http_module.DownloadError, match="Incomplete transfer"):
        http_module.download_file(URL, target)
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize(
    ("target", "allowed"),
    [
        ("https://objects.githubusercontent.com/asset.zip", True),
        ("http://raw.githubusercontent.com/downgraded.json", False),
        ("https://evil.example/data.json", False),
    ],
)
def test_redirects_are_restricted_to_https_dataset_hosts(*, target: str, allowed: bool) -> None:
    handler = http_module._AllowedHostRedirectHandler()
    request = urllib.request.Request(URL)

    def redirect() -> urllib.request.Request | None:
        return handler.redirect_request(request, io.BytesIO(), 302, "Found", Message(), target)

    if allowed:
        assert redirect() is not None
    else:
        with pytest.raises(ValueError, match="must use HTTPS and a supported host"):
            redirect()


def test_extraction_rejects_oversized_members_and_keeps_the_target(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("data.geojson", b"x" * 100)
    target = tmp_path / "data.geojson"
    target.write_bytes(b"previous")
    with pytest.raises(ValueError, match="declares 100 bytes"):
        http_module.extract_single_member(
            archive_path, frozenset({".geojson"}), target, label="Extracting", max_bytes=10
        )
    assert target.read_bytes() == b"previous"
    http_module.extract_single_member(archive_path, frozenset({".geojson"}), target, label="Extracting", max_bytes=100)
    assert target.read_bytes() == b"x" * 100
