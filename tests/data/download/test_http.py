"""HTTP downloads: retries, redirects, progress, and extraction limits."""

from __future__ import annotations

import http.client
import io
import urllib.request
from email.message import Message
from http.client import HTTPMessage
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError
from zipfile import ZipFile

import pytest
from factories import (
    DatasetResponse,
)

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


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(http_module.time, "sleep", delays.append)
    return delays


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
        return handler.redirect_request(request, io.BytesIO(), 302, "Found", HTTPMessage(), target)

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


def test_plain_http_urls_are_refused_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_module, "_urlopen", pytest.fail)
    with pytest.raises(ValueError, match="must use HTTPS"):
        http_module.download_bytes("http://example.invalid/data.json")


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_redirects_keep_head_requests_so_probes_do_not_download(method: str) -> None:
    handler = http_module._AllowedHostRedirectHandler()
    request = urllib.request.Request(URL, method=method)
    redirected = handler.redirect_request(
        request, io.BytesIO(), 302, "Found", HTTPMessage(), "https://objects.githubusercontent.com/asset.zip"
    )
    assert redirected is not None
    assert redirected.get_method() == method


def test_metadata_is_fetched_in_one_attempt(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    requested = _serve(monkeypatch, b'{"tag_name": "v1"}')
    assert http_module.fetch_metadata(URL) == b'{"tag_name": "v1"}'
    _serve(monkeypatch, _http_error(503))
    with pytest.raises(http_module.DownloadError, match="HTTP 503") as caught:
        http_module.fetch_metadata(URL)
    assert caught.value.status == 503
    assert requested == [URL]
    assert sleeps == []


@pytest.mark.parametrize(
    ("outcome", "exists"),
    [(b"", True), (_http_error(404), False)],
    ids=["published", "missing"],
)
def test_url_exists_probes_with_head(
    monkeypatch: pytest.MonkeyPatch, outcome: Exception | bytes, *, exists: bool
) -> None:
    methods: list[str] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        del timeout
        methods.append(request.get_method())
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)

    monkeypatch.setattr(http_module, "_urlopen", urlopen)
    assert http_module.url_exists(URL) is exists
    assert methods == ["HEAD"]


@pytest.mark.parametrize("error", [_http_error(500), URLError("offline")], ids=["server-error", "offline"])
def test_url_exists_reports_errors_other_than_not_found(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    _serve(monkeypatch, error)
    with pytest.raises(http_module.DownloadError, match=URL):
        http_module.url_exists(URL)
