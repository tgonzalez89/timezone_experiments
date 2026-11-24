"""HTTPS downloads restricted to known dataset hosts, with retries and single-member ZIP extraction."""

from __future__ import annotations

import http.client
import io
import logging
import time
import urllib.request
from pathlib import Path, PurePosixPath
from typing import IO, TYPE_CHECKING, cast, override
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from zipfile import ZipFile

from timezone_experiments.core.files import atomic_path
from timezone_experiments.core.progress import copy_with_progress

if TYPE_CHECKING:
    from collections.abc import Callable

ALLOWED_HOSTS = frozenset(
    {
        "geodata.ucdavis.edu",
        "www.geoboundaries.org",
        "raw.githubusercontent.com",
        "github.com",
        # GitHub serves release assets and LFS files (GeoBoundaries) from these hosts after a redirect.
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
        "media.githubusercontent.com",
        "jeodpp.jrc.ec.europa.eu",
    }
)
_TIMEOUT_SECONDS = 120
# Seconds to wait before the second and third attempts of a download that failed transiently.
_RETRY_DELAYS = (2.0, 8.0)
_SERVER_ERROR = 500
_TOO_MANY_REQUESTS = 429
logger = logging.getLogger(__name__)


class DownloadError(OSError):
    """A dataset download failed; ``status`` holds the HTTP status code when the server answered."""

    def __init__(self, url: str, reason: object, status: int | None = None) -> None:
        """Describe the failing URL and its cause."""
        super().__init__(f"Download failed for {url}: {reason}")
        self.url = url
        self.status = status


class _AllowedHostRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-check every redirect target so a redirect cannot leave HTTPS or the allowed hosts."""

    @override
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        validate_dataset_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_AllowedHostRedirectHandler)


def validate_dataset_url(url: str) -> None:
    """Reject URLs that are not HTTPS or not served by a known dataset host."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        msg = f"Dataset URL must use HTTPS and a supported host: {url!r}."
        raise ValueError(msg)


def download_file(url: str, path: Path) -> None:
    """Stream a dataset to ``path`` without buffering it in memory; ``path`` only appears when complete."""
    logger.info("Downloading %s -> %s", url, path)

    def attempt() -> None:
        with atomic_path(path) as temporary, temporary.open("wb") as target:
            _copy_response(url, target, label=f"Downloading {path.name}")

    _with_retries(url, attempt)


def download_bytes(url: str) -> bytes:
    """Download a (small) dataset into memory."""
    logger.info("Downloading %s", url)

    def attempt() -> bytes:
        with io.BytesIO() as target:
            _copy_response(url, target, label=f"Downloading {PurePosixPath(urlsplit(url).path).name or url}")
            return target.getvalue()

    return _with_retries(url, attempt)


def extract_single_member(
    archive_path: Path, suffixes: frozenset[str], target: Path, *, label: str, max_bytes: int
) -> None:
    """Extract the only archive member with one of ``suffixes`` to ``target``.

    ``max_bytes`` bounds the declared uncompressed size, so a malformed or hostile archive cannot fill the
    disk (zipfile never reads past the declared size).
    """
    with ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if Path(member.filename).suffix.lower() in suffixes]
        if len(members) != 1:
            msg = f"Expected exactly one {'/'.join(sorted(suffixes))} file in {archive_path}, found {len(members)}."
            raise ValueError(msg)
        member = members[0]
        if member.file_size > max_bytes:
            msg = f"{member.filename} in {archive_path} declares {member.file_size:,} bytes (limit {max_bytes:,})."
            raise ValueError(msg)
        logger.info("Extracting %s (%s bytes).", member.filename, f"{member.file_size:,}")
        with archive.open(member) as source, atomic_path(target) as temporary, temporary.open("wb") as output:
            copy_with_progress(source, output, label=label, total=member.file_size)


def _urlopen(request: urllib.request.Request, timeout: float) -> http.client.HTTPResponse:
    """Open a request through the redirect-checking opener (the single network seam, patched in tests)."""
    return cast("http.client.HTTPResponse", _OPENER.open(request, timeout=timeout))


def _copy_response(url: str, target: IO[bytes], *, label: str) -> None:
    validate_dataset_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "timezone-experiments/0.1"})  # noqa: S310
    with _urlopen(request, _TIMEOUT_SECONDS) as response:
        length = response.headers.get("Content-Length")
        copy_with_progress(response, target, label=label, total=int(length) if length is not None else None)


def _with_retries[T](url: str, attempt: Callable[[], T]) -> T:
    for delay in (*_RETRY_DELAYS, None):
        try:
            return attempt()
        except (URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
            status = error.code if isinstance(error, HTTPError) else None
            if delay is None or not _is_transient(error):
                raise DownloadError(url, _reason(error), status) from error
            logger.warning("Download of %s failed (%s); retrying in %.0f s.", url, _reason(error), delay)
            time.sleep(delay)
    msg = "unreachable: the last attempt either returns or raises"
    raise AssertionError(msg)


def _is_transient(error: Exception) -> bool:
    if isinstance(error, HTTPError):
        return error.code >= _SERVER_ERROR or error.code == _TOO_MANY_REQUESTS
    return True


def _reason(error: Exception) -> str:
    if isinstance(error, HTTPError):
        return f"HTTP {error.code} {error.reason}"
    if isinstance(error, URLError):
        return str(error.reason)
    return str(error) or type(error).__name__
