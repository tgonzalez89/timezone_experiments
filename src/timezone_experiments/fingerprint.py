"""Stable digests used to key on-disk caches by their inputs, code, and library versions."""

from __future__ import annotations

import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_PACKAGE_ROOT = Path(__file__).parent


def json_digest(value: object) -> str:
    """Return the SHA-256 of a canonical (sorted, compact) JSON encoding of ``value``."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file, reading it in chunks so large inputs are not loaded into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(sources: Iterable[str], packages: Iterable[str] = ()) -> dict[str, object]:
    """Describe the code a cached result depends on.

    ``sources`` are module paths relative to the package root (e.g. ``"data/geometry.py"``); ``packages`` are
    installed distributions whose versions matter. Changing any of them changes the fingerprint.
    """
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = ""
    return {
        "sources": {source: file_digest(_PACKAGE_ROOT / source) for source in sources},
        "packages": versions,
        "python": list(sys.version_info[:3]),
        "byteorder": sys.byteorder,
    }
