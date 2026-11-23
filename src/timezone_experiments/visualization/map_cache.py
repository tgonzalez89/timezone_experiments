"""Versioned compressed disk caches for reusable map intermediates."""

from __future__ import annotations

import hashlib
import json
import logging
import zlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from timezone_experiments.files import write_bytes_atomic

logger = logging.getLogger(__name__)
_FEATURE_CACHE_SCHEMA = 1


def cache_feature_data(
    cache_directory: Path | None,
    kind: str,
    dependency: object,
) -> tuple[Path, str] | None:
    """Return the cache path and key for a spatial map dependency."""
    if cache_directory is None:
        return None
    source_directory = Path(__file__).parents[1]
    sources = {}
    for relative in (
        "visualization/maps.py",
        "astronomy/events.py",
        "astronomy/solar.py",
        "data/geometry.py",
    ):
        source = source_directory / relative
        sources[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    dependencies = {}
    for package in ("numpy", "plotly", "shapely"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = ""
    payload = {
        "schema": _FEATURE_CACHE_SCHEMA,
        "kind": kind,
        "dependency": dependency,
        "sources": sources,
        "dependencies": dependencies,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    key = hashlib.sha256(encoded).hexdigest()
    return cache_directory / "map-features" / kind / f"{key}.json.zlib", key


def read_feature_cache(cache: tuple[Path, str] | None) -> object | None:
    """Read a compressed cache entry, returning None on any integrity failure."""
    if cache is None:
        return None
    path, expected_key = cache
    if not path.is_file():
        return None
    try:
        envelope = json.loads(zlib.decompress(path.read_bytes()))
    except (OSError, UnicodeError, zlib.error, json.JSONDecodeError) as error:
        logger.warning("Ignoring unusable map feature cache %s: %s", path, error)
        return None
    if not isinstance(envelope, dict) or envelope.get("schema") != _FEATURE_CACHE_SCHEMA:
        return None
    if envelope.get("key") != expected_key or "data" not in envelope:
        return None
    return envelope["data"]


def write_feature_cache(cache: tuple[Path, str] | None, data: object) -> None:
    """Atomically write one compressed map-intermediate cache entry."""
    if cache is None:
        return
    path, key = cache
    encoded = json.dumps(
        {"schema": _FEATURE_CACHE_SCHEMA, "key": key, "data": data},
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    try:
        write_bytes_atomic(path, zlib.compress(encoded, level=3))
    except OSError as error:
        logger.warning("Could not save map feature cache %s: %s", path, error)
