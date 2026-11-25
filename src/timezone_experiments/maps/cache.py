"""Map caches: compressed intermediates (borders, gradient layers) and the final-render manifest per output folder."""

from __future__ import annotations

import json
import logging
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from timezone_experiments.core.files import write_bytes_atomic, write_text_atomic
from timezone_experiments.core.fingerprint import file_digest, json_digest, source_fingerprint
from timezone_experiments.maps.spec import STATIC_FORMATS, options_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from timezone_experiments.maps.spec import GradientLayer, MapJob

logger = logging.getLogger(__name__)
_FEATURE_CACHE_SCHEMA = 2
_MAP_CACHE_SCHEMA = 2
MAP_CACHE_MANIFEST = ".map-render-cache.json"
# Every module of this package, plus the astronomy and geometry code the map layers are computed with.
_SOURCES = (
    *(f"maps/{path.name}" for path in sorted(Path(__file__).parent.glob("*.py"))),
    "astronomy/events.py",
    "astronomy/solar.py",
    "core/geometry.py",
)


def geometry_digests(features: Sequence[dict[str, Any]]) -> list[str]:
    """Hash each feature's geometry once, so cache keys need not re-serialize the coordinates."""
    return [json_digest(feature["geometry"]) for feature in features]


# --- Intermediate feature caches ----------------------------------------------------------------------------------


def cache_feature_data(cache_directory: Path | None, kind: str, dependency: object) -> tuple[Path, str] | None:
    """Return the cache path and key for a spatial map dependency."""
    if cache_directory is None:
        return None
    key = json_digest(
        {
            "schema": _FEATURE_CACHE_SCHEMA,
            "kind": kind,
            "dependency": dependency,
            "code": source_fingerprint(_SOURCES, ["numpy", "plotly", "shapely"]),
        }
    )
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


def cached_layer(
    collection: dict[str, Any], layer: GradientLayer, digests: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Return a gradient layer's features, rebuilding only when a territory input the layer depends on changed."""
    options = options_of(collection)
    features = collection["features"]
    dependency = [
        {
            "id": feature["id"],
            "geometry": digest,
            "proposed_utc_offset_hours": feature["properties"]["proposed_utc_offset_hours"],
            "geometry_simplification_degrees": options.map_geometry_simplification_degrees,
            **layer.dependency(feature["properties"], options),
        }
        for feature, digest in zip(features, digests or geometry_digests(features), strict=True)
    ]
    cache = cache_feature_data(options.cache_directory, layer.name, dependency)
    cached = read_feature_cache(cache)
    if isinstance(cached, list) and all(
        isinstance(feature, dict) and {"type", "id", "properties", "geometry"} <= feature.keys() for feature in cached
    ):
        logger.info("Using cached %s map features.", layer.name.replace("_", " "))
        return cached
    built = layer.build(collection)
    write_feature_cache(cache, built)
    return built


# --- Final-render manifest ------------------------------------------------------------------------------------------


def map_cache_key(
    collection: dict[str, Any], digests: Sequence[str], jobs: list[MapJob], settings: dict[str, object]
) -> str:
    """Fingerprint everything a batch of rendered maps depends on."""
    return json_digest(
        {
            "schema": _MAP_CACHE_SCHEMA,
            "jobs": jobs,
            "settings": settings,
            "properties": [(feature["id"], feature["properties"]) for feature in collection["features"]],
            "geometry": digests,
            "excluded_areas": json_digest(collection.get("excluded_areas", [])),
            "code": source_fingerprint(_SOURCES, ["geopandas", "matplotlib", "numpy", "plotly", "shapely"]),
        }
    )


def expected_map_paths(output: Path, jobs: list[MapJob]) -> list[Path]:
    """Return the files a batch of jobs writes, in job order (a static job writes PNG and SVG)."""
    return [
        path
        for field, kind in jobs
        for path in (
            tuple(output / f"{field}.{extension}" for extension in STATIC_FORMATS[kind])
            if kind in STATIC_FORMATS
            else (output / f"{field}.html",)
        )
    ]


def cached_map_paths(output: Path, jobs: list[MapJob], cache_key: str) -> list[Path] | None:
    """Return the existing outputs when the manifest matches the key and every file is intact, else None."""
    files = _manifest_files(output / MAP_CACHE_MANIFEST, cache_key)
    expected = expected_map_paths(output, jobs)
    if files is None or [item.get("name") for item in files] != [path.name for path in expected]:
        return None
    return expected if all(_intact(path, item) for item, path in zip(files, expected, strict=True)) else None


def _manifest_files(path: Path, cache_key: str) -> list[dict[str, Any]] | None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        logger.warning("Ignoring unreadable map cache manifest %s: %s", path, error)
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != _MAP_CACHE_SCHEMA
        or manifest.get("key") != cache_key
    ):
        return None
    files = manifest.get("files")
    if not isinstance(files, list) or not all(isinstance(item, dict) for item in files):
        return None
    return files


def _intact(path: Path, item: dict[str, Any]) -> bool:
    try:
        return path.stat().st_size == item.get("size") and file_digest(path) == item.get("sha256")
    except OSError:
        return False


def write_map_cache(output: Path, cache_key: str, paths: list[Path]) -> None:
    """Record the rendered files' sizes and checksums, so an identical rerun can reuse them."""
    try:
        files = [{"name": path.name, "size": path.stat().st_size, "sha256": file_digest(path)} for path in paths]
    except OSError:
        logger.warning("Map outputs were incomplete; not writing cache manifest in %s.", output)
        return
    manifest_path = output / MAP_CACHE_MANIFEST
    manifest = {"schema": _MAP_CACHE_SCHEMA, "key": cache_key, "files": files}
    try:
        write_text_atomic(manifest_path, json.dumps(manifest, indent=2) + "\n")
    except OSError as error:
        logger.warning("Could not save map cache manifest %s: %s", manifest_path, error)
