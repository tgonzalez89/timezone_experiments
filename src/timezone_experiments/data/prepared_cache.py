"""Versioned disk caches for prepared territory samples and overlays."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import zlib
from array import array
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from timezone_experiments.core.files import atomic_path
from timezone_experiments.core.fingerprint import file_digest, json_digest, source_fingerprint
from timezone_experiments.core.models import (
    BASE_SAMPLE_COLUMNS,
    Territory,
    TerritorySample,
    pack_base_samples,
    unpack_base_samples,
)
from timezone_experiments.data.paths import cache_directory

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence
    from pathlib import Path

    from timezone_experiments.core.config import ExperimentConfig

logger = logging.getLogger(__name__)
_CACHE_SCHEMA = 2
_POPULATION_SAMPLE_COLUMNS = 3


def base_cache_path(directory: Path, boundaries: Path, config: ExperimentConfig) -> Path:
    """Return the content-addressed path for base territory samples."""
    payload = {
        "schema": _CACHE_SCHEMA,
        "boundaries": file_digest(boundaries),
        "default_admin_level": config.default_admin_level,
        "subdivision_rules": config.subdivision_rules,
        "merged_subdivision_levels": config.merged_subdivision_levels,
        "sample_grid_size": config.sample_grid_size,
        "code": source_fingerprint(["data/boundaries.py", "core/geometry.py", "core/models.py", "core/constants.py"]),
    }
    return _cache_path(directory, "base", payload)


def timezone_cache_path(directory: Path, base_path: Path, timezones: Path) -> Path:
    """Return the content-addressed path for representative timezone labels."""
    payload = {
        "schema": _CACHE_SCHEMA,
        "base": base_path.stem,
        "timezones": file_digest(timezones),
        "code": source_fingerprint(
            ["data/timezone_index.py", "data/boundaries.py", "core/geometry.py", "core/constants.py"]
        ),
    }
    return _cache_path(directory, "timezones", payload)


def population_cache_path(directory: Path, base_path: Path, population: Path) -> Path:
    """Return the content-addressed path for matched population additions."""
    payload = {
        "schema": _CACHE_SCHEMA,
        "base": base_path.stem,
        "population": file_digest(population),
        "code": source_fingerprint(["data/population.py", "core/geometry.py", "core/models.py", "core/constants.py"]),
    }
    return _cache_path(directory, "population", payload)


def read_base_territories(path: Path) -> list[Territory] | None:
    """Load base geometry and numeric samples from a verified SQLite cache."""
    if not path.is_file():
        return None
    try:
        with _connection(path) as connection:
            _validate_metadata(connection, "base")
            rows = connection.execute(
                "SELECT territory_id,country_code,name,admin_level,current_timezone,geometry,samples "
                "FROM territories ORDER BY position"
            )
            territories = [
                Territory(
                    territory_id=territory_id,
                    country_code=country,
                    name=name,
                    admin_level=admin_level,
                    current_timezone=timezone,
                    geometry=None if geometry is None else json.loads(zlib.decompress(geometry)),
                    samples=_decode_base_samples(samples),
                )
                for territory_id, country, name, admin_level, timezone, geometry, samples in rows
            ]
            if not territories:
                logger.warning("Ignoring empty prepared territory cache %s.", path)
                return None
            return territories
    except (FileNotFoundError, OSError, sqlite3.DatabaseError, ValueError, TypeError, zlib.error) as error:
        logger.warning("Ignoring unusable prepared territory cache %s: %s", path, error)
        return None


def write_base_territories(path: Path, territories: Sequence[Territory]) -> None:
    """Atomically persist sampled territory geometry and base weights."""

    def populate(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE territories (position INTEGER PRIMARY KEY, territory_id TEXT NOT NULL, "
            "country_code TEXT NOT NULL, name TEXT NOT NULL, admin_level INTEGER NOT NULL, "
            "current_timezone TEXT, geometry BLOB, samples BLOB NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO territories VALUES (?,?,?,?,?,?,?,?)",
            (
                (
                    position,
                    territory.territory_id,
                    territory.country_code,
                    territory.name,
                    territory.admin_level,
                    territory.current_timezone,
                    _encode_geometry(territory.geometry),
                    _encode_base_samples(territory.samples),
                )
                for position, territory in enumerate(territories)
            ),
        )

    _write_cache(path, "base", populate)


def read_timezone_layer(path: Path, territories: Sequence[Territory]) -> list[Territory] | None:
    """Apply cached representative timezone labels to base territories."""
    if not path.is_file():
        return None
    try:
        with _connection(path) as connection:
            _validate_metadata(connection, "timezones")
            zones = dict(connection.execute("SELECT territory_id,current_timezone FROM territories"))
        if set(zones) != {territory.territory_id for territory in territories}:
            logger.warning("Ignoring timezone overlay cache with mismatched territories %s.", path)
            return None
        return [replace(territory, current_timezone=zones[territory.territory_id]) for territory in territories]
    except (FileNotFoundError, OSError, sqlite3.DatabaseError, ValueError, TypeError) as error:
        logger.warning("Ignoring unusable timezone overlay cache %s: %s", path, error)
        return None


def write_timezone_layer(path: Path, territories: Sequence[Territory]) -> None:
    """Atomically persist representative timezone labels only."""

    def populate(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE territories (territory_id TEXT PRIMARY KEY, current_timezone TEXT)")
        connection.executemany(
            "INSERT INTO territories VALUES (?,?)",
            ((territory.territory_id, territory.current_timezone) for territory in territories),
        )

    _write_cache(path, "timezones", populate)


def read_population_layer(path: Path, territories: Sequence[Territory]) -> list[Territory] | None:
    """Apply cached population-cell additions to base territory samples."""
    if not path.is_file():
        return None
    try:
        with _connection(path) as connection:
            _validate_metadata(connection, "population")
            additions = {
                territory_id: _decode_population_samples(samples)
                for territory_id, samples in connection.execute("SELECT territory_id,samples FROM territories")
            }
        if set(additions) != {territory.territory_id for territory in territories}:
            logger.warning("Ignoring population overlay cache with mismatched territories %s.", path)
            return None
        return [
            replace(
                territory,
                samples=tuple(sample.with_zero_population() for sample in territory.samples)
                + additions[territory.territory_id],
            )
            for territory in territories
        ]
    except (FileNotFoundError, OSError, sqlite3.DatabaseError, ValueError, TypeError, zlib.error) as error:
        logger.warning("Ignoring unusable population overlay cache %s: %s", path, error)
        return None


def write_population_layer(
    path: Path, base_territories: Sequence[Territory], weighted_territories: Sequence[Territory]
) -> None:
    """Persist only population points added after the base sample set."""
    if [item.territory_id for item in base_territories] != [item.territory_id for item in weighted_territories]:
        msg = "Population cache territories do not match base territories."
        raise ValueError(msg)
    rows = []
    for base, weighted in zip(base_territories, weighted_territories, strict=True):
        count = len(base.samples)
        if weighted.samples[:count] and any(
            (actual.latitude, actual.longitude, actual.area_weight, actual.longitude_weight)
            != (expected.latitude, expected.longitude, expected.area_weight, expected.longitude_weight)
            for actual, expected in zip(weighted.samples[:count], base.samples, strict=True)
        ):
            msg = "Population overlay changed base samples."
            raise ValueError(msg)
        rows.append((base.territory_id, _encode_population_samples(weighted.samples[count:])))

    def populate(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE territories (territory_id TEXT PRIMARY KEY, samples BLOB NOT NULL)")
        connection.executemany("INSERT INTO territories VALUES (?,?)", rows)

    _write_cache(path, "population", populate)


def _cache_path(directory: Path, layer: str, payload: dict[str, Any]) -> Path:
    return cache_directory(directory) / "prepared" / layer / f"{json_digest(payload)}.sqlite3"


def _write_cache(path: Path, kind: str, populate: Callable[[sqlite3.Connection], None]) -> None:
    """Build a cache file atomically; a cache is optional, so failing to save one only logs a warning."""
    try:
        with atomic_path(path) as temporary, _connection(temporary) as connection:
            _create_metadata(connection, kind)
            populate(connection)
    except (OSError, sqlite3.Error) as error:
        logger.warning("Could not save prepared %s cache %s: %s", kind, path, error)
        return
    logger.info("Saved prepared %s cache: %s", kind, path)


@contextmanager
def _connection(path: Path) -> Generator[sqlite3.Connection]:
    connection = sqlite3.connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _create_metadata(connection: sqlite3.Connection, kind: str) -> None:
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO metadata VALUES (?,?)",
        (("schema", str(_CACHE_SCHEMA)), ("kind", kind), ("byteorder", sys.byteorder)),
    )


def _validate_metadata(connection: sqlite3.Connection, kind: str) -> None:
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    if metadata != {"schema": str(_CACHE_SCHEMA), "kind": kind, "byteorder": sys.byteorder}:
        msg = "cache metadata does not match this runtime and cache schema"
        raise ValueError(msg)


def _encode_geometry(geometry: dict[str, Any] | None) -> bytes | None:
    if geometry is None:
        return None
    return zlib.compress(json.dumps(geometry, separators=(",", ":")).encode(), level=3)


def _encode_base_samples(samples: Sequence[TerritorySample]) -> bytes:
    return zlib.compress(pack_base_samples(samples).tobytes(), level=3)


def _decode_base_samples(data: bytes) -> tuple[TerritorySample, ...]:
    return unpack_base_samples(_decode_array(data, BASE_SAMPLE_COLUMNS))


def _encode_population_samples(samples: Sequence[TerritorySample]) -> bytes:
    values = array(
        "d",
        (value for sample in samples for value in (sample.latitude, sample.longitude, sample.population_weight or 0)),
    )
    return zlib.compress(values.tobytes(), level=3)


def _decode_population_samples(data: bytes) -> tuple[TerritorySample, ...]:
    values = _decode_array(data, _POPULATION_SAMPLE_COLUMNS)
    return tuple(
        TerritorySample(
            values[index],
            values[index + 1],
            area_weight=0,
            population_weight=values[index + 2],
            longitude_weight=0,
        )
        for index in range(0, len(values), _POPULATION_SAMPLE_COLUMNS)
    )


def _decode_array(data: bytes, columns: int) -> array[float]:
    raw = zlib.decompress(data)
    if len(raw) % (columns * 8):
        msg = "prepared cache sample payload has an invalid length"
        raise ValueError(msg)
    values = array("d")
    values.frombytes(raw)
    return values
