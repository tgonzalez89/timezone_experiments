"""Versioned disk caches for prepared territory samples and overlays."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import sys
import zlib
from array import array
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from timezone_experiments.data.models import Territory, TerritorySample
from timezone_experiments.data.paths import cache_directory_for

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from timezone_experiments.config import ExperimentConfig

logger = logging.getLogger(__name__)
_CACHE_SCHEMA = 1
_BASE_SAMPLE_COLUMNS = 4
_POPULATION_SAMPLE_COLUMNS = 3


def content_digest(path: Path) -> str:
    """Hash a cache input without loading its full contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_cache_path(
    directory: Path,
    boundaries: Path,
    config: ExperimentConfig,
    source_admin_level: int | None,
) -> Path:
    """Return the content-addressed path for base territory samples."""
    payload = {
        "schema": _CACHE_SCHEMA,
        "boundaries": content_digest(boundaries),
        "default_admin_level": config.default_admin_level,
        "subdivision_rules": config.subdivision_rules,
        "country_boundary_levels": config.country_boundary_levels,
        "source_admin_level": source_admin_level,
        "sample_grid_size": config.sample_grid_size,
        "byteorder": sys.byteorder,
        "sources": _source_hashes("boundaries.py", "geometry.py", "models.py"),
    }
    return _cache_path(directory, "base", payload)


def timezone_cache_path(directory: Path, base_path: Path, timezones: Path) -> Path:
    """Return the content-addressed path for representative timezone labels."""
    payload = {
        "schema": _CACHE_SCHEMA,
        "base": base_path.stem,
        "timezones": content_digest(timezones),
        "sources": _source_hashes("timezone_index.py", "geometry.py"),
    }
    return _cache_path(directory, "timezones", payload)


def population_cache_path(directory: Path, base_path: Path, population: Path) -> Path:
    """Return the content-addressed path for matched population additions."""
    payload = {
        "schema": _CACHE_SCHEMA,
        "base": base_path.stem,
        "population": content_digest(population),
        "sources": _source_hashes("population.py", "geometry.py"),
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
    _prepare_parent(path)
    temporary = path.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with _connection(temporary) as connection:
            _create_metadata(connection, "base")
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
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
    _prepare_parent(path)
    temporary = path.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with _connection(temporary) as connection:
            _create_metadata(connection, "timezones")
            connection.execute("CREATE TABLE territories (territory_id TEXT PRIMARY KEY, current_timezone TEXT)")
            connection.executemany(
                "INSERT INTO territories VALUES (?,?)",
                ((territory.territory_id, territory.current_timezone) for territory in territories),
            )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
                samples=tuple(
                    TerritorySample(
                        sample.latitude,
                        sample.longitude,
                        sample.area_weight,
                        population_weight=0,
                        longitude_weight=sample.longitude_weight,
                    )
                    for sample in territory.samples
                )
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
    _prepare_parent(path)
    temporary = path.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with _connection(temporary) as connection:
            _create_metadata(connection, "population")
            connection.execute("CREATE TABLE territories (territory_id TEXT PRIMARY KEY, samples BLOB NOT NULL)")
            connection.executemany("INSERT INTO territories VALUES (?,?)", rows)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _source_hashes(*names: str) -> dict[str, str]:
    directory = Path(__file__).parent
    return {name: content_digest(directory / name) for name in names}


def _cache_path(directory: Path, layer: str, payload: dict[str, Any]) -> Path:
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return cache_directory_for(directory) / "prepared" / layer / f"{key}.sqlite3"


def _prepare_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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
    values = array(
        "d",
        (
            value
            for sample in samples
            for value in (sample.latitude, sample.longitude, sample.area_weight, sample.longitude_weight)
        ),
    )
    return zlib.compress(values.tobytes(), level=3)


def _decode_base_samples(data: bytes) -> tuple[TerritorySample, ...]:
    values = _decode_array(data, _BASE_SAMPLE_COLUMNS)
    return tuple(
        TerritorySample(values[index], values[index + 1], values[index + 2], longitude_weight=values[index + 3])
        for index in range(0, len(values), _BASE_SAMPLE_COLUMNS)
    )


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
