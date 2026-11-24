"""Prepared-territory caches."""

from __future__ import annotations

import sqlite3
import sys
import zlib
from typing import TYPE_CHECKING

import pytest
from factories import (
    make_territory,
)

from timezone_experiments.core import fingerprint
from timezone_experiments.core.config import ExperimentConfig
from timezone_experiments.core.models import Territory, TerritorySample
from timezone_experiments.data import prepared_cache
from timezone_experiments.data.prepared_cache import read_base_territories, write_base_territories

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.parametrize(
    ("cache_path", "changed_source"),
    [
        ("base", "constants.py"),
        ("timezone", "constants.py"),
        ("timezone", "boundaries.py"),
        ("population", "constants.py"),
    ],
)
def test_prepared_cache_keys_track_the_source_files_they_depend_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_path: str, changed_source: str
) -> None:
    data_file = tmp_path / "input.geojson"
    data_file.write_text("{}", encoding="utf-8")

    def key() -> Path:
        if cache_path == "base":
            return prepared_cache.base_cache_path(tmp_path, data_file, ExperimentConfig())
        if cache_path == "timezone":
            return prepared_cache.timezone_cache_path(tmp_path, data_file, data_file)
        return prepared_cache.population_cache_path(tmp_path, data_file, data_file)

    before = key()
    original = fingerprint.file_digest
    monkeypatch.setattr(
        fingerprint, "file_digest", lambda path: "edited" if path.name == changed_source else original(path)
    )
    assert key() != before


def test_prepared_cache_write_failures_only_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    path = blocker / "base.sqlite3"
    write_base_territories(path, [make_territory()])
    assert "Could not save prepared base cache" in caplog.text
    assert read_base_territories(path) is None


def test_prepared_cache_can_be_rewritten_in_place(tmp_path: Path) -> None:
    path = tmp_path / "base.sqlite3"
    write_base_territories(path, [make_territory()])
    write_base_territories(path, [make_territory(), Territory("TST:2", "TST", "Other", 0, (TerritorySample(1, 1),))])
    cached = read_base_territories(path)
    assert cached is not None
    assert [territory.territory_id for territory in cached] == ["TST:1", "TST:2"]
    assert list(tmp_path.iterdir()) == [path]


def _two_territories() -> list[Territory]:
    return [make_territory("TST:1"), make_territory("TST:2", 10)]


def _rewrite_metadata(path: Path, key: str, value: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE metadata SET value=? WHERE key=?", (value, key))
    connection.close()


def _not_sqlite(path: Path) -> None:
    path.write_bytes(b"not a database")


def _timezone_layer_instead(path: Path) -> None:
    path.unlink()
    prepared_cache.write_timezone_layer(path, _two_territories())


def _empty_table(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM territories")
    connection.close()


def _truncated_samples(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE territories SET samples=?", (zlib.compress(b"1234567"),))
    connection.close()


@pytest.mark.parametrize(
    "damage",
    [
        _not_sqlite,
        _timezone_layer_instead,
        lambda path: _rewrite_metadata(path, "schema", "0"),
        lambda path: _rewrite_metadata(path, "byteorder", "big" if sys.byteorder == "little" else "little"),
        _empty_table,
        _truncated_samples,
    ],
    ids=["not-sqlite", "wrong-kind", "wrong-schema", "wrong-byte-order", "empty", "truncated-samples"],
)
def test_unusable_base_caches_are_ignored(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, damage: Callable[[Path], None]
) -> None:
    path = tmp_path / "base.sqlite3"
    write_base_territories(path, _two_territories())
    assert read_base_territories(path) == _two_territories()
    damage(path)
    assert read_base_territories(path) is None
    assert "Ignoring" in caplog.text


def test_overlay_layers_with_mismatched_territories_are_ignored(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    territories = _two_territories()
    timezones = tmp_path / "timezones.sqlite3"
    population = tmp_path / "population.sqlite3"
    prepared_cache.write_timezone_layer(timezones, territories)
    prepared_cache.write_population_layer(population, territories, territories)
    assert prepared_cache.read_timezone_layer(timezones, territories) == territories
    assert prepared_cache.read_population_layer(population, territories) is not None
    others = [*territories[:1], make_territory("TST:3")]
    assert prepared_cache.read_timezone_layer(timezones, others) is None
    assert prepared_cache.read_population_layer(population, others) is None
    assert "timezone overlay cache with mismatched territories" in caplog.text
    assert "population overlay cache with mismatched territories" in caplog.text


def test_a_cache_write_that_fails_midway_leaves_no_file(tmp_path: Path) -> None:
    path = tmp_path / "base.sqlite3"
    unserializable = make_territory("TST:2", geometry={"type": "Polygon", "coordinates": object()})
    with pytest.raises(TypeError):
        write_base_territories(path, [make_territory(), unserializable])
    assert not any(tmp_path.iterdir())
