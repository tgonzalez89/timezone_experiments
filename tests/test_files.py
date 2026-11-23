"""Verify atomic writes, best-effort caches, and corrupt-dataset diagnostics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from timezone_experiments.data.boundaries import read_feature_collection
from timezone_experiments.data.models import Territory, TerritorySample
from timezone_experiments.data.prepared_cache import read_base_territories, write_base_territories
from timezone_experiments.files import atomic_path, read_json_dataset
from timezone_experiments.visualization import maps
from timezone_experiments.visualization.map_cache import write_feature_cache

if TYPE_CHECKING:
    from pathlib import Path


def _territory() -> Territory:
    return Territory("TST:1", "TST", "Test", 0, (TerritorySample(0, 0),))


def test_atomic_path_keeps_previous_content_when_the_write_fails(tmp_path: Path) -> None:
    target = tmp_path / "data.txt"
    target.write_text("previous", encoding="utf-8")

    def interrupted_write() -> None:
        with atomic_path(target) as temporary:
            temporary.write_text("half", encoding="utf-8")
            msg = "interrupted"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="interrupted"):
        interrupted_write()
    assert target.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.iterdir()) == [target]


def test_concurrent_atomic_writers_use_distinct_temporary_files(tmp_path: Path) -> None:
    target = tmp_path / "cache" / "data.sqlite3"
    with atomic_path(target) as first, atomic_path(target) as second:
        assert first != second
        first.write_text("first", encoding="utf-8")
        second.write_text("second", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "first"
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("content", ['{"type": "FeatureCollection", "features": [', "\xff\xfe garbage"])
def test_corrupt_datasets_name_the_file_and_the_fix(tmp_path: Path, content: str) -> None:
    path = tmp_path / "world_countries.geojson"
    path.write_bytes(content.encode("latin-1"))
    with pytest.raises(ValueError, match=r"Corrupt or unreadable dataset .*world_countries\.geojson.*--refresh"):
        read_json_dataset(path)


@pytest.mark.parametrize("payload", [[], {"type": "Feature"}, {"type": "FeatureCollection", "features": {}}])
def test_malformed_feature_collections_are_rejected_with_the_path(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "boundaries.geojson"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"boundaries\.geojson"):
        read_feature_collection(path)


def test_prepared_cache_write_failures_only_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    path = blocker / "base.sqlite3"
    write_base_territories(path, [_territory()])
    assert "Could not save prepared base cache" in caplog.text
    assert read_base_territories(path) is None


def test_prepared_cache_can_be_rewritten_in_place(tmp_path: Path) -> None:
    path = tmp_path / "base.sqlite3"
    write_base_territories(path, [_territory()])
    write_base_territories(path, [_territory(), Territory("TST:2", "TST", "Other", 0, (TerritorySample(1, 1),))])
    cached = read_base_territories(path)
    assert cached is not None
    assert [territory.territory_id for territory in cached] == ["TST:1", "TST:2"]
    assert list(tmp_path.iterdir()) == [path]


def test_map_feature_cache_write_failures_only_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    write_feature_cache((blocker / "features.json.zlib", "key"), {"features": []})
    assert "Could not save map feature cache" in caplog.text


def test_map_manifest_that_is_not_an_object_is_ignored(tmp_path: Path) -> None:
    (tmp_path / maps._MAP_CACHE_MANIFEST).write_text("[]", encoding="utf-8")
    assert maps._cached_map_paths(tmp_path, [], "key") is None
