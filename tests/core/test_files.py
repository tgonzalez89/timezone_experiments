"""Atomic writes and dataset reading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from timezone_experiments.core.files import atomic_path, read_json_dataset

if TYPE_CHECKING:
    from pathlib import Path


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
