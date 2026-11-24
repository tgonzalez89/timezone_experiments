"""Map intermediate caches and the render manifest."""

from __future__ import annotations

import json
import zlib
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from timezone_experiments.maps import (
    MapOptions,
    export_maps,
)
from timezone_experiments.maps import borders as borders_module
from timezone_experiments.maps import export as export_module
from timezone_experiments.maps.cache import (
    MAP_CACHE_MANIFEST,
    cached_layer,
    cached_map_paths,
    expected_map_paths,
    read_feature_cache,
    write_feature_cache,
    write_map_cache,
)
from timezone_experiments.maps.median_noon import MEDIAN_NOON_LAYER, MedianNoonMap
from timezone_experiments.maps.solar_events import SOLAR_EVENT_LAYER, SolarEventMap

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from timezone_experiments.maps.spec import MapJob


def test_map_feature_cache_reuses_borders_and_values_for_similar_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "TST:1",
                "properties": {
                    "territory_id": "TST:1",
                    "country_code": "TST",
                    "proposed_utc_offset_hours": 1,
                    "solar_year": 2025,
                    "compliance_rate": 1,
                    "noon": {"median_equation_of_time_minutes": -0.5},
                },
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            }
        ],
    }
    calls: dict[str, int] = {"borders": 0, "median_noon": 0, "solar_events": 0}
    monkeypatch.setattr(
        borders_module,
        "country_borders",
        lambda *_args, **_kwargs: calls.__setitem__("borders", calls["borders"] + 1) or [],
    )

    def builder(kind: str) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
        def build(_collection: dict[str, Any]) -> list[dict[str, Any]]:
            calls[kind] += 1
            return [
                {
                    "type": "Feature",
                    "id": kind,
                    "properties": {"value": calls[kind]},
                    "geometry": collection["features"][0]["geometry"],
                }
            ]

        return build

    median_layer = replace(MEDIAN_NOON_LAYER, build=builder("median_noon"))
    event_layer = replace(SOLAR_EVENT_LAYER, build=builder("solar_events"))
    monkeypatch.setattr(MedianNoonMap, "layer", median_layer)
    monkeypatch.setattr(SolarEventMap, "layer", event_layer)
    monkeypatch.setattr(
        export_module,
        "_map_results",
        lambda _collection, output, requested, _workers: iter(
            (index, [output / f"{field}.{kind}"]) for index, (field, kind) in enumerate(requested)
        ),
    )
    cache = tmp_path / "cache"

    def with_options(source: dict[str, Any], *, cell: float = 0.5, simplify: float = 0.01) -> dict[str, Any]:
        options = MapOptions(
            cache_directory=cache, event_map_cell_degrees=cell, map_geometry_simplification_degrees=simplify
        )
        return {**source, "options": options}

    export_maps(collection, tmp_path / "first", MapOptions(cache_directory=cache), kinds=("html",))
    similar = deepcopy(collection)
    similar["features"][0]["properties"]["compliance_rate"] = 0.75
    export_maps(similar, tmp_path / "second", MapOptions(cache_directory=cache), kinds=("html",))
    # Country and territory outlines once each; display-only changes reuse every layer.
    assert calls == {"borders": 2, "median_noon": 1, "solar_events": 1}

    first = cached_layer(with_options(collection), median_layer)
    changed_display_only = deepcopy(collection)
    changed_display_only["features"][0]["properties"]["compliance_rate"] = 0.5
    assert cached_layer(with_options(changed_display_only), median_layer) == first
    assert calls["median_noon"] == 1
    median_cache = next((cache / "map-features" / "median_noon").glob("*.json.zlib"))
    median_cache.write_bytes(b"corrupt")
    cached_layer(with_options(collection), median_layer)
    assert calls["median_noon"] == 2
    assert cached_layer(with_options(collection), event_layer)
    cached_layer(with_options(collection, cell=1.0), event_layer)
    assert calls["solar_events"] == 2
    cached_layer(with_options(collection, simplify=0.02), event_layer)
    assert calls["solar_events"] == 3
    changed_year = deepcopy(collection)
    changed_year["features"][0]["properties"]["solar_year"] = 2026
    cached_layer(with_options(changed_year), event_layer)
    assert calls["solar_events"] == 4


def test_map_feature_cache_write_failures_only_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    write_feature_cache((blocker / "features.json.zlib", "key"), {"features": []})
    assert "Could not save map feature cache" in caplog.text


def test_map_manifest_that_is_not_an_object_is_ignored(tmp_path: Path) -> None:
    (tmp_path / MAP_CACHE_MANIFEST).write_text("[]", encoding="utf-8")
    assert cached_map_paths(tmp_path, [], "key") is None


def _envelope(**fields: object) -> bytes:
    return zlib.compress(json.dumps(fields).encode())


@pytest.mark.parametrize(
    "content",
    [
        b"not compressed",
        zlib.compress(b"not json"),
        zlib.compress(b"[1, 2]"),
        _envelope(schema=0, key="KEY", data=[]),
        _envelope(schema=2, key="OTHER", data=[]),
        _envelope(schema=2, key="KEY"),
    ],
    ids=["not-zlib", "not-json", "not-an-object", "old-schema", "other-key", "no-data"],
)
def test_corrupt_or_foreign_map_feature_caches_are_ignored(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "entry.json.zlib"
    write_feature_cache((path, "KEY"), {"cached": True})
    assert read_feature_cache((path, "KEY")) == {"cached": True}
    path.write_bytes(content)
    assert read_feature_cache((path, "KEY")) is None


@pytest.mark.parametrize("tamper", ["missing-file", "resized-file", "edited-file", "other-key", "other-jobs"])
def test_render_manifest_is_reused_only_for_intact_matching_outputs(tmp_path: Path, tamper: str) -> None:
    jobs: list[MapJob] = [("timezone_shift_hours", "html")]
    (output,) = expected_map_paths(tmp_path, jobs)
    output.write_text("<html>map</html>", encoding="utf-8")
    write_map_cache(tmp_path, "KEY", [output])
    assert cached_map_paths(tmp_path, jobs, "KEY") == [output]
    key = "KEY"
    if tamper == "missing-file":
        output.unlink()
    elif tamper == "resized-file":
        output.write_text("<html>other map</html>", encoding="utf-8")
    elif tamper == "edited-file":
        output.write_text("<html>MAP</html>", encoding="utf-8")
    elif tamper == "other-key":
        key = "OTHER"
    else:
        jobs = [("timezone_shift_hours", "static")]
    assert cached_map_paths(tmp_path, jobs, key) is None
