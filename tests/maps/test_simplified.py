"""Joint simplification of the territory geometry shared by every map."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shapely import box
from shapely.geometry import shape

from timezone_experiments.maps import geometry as geometry_module
from timezone_experiments.maps.cache import geometry_digests
from timezone_experiments.maps.geometry import SIMPLIFIED, wrapped_geometry
from timezone_experiments.maps.simplified import simplified_collection
from timezone_experiments.maps.spec import MapOptions

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _collection(options: MapOptions) -> dict[str, Any]:
    border = [(1 + (0.001 if step % 2 else -0.001), step / 10) for step in range(11)]
    rings = [[(0, 0), *border, (0, 1), (0, 0)], [*border, (2, 1), (2, 0), border[0]]]
    features = [
        {
            "id": f"T{index}",
            "geometry": {"type": "Polygon", "coordinates": [[list(point) for point in ring]]},
            "properties": {"proposed_utc_offset_hours": 0},
        }
        for index, ring in enumerate(rings)
    ]
    return {"features": features, "options": options}


def test_wrapped_geometry_cuts_at_the_dateline_and_keeps_the_source_type() -> None:
    crossing = wrapped_geometry(box(170, 0, 190, 10), "Polygon")
    assert crossing["type"] == "MultiPolygon"
    west, east = sorted(
        (shape({"type": "Polygon", "coordinates": part}) for part in crossing["coordinates"]), key=lambda p: p.bounds[0]
    )
    assert west.bounds == (-180, 0, -170, 10)
    assert east.bounds == (170, 0, 180, 10)
    assert wrapped_geometry(box(0, 0, 1, 1), "Polygon")["type"] == "Polygon"
    assert wrapped_geometry(box(0, 0, 1, 1))["type"] == "MultiPolygon"


def test_zero_tolerance_keeps_the_geometry_and_marks_the_collection() -> None:
    collection = _collection(MapOptions(map_geometry_simplification_degrees=0))
    digests = geometry_digests(collection["features"])
    result, result_digests = simplified_collection(collection, digests)
    assert result[SIMPLIFIED] is True
    assert result["features"] == collection["features"]
    assert result_digests == digests


def test_simplified_geometry_is_computed_once_per_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options = MapOptions(map_geometry_simplification_degrees=0.02, cache_directory=tmp_path)
    collection = _collection(options)
    digests = geometry_digests(collection["features"])
    calls: list[int] = []
    original = geometry_module.gradient_geometries

    def counted(*args: Any) -> Any:  # noqa: ANN401
        calls.append(1)
        return original(*args)

    monkeypatch.setattr("timezone_experiments.maps.simplified.gradient_geometries", counted)
    first, first_digests = simplified_collection(collection, digests)
    second, second_digests = simplified_collection(collection, digests)
    assert len(calls) == 1
    assert first["features"] == second["features"]
    assert first_digests == second_digests
    count = lambda feature: len(feature["geometry"]["coordinates"][0])  # noqa: E731
    assert sum(map(count, first["features"])) < sum(map(count, collection["features"]))
    assert first["features"][0]["id"] == "T0"
    # A different tolerance is a different simplification.
    _, other_digests = simplified_collection(
        _collection(MapOptions(map_geometry_simplification_degrees=0.005, cache_directory=tmp_path)), digests
    )
    assert other_digests != first_digests


def test_internal_borders_drop_the_outlines_the_country_layer_already_draws() -> None:
    from timezone_experiments.maps.borders import internal_borders  # noqa: PLC0415

    # Two territories side by side: each outlines itself, and the country outlines both together.
    west = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0))
    east = ((1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 0.0))
    country = ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0))
    (line,) = internal_borders([west, east], [country])
    assert sorted(line) == [(1.0, 0.0), (1.0, 1.0)]  # only the shared edge remains
    assert internal_borders([west], []) == [west]
    assert internal_borders([west], [west]) == []
