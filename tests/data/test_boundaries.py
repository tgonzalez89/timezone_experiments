"""Boundary loading: territories, subdivisions, degenerate geometry, and parallel sampling."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from factories import (
    TOY_BOUNDARIES,
    box_ring,
)

from timezone_experiments.core.config import (
    ExperimentConfig,
)
from timezone_experiments.data import boundaries as boundaries_module
from timezone_experiments.data import boundaries as boundary_module
from timezone_experiments.data.boundaries import (
    load_territories,
    load_territory_geometries,
    read_feature_collection,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_boundary_loader_preserves_valid_parts_of_degenerate_multipolygon(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    collection = json.loads(TOY_BOUNDARIES.read_text(encoding="utf-8"))
    valid = collection["features"][0]["geometry"]["coordinates"]
    collapsed = [[[121.203, -6.7779], [121.2032, -6.7766], [121.203, -6.7779], [121.203, -6.7779]]]
    collection["features"] = collection["features"][:1]
    feature = collection["features"][0]
    feature["geometry"] = {"type": "MultiPolygon", "coordinates": [collapsed, valid]}
    path = tmp_path / "boundaries.geojson"
    path.write_text(json.dumps(collection), encoding="utf-8")
    config = ExperimentConfig(sample_grid_size=2)
    territories = load_territories(path, config)
    assert len(territories) == 1
    assert all(0 <= sample.longitude <= 8 for sample in territories[0].samples)
    assert "zero area" in caplog.text
    feature["geometry"]["coordinates"] = [collapsed]
    path.write_text(json.dumps(collection), encoding="utf-8")
    with pytest.raises(ValueError, match=r"TST.*no non-zero-area"):
        load_territories(path, config)


def test_boundary_loader_builds_country_and_mixed_subdivision_units() -> None:
    country = load_territories(TOY_BOUNDARIES, ExperimentConfig(sample_grid_size=3))
    subdivided = load_territories(
        TOY_BOUNDARIES,
        ExperimentConfig(subdivision_rules={"TST": 1}, sample_grid_size=3),
    )
    assert len(country) == 1
    assert country[0].name == "Testland"
    assert country[0].geometry is not None
    assert len(subdivided) == 2
    assert {territory.name for territory in subdivided} == {"West Province", "East Province"}
    with pytest.raises(ValueError, match="only contains level"):
        load_territories(
            TOY_BOUNDARIES,
            ExperimentConfig(subdivision_rules={"TST": 2}),
        )


def test_load_territories_rejects_non_feature_collections(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text('{"solar_metric": "min"}', encoding="utf-8")
    with pytest.raises(ValueError, match="FeatureCollection"):
        load_territories(settings, ExperimentConfig())


@pytest.mark.parametrize("admin_level", [0, 1])
def test_geometry_only_loading_preserves_grouping_without_sampling(
    admin_level: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig(subdivision_rules={"TST": admin_level}, sample_grid_size=2)
    expected = {territory.territory_id: territory.geometry for territory in load_territories(TOY_BOUNDARIES, config)}

    def no_sampling(*_args: object) -> None:
        pytest.fail("Geometry-only loading must not sample territories.")

    monkeypatch.setattr(boundaries_module, "sample_geometry", no_sampling)
    assert load_territory_geometries(TOY_BOUNDARIES, config) == expected


def test_parallel_sampling_matches_sequential_and_preserves_error_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(boundary_module, "MIN_PARALLEL_SAMPLING_WORK", 0)
    collapsed = [[1, 1], [1, 2], [1, 1], [1, 1]]
    features = [
        {
            "type": "Feature",
            "properties": {"GID_0": "TST", "GID_1": f"TST.{number}_1", "NAME_1": f"Province {number}"},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        box_ring(number * 5, 0, number * 5 + 2, 2),
                        box_ring(number * 5 + 0.5, 0.5, number * 5 + 1.5, 1.5),
                    ],
                    [collapsed],
                ],
            },
        }
        for number in (1, 2)
    ]
    path = tmp_path / "boundaries.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    config = {"subdivision_rules": {"TST": 1}, "sample_grid_size": 4}
    sequential = load_territories(path, ExperimentConfig.from_mapping({**config, "workers": 1}))
    parallel = load_territories(path, ExperimentConfig.from_mapping({**config, "workers": 2}))
    assert parallel == sequential
    features[1]["geometry"] = {"type": "MultiPolygon", "coordinates": [[collapsed]]}
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"TST:TST.2_1.*non-zero-area"):
        load_territories(path, ExperimentConfig.from_mapping({**config, "workers": 2}))


@pytest.mark.parametrize("payload", [[], {"type": "Feature"}, {"type": "FeatureCollection", "features": {}}])
def test_malformed_feature_collections_are_rejected_with_the_path(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "boundaries.geojson"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"boundaries\.geojson"):
        read_feature_collection(path)
