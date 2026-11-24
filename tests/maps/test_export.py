"""Map export: jobs, parallel workers, and the final-render cache."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest
from factories import (
    TOY_BOUNDARIES,
    html_map_data,
    map_inputs,
)

from timezone_experiments.core.config import (
    ExperimentConfig,
)
from timezone_experiments.data.boundaries import load_territories
from timezone_experiments.experiments.runner import run_experiment
from timezone_experiments.maps import (
    MAP_FIELDS,
    MapKind,
    MapOptions,
    build_plotly_figure,
    export_maps,
    result_feature_collection,
)
from timezone_experiments.maps import export as export_module
from timezone_experiments.maps.cache import MAP_CACHE_MANIFEST

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from timezone_experiments.maps.spec import MapJob


def test_map_exports(tmp_path: Path) -> None:
    territories = load_territories(
        TOY_BOUNDARIES,
        ExperimentConfig(subdivision_rules={"TST": 1}, sample_grid_size=2),
    )
    comparison = run_experiment(
        ExperimentConfig(minimum_noon=None, utc_offsets=(1,)),
        territories,
        on_territory_done=lambda *_: None,
    )
    figure = build_plotly_figure(territories, comparison, MAP_FIELDS[0])
    assert figure.data[0].type == "choropleth"
    assert figure.layout.geo.fitbounds is False
    meridians, boundaries, labels, country_borders = figure.data[1:]
    assert country_borders.name == "Country borders"
    assert country_borders.line.width > max(figure.data[0].marker.line.width)
    assert country_borders.line.color != figure.data[0].marker.line.color
    assert {longitude for longitude in meridians.lon if longitude is not None} == set(range(-180, 181, 15))
    assert {longitude for longitude in boundaries.lon if longitude is not None} == {
        longitude + 7.5 for longitude in range(-180, 180, 15)
    }
    assert meridians.line.dash == "dot"
    assert boundaries.line.dash == "dash"
    assert list(labels.lon) == list(range(-180, 180, 15))
    assert list(labels.text) == ["UTC-12/+12"] + [f"UTC{offset:+d}" for offset in range(-11, 12)]
    assert all(trace.hoverinfo == "skip" and not trace.showlegend for trace in figure.data[1:])
    static_paths = export_maps(
        result_feature_collection(territories, comparison), tmp_path / "static", kinds=("static",)
    )
    assert {path.name for path in static_paths} == {
        f"{field}.{suffix}" for field in MAP_FIELDS for suffix in ("png", "svg")
    }
    export_maps(result_feature_collection(territories, comparison), tmp_path / "interactive", kinds=("html",))
    assert (tmp_path / "interactive" / f"{MAP_FIELDS[0]}.html").exists()
    assert (tmp_path / "interactive" / "plotly.min.js").exists()
    with pytest.raises(ValueError, match="Unsupported map statistic"):
        build_plotly_figure(territories, comparison, "unknown")


def test_final_render_cache_reuses_exact_outputs_and_rejects_changed_or_corrupt_inputs(
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
                    "fallback_used": False,
                },
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            }
        ],
    }
    calls = 0
    # Only output reuse is under test here; layers are covered by the feature-cache test.
    monkeypatch.setattr(export_module, "with_map_layers", lambda collection, *_args: collection)

    def fake_results(
        inputs: dict[str, Any], output: Path, requested: list[MapJob], _workers: int
    ) -> Iterator[tuple[int, list[Path]]]:
        nonlocal calls
        calls += 1
        output.mkdir(parents=True, exist_ok=True)
        for position, (field, kind) in enumerate(requested):
            assert kind == "html"
            path = output / f"{field}.html"
            path.write_bytes(json.dumps(inputs["features"]).encode())
            yield position, [path]

    def export(source: dict[str, Any], workers: int) -> list[Path]:
        return export_maps(source, tmp_path, MapOptions(workers=workers), kinds=("html",))

    monkeypatch.setattr(export_module, "_map_results", fake_results)
    first = export(collection, 2)
    assert calls == 1
    assert (tmp_path / MAP_CACHE_MANIFEST).is_file()
    assert export(collection, 4) == first
    assert calls == 1

    first[0].write_text("corrupt", encoding="utf-8")
    export(collection, 1)
    assert calls == 2

    changed = deepcopy(collection)
    changed["features"][0]["properties"]["proposed_utc_offset_hours"] = 2
    export(changed, 1)
    assert calls == 3


@pytest.mark.slow
@pytest.mark.parametrize("interactive", [False, True])
def test_parallel_map_exports_match_serial_outputs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str], *, interactive: bool
) -> None:
    caplog.set_level(logging.INFO)
    territories, results = map_inputs((-2, 1.25, None))
    kinds: tuple[MapKind, ...] = ("static", "html") if interactive else ("static",)
    jobs = len(MAP_FIELDS) * len(kinds)
    collection = result_feature_collection(territories, results)
    serial = export_maps(collection, tmp_path / "serial", MapOptions(workers=1), kinds=kinds)
    capsys.readouterr()
    parallel = export_maps(collection, tmp_path / "parallel", MapOptions(workers=2), kinds=kinds)
    assert [path.name for path in parallel] == [path.name for path in serial]
    assert len(parallel) == len(MAP_FIELDS) * (3 if interactive else 2)
    for expected, actual in zip(serial, parallel, strict=True):
        assert actual.is_file()
        if actual.suffix == ".png":
            assert actual.read_bytes() == expected.read_bytes()
        elif actual.suffix == ".html":
            assert html_map_data(actual) == html_map_data(expected)
        else:
            assert b"<svg" in actual.read_bytes()
    assert f"Exporting {jobs} map jobs with 2 workers" in caplog.text


@pytest.mark.slow
def test_parallel_map_export_propagates_failures_and_caps_workers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    territories, results = map_inputs((-2, 1.25))
    (tmp_path / "proposed_utc_offset_hours.png").mkdir()
    with pytest.raises(OSError, match=r"proposed_utc_offset_hours\.png"):
        export_maps(result_feature_collection(territories, results), tmp_path, MapOptions(workers=128))
    output = capsys.readouterr().err
    assert "Exporting map jobs [failed]:" in output
    assert "Exporting 14 map jobs with 14 workers" in caplog.text
    with pytest.raises(ValueError, match="Map export workers must be positive"):
        export_maps(result_feature_collection(territories, results), tmp_path, MapOptions(workers=0))
