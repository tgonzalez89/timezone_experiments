"""Map feature collections built from results and saved boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from factories import (
    TOY_BOUNDARIES,
    html_map_data,
    map_inputs,
)

from timezone_experiments.astronomy import solar as solar_module
from timezone_experiments.core.config import (
    ExperimentConfig,
)
from timezone_experiments.data.boundaries import load_territories, load_territory_geometries
from timezone_experiments.experiments.results_io import read_results_csv, write_results_csv
from timezone_experiments.experiments.runner import run_experiment
from timezone_experiments.maps import (
    MAP_FIELDS,
    MapOptions,
    build_plotly_figure,
    export_maps,
    geometry_feature_collection,
    result_feature_collection,
)
from timezone_experiments.optimization import assessment as assessment_module
from timezone_experiments.optimization import optimizer as optimizer_module

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.slow
def test_saved_result_maps_match_experiment_maps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ExperimentConfig(subdivision_rules={"TST": 1}, sample_grid_size=2, workers=1)
    territories = load_territories(TOY_BOUNDARIES, config)
    results = run_experiment(config, territories, on_territory_done=lambda *_: None)
    csv_path = write_results_csv(results, tmp_path / "results.csv")
    expected = export_maps(result_feature_collection(territories, results), tmp_path / "original")

    def no_calculation(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Saved-result maps must use persisted noon statistics, not recompute astronomy.")

    monkeypatch.setattr(solar_module, "equation_of_time_minutes", no_calculation)
    monkeypatch.setattr(assessment_module, "equation_of_time_minutes", no_calculation)
    monkeypatch.setattr(assessment_module, "current_noon_ranges", no_calculation)
    monkeypatch.setattr(assessment_module, "describe_timezone", no_calculation)
    monkeypatch.setattr(optimizer_module, "calculate_noon_statistics", no_calculation)
    # Also exercise rendering in this process, where the guards are installed.
    saved_results = read_results_csv(csv_path)
    saved_figure = build_plotly_figure(territories, saved_results, MAP_FIELDS[0])
    assert str(results[0].noon.year) in saved_figure.data[0].customdata[0][4]
    saved_median_figure = build_plotly_figure(territories, saved_results, "median_solar_noon")
    assert len(set(saved_median_figure.data[0].z)) > 1
    actual = export_maps(
        geometry_feature_collection(load_territory_geometries(TOY_BOUNDARIES, config), saved_results),
        tmp_path / "saved",
        MapOptions(workers=2),
    )
    assert [path.name for path in actual] == [path.name for path in expected]
    for original, saved in zip(expected, actual, strict=True):
        if saved.suffix == ".png":
            assert saved.read_bytes() == original.read_bytes()
        elif saved.suffix == ".html":
            assert html_map_data(saved) == html_map_data(original)
        else:
            assert b"<svg" in saved.read_bytes()


def test_saved_result_maps_reject_unmatched_territories(tmp_path: Path) -> None:
    territories, results = map_inputs((1,))
    geometries = {
        territory.territory_id: territory.geometry for territory in territories if territory.geometry is not None
    }
    with pytest.raises(ValueError, match=r"No boundary geometry matches.*unmapped"):
        export_maps(geometry_feature_collection(geometries, results), tmp_path / "maps")
    assert not (tmp_path / "maps").exists()
