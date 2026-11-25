"""Median solar-noon gradient map and its scale."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from factories import (
    daily_noon,
    make_territory,
    map_inputs,
    statistic,
)
from matplotlib.figure import Figure
from plotly.colors import sample_colorscale, unlabel_rgb
from shapely import union_all
from shapely.geometry import box, shape

from timezone_experiments.core.config import (
    ExperimentConfig,
)
from timezone_experiments.maps import (
    MapOptions,
    build_plotly_figure,
    export_maps,
    result_feature_collection,
)
from timezone_experiments.maps.geometry import GRADIENT_OVERLAP_DEGREES
from timezone_experiments.maps.median_noon import median_noon_features, noon_scale
from timezone_experiments.maps.scales import clock
from timezone_experiments.optimization.assessment import (
    calculate_noon_statistics,
)
from timezone_experiments.optimization.optimizer import optimize_territory

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(("offset", "year"), [(0, 2024), (5.75, 2025), (-3.5, 2024)])
def test_median_noon_gradient_matches_daily_median_and_preserves_geometry(offset: float, year: int) -> None:

    geometry: dict[str, object] = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [[0, 0], [2, 0], [2, 3], [0, 3], [0, 0]],
                [[0.5, 1], [1.5, 1], [1.5, 2], [0.5, 2], [0.5, 1]],
            ],
            [[[3.01, 0], [3.02, 0], [3.02, 0.01], [3.01, 0.01], [3.01, 0]]],
        ],
    }
    original = deepcopy(geometry)
    territory = make_territory(geometry=geometry)
    result = optimize_territory(territory, ExperimentConfig(minimum_noon=None, utc_offsets=(offset,)))
    result = replace(result, noon=calculate_noon_statistics(territory, offset, year=year))
    collection = result_feature_collection([territory], [result])
    bands = median_noon_features({**collection, "options": MapOptions(workers=2)})
    assert union_all([shape(band["geometry"]) for band in bands]).equals(shape(geometry))
    assert len({band["id"] for band in bands}) == len(bands)
    for band in bands:
        west, _, east, _ = shape(band["geometry"]).bounds
        # Bands are drawn slightly under their neighbors; the value belongs to the nominal 0.25-degree band.
        nominal_west = int(band["id"].rsplit(":", 1)[1]) * 0.25
        west, east = max(west, nominal_west), min(east, nominal_west + 0.25)
        longitude = band["properties"]["longitude"]
        value = band["properties"]["median_solar_noon"]
        assert value == pytest.approx(statistic(daily_noon(longitude, year, offset), "median"))
        for endpoint in (west, east):
            exact = statistic(daily_noon(endpoint, year, offset), "median")
            assert abs(value - exact) <= 0.5 + 1e-10
    assert geometry == original
    figure = build_plotly_figure([territory], [result], "median_solar_noon")
    trace = figure.data[0]
    assert len(set(trace.z)) > 1
    assert trace.marker.line.width == 0
    assert trace.zmin % 60 == trace.zmax % 60 == 0
    assert 540 <= trace.zmin < trace.zmax <= 900
    assert list(trace.z) == [band["properties"]["median_solar_noon"] for band in bands]
    assert all(trace.zmin <= value <= trace.zmax for value in trace.colorbar.tickvals)
    assert "%{customdata[3]}" in trace.hovertemplate
    assert trace.customdata[0][3] == clock(trace.z[0])
    assert trace.customdata[0][5] == year
    assert [feature["id"] for feature in trace.geojson["features"]] == [band["id"] for band in bands]
    assert figure.data[1].name == "Territory borders"
    assert figure.data[-1].name == "Country borders"


@pytest.mark.parametrize("offset", [-12, 12, 14])
def test_median_noon_gradient_splits_dateline_without_clock_or_color_seams(offset: float) -> None:

    geometry: dict[str, object] = {
        "type": "Polygon",
        "coordinates": [[[179, 0], [-179, 0], [-179, 1], [179, 1], [179, 0]]],
    }
    territory = make_territory(geometry=geometry)
    result = optimize_territory(territory, ExperimentConfig(minimum_noon=None, utc_offsets=(offset,)))
    collection = result_feature_collection([territory], [result])
    bands = median_noon_features({**collection, "options": MapOptions(workers=1)})
    expected = union_all([box(179, 0, 180, 1), box(-180, 0, -179, 1)])
    assert union_all([shape(band["geometry"]) for band in bands]).equals(expected)
    medians = []
    for band in bands:
        west, _, east, _ = shape(band["geometry"]).bounds
        assert -180 <= west < east <= 180
        assert east - west <= 0.25 + GRADIENT_OVERLAP_DEGREES
        medians.append(band["properties"]["median_solar_noon"])
    assert max(medians) - min(medians) == pytest.approx(7)
    assert all(710 < value < 850 for value in medians)


@pytest.mark.parametrize("offset", [-4, 0, 0.05, 1, 4])
def test_static_median_noon_uses_same_values_scale_and_clock_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offset: float
) -> None:

    territories, results = map_inputs((0,))
    results[0] = replace(
        results[0],
        proposed_utc_offset_hours=offset,
        noon=calculate_noon_statistics(territories[0], offset, year=2025),
    )
    interactive = build_plotly_figure(territories, results, "median_solar_noon").data[0]
    assert interactive.zmin % 60 == interactive.zmax % 60 == 0
    captured = []

    def capture(figure: Figure, path: Path, **_kwargs: object) -> None:
        if not path.name.startswith("median_solar_noon"):
            return
        axis = figure.axes[0]
        fill = axis.collections[0]
        observed = fill.get_array()
        assert observed is not None
        np.testing.assert_allclose(sorted(observed), sorted(interactive.z))
        assert fill.norm.vmin == interactive.zmin
        assert fill.norm.vmax == interactive.zmax
        samples = np.linspace(interactive.zmin, interactive.zmax, 9)
        positions = (samples - interactive.zmin) / (interactive.zmax - interactive.zmin)
        scale = [list(stop) for stop in interactive.colorscale]
        expected = np.array([unlabel_rgb(color) for color in sample_colorscale(scale, positions)]) / 255
        np.testing.assert_allclose(np.asarray(fill.cmap(fill.norm(samples)))[:, :3], expected, atol=1 / 128)
        outside = np.array([interactive.zmin - 120, interactive.zmax + 120])
        endpoints = np.array([interactive.zmin, interactive.zmax])
        np.testing.assert_allclose(fill.cmap(fill.norm(outside)), fill.cmap(fill.norm(endpoints)))
        if interactive.zmin < 720 < interactive.zmax:
            np.testing.assert_array_equal(fill.cmap(fill.norm(np.array([720])))[0], (1, 1, 1, 1))
        assert list(figure.axes[1].get_yticks()) == list(interactive.colorbar.tickvals)
        assert [tick.get_text() for tick in figure.axes[1].get_yticklabels()] == list(interactive.colorbar.ticktext)
        assert np.asarray(fill.get_edgecolor()).size == 0
        np.testing.assert_allclose(axis.collections[-2].get_linewidth(), 0.2)
        captured.append(path.suffix)

    monkeypatch.setattr(Figure, "savefig", capture)
    export_maps(result_feature_collection(territories, results), tmp_path, kinds=("static",))
    assert captured == [".png", ".svg"]


@pytest.mark.parametrize(
    ("values", "expected_limits"),
    [
        ([583, 909], (540, 900)),
        ([600, 720, 900], (600, 900)),
        ([750, 780], (720, 780)),
        ([615, 650], (600, 660)),
        ([720, 720], (660, 780)),
        ([780, 780], (720, 840)),
        ([735, 735], (720, 780)),
        ([600, 600], (540, 660)),
        ([1445, 1500], (840, 900)),
        ([-17, 9], (540, 600)),
        ([520, 800], (540, 840)),
        ([480, 1020], (540, 900)),
        ([540, 540], (540, 600)),
        ([900, 900], (840, 900)),
    ],
)
def test_median_noon_scale_rounds_outward_to_hours_and_keeps_white_at_noon(
    values: list[float], expected_limits: tuple[float, float]
) -> None:
    scale = noon_scale(values)
    minimum, maximum, colors = scale.minimum, scale.maximum, scale.colors
    assert (minimum, maximum) == expected_limits
    assert scale.ticks[0] == minimum
    assert scale.ticks[-1] == maximum
    assert all(minimum <= tick <= maximum for tick in scale.ticks)
    labels = [clock(tick) for tick in scale.ticks]
    if min(values) < minimum:
        labels[0] = f"≤ {labels[0]}"
    if max(values) > maximum:
        labels[-1] = f"≥ {labels[-1]}"
    assert scale.tick_labels == tuple(labels)
    if minimum < 720 < maximum:
        white = next(position for position, color in colors if color == "#ffffff")
        assert isinstance(white, (int, float))
        assert minimum + white * (maximum - minimum) == pytest.approx(720)
        assert colors[0][1] == "#2166ac"
        assert colors[-1][1] == "#b35806"
        assert "12:00" in scale.tick_labels
    elif maximum <= 720:
        assert colors[0][1] == "#2166ac"
        assert colors[-1][1] != "#b35806"
    else:
        assert colors[0][1] != "#2166ac"
        assert colors[-1][1] == "#b35806"


def test_median_noon_scale_marks_clamped_ends_only_when_values_exceed_limits() -> None:
    clamped = noon_scale([400, 700, 800])
    assert clamped.tick_labels[0].startswith("≤ ")
    assert not clamped.tick_labels[-1].startswith("≥ ")
    clamped = noon_scale([700, 1000, 1200])
    assert not clamped.tick_labels[0].startswith("≤ ")
    assert clamped.tick_labels[-1].startswith("≥ ")
    unclamped = noon_scale([600, 840])
    assert not any(label.startswith(("≤", "≥")) for label in unclamped.tick_labels)


def test_median_noon_bands_overlap_their_neighbors_without_leaving_the_territory() -> None:
    geometry: dict[str, object] = {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 1], [0, 1], [0, 0]]]}
    territory = make_territory(geometry=geometry)
    result = optimize_territory(territory, ExperimentConfig(minimum_noon=None, utc_offsets=(0,)))
    collection = result_feature_collection([territory], [result])
    bands = {
        int(band["id"].rsplit(":", 1)[1]): shape(band["geometry"])
        for band in median_noon_features({**collection, "options": MapOptions()})
    }
    interior = bands[4]  # 1.0-1.25 degrees, well inside the territory
    assert interior.bounds[0] == pytest.approx(1)
    assert interior.bounds[2] == pytest.approx(1.25 + GRADIENT_OVERLAP_DEGREES)
    assert interior.intersection(bands[5]).area > 0
    assert union_all(list(bands.values())).equals(shape(geometry))
