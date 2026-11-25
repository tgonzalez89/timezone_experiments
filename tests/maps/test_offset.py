"""Proposed-offset map scale."""

from __future__ import annotations

import colorsys
import itertools
from dataclasses import replace

from factories import (
    map_inputs,
    territory_trace,
)

from timezone_experiments.maps import (
    build_plotly_figure,
)
from timezone_experiments.maps.registry import spec_for
from timezone_experiments.maps.scales import Scale, blend_colors, categorical_colors


def test_proposed_offset_scale_remains_discrete() -> None:
    territories, results = map_inputs((0, 0, 0))
    for index, value in enumerate((-2, 0.5, 1.25)):
        results[index] = replace(results[index], proposed_utc_offset_hours=value)
    trace = territory_trace(build_plotly_figure(territories, results, "proposed_utc_offset_hours"))
    assert tuple(trace.z) == (0, 1, 2)
    assert tuple(trace.colorbar.ticktext) == ("-2", "+0.5", "+1.25")
    assert tuple(trace.colorbar.tickvals) == (0, 1, 2)
    assert [stop[0] for stop in trace.colorscale] == [0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1]


def test_categorical_colors_make_neighbors_differ_in_hue_and_lightness() -> None:
    colors = categorical_colors(30)
    assert len(set(colors)) == 30
    hls = [colorsys.rgb_to_hls(*(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))) for color in colors]
    for first, second in itertools.pairwise(hls):
        hue_gap = abs(first[0] - second[0])
        assert min(hue_gap, 1 - hue_gap) > 0.2
        assert abs(first[1] - second[1]) > 0.2


def _offset_scale(*offsets: float) -> Scale:
    features = [{"properties": {"proposed_utc_offset_hours": offset}} for offset in offsets]
    return spec_for("proposed_utc_offset_hours").scale(features)


def test_an_offset_keeps_its_color_whichever_offsets_a_run_assigns() -> None:
    few, many = _offset_scale(3), _offset_scale(-5, 3, 9)
    assert few.category_colors[0] == many.category_colors[1]
    assert few.bands == many.bands  # the background bands do not depend on what is observed either


def test_background_bands_cover_the_zones_in_a_faded_version_of_their_colors() -> None:
    scale = _offset_scale(3)
    assert len(scale.bands) == 25  # UTC-12 to UTC+12
    assert (scale.bands[0][0], scale.bands[0][1]) == (-180, -172.5)  # UTC-12, west half of the date-line zone
    assert (scale.bands[-1][0], scale.bands[-1][1]) == (172.5, 180)  # UTC+12, east half
    west, east, color = scale.bands[12 + 3]  # UTC+3 is centered on 45 degrees
    assert (west, east) == (37.5, 52.5)
    assert color == blend_colors(scale.category_colors[0], "#ffffff", 0.25)
    assert all(first[1] == second[0] for first, second in itertools.pairwise(scale.bands))


def test_zone_bands_are_the_bottom_layer_of_both_renderings() -> None:
    territories, results = map_inputs((0, 0, 0))
    figure = build_plotly_figure(territories, results, "proposed_utc_offset_hours")
    bands = figure.data[0]
    assert bands.name == "UTC zone bands"
    assert not bands.showscale
    assert bands.hoverinfo == "skip"
    assert len(bands.locations) == 25
    assert "Faded background bands" in figure.layout.annotations[0].text
