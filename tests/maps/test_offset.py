"""Proposed-offset map scale."""

from __future__ import annotations

import colorsys
import itertools
from dataclasses import replace

from factories import (
    map_inputs,
)

from timezone_experiments.maps import (
    build_plotly_figure,
)
from timezone_experiments.maps.scales import categorical_colors


def test_proposed_offset_scale_remains_discrete() -> None:
    territories, results = map_inputs((0, 0, 0))
    for index, value in enumerate((-2, 0.5, 1.25)):
        results[index] = replace(results[index], proposed_utc_offset_hours=value)
    trace = build_plotly_figure(territories, results, "proposed_utc_offset_hours").data[0]
    assert tuple(trace.z) == (0, 1, 2)
    assert tuple(trace.colorbar.ticktext) == ("-2", "0.5", "1.25")
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
