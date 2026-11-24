"""Proposed-offset map scale."""

from __future__ import annotations

from dataclasses import replace

from factories import (
    map_inputs,
)

from timezone_experiments.maps import (
    build_plotly_figure,
)


def test_proposed_offset_scale_remains_discrete() -> None:
    territories, results = map_inputs((0, 0, 0))
    for index, value in enumerate((-2, 0.5, 1.25)):
        results[index] = replace(results[index], proposed_utc_offset_hours=value)
    trace = build_plotly_figure(territories, results, "proposed_utc_offset_hours").data[0]
    assert tuple(trace.z) == (0, 1, 2)
    assert tuple(trace.colorbar.ticktext) == ("-2", "0.5", "1.25")
    assert tuple(trace.colorbar.tickvals) == (0, 1, 2)
    assert [stop[0] for stop in trace.colorscale] == [0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1]
