"""Timezone-shift map scale."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from factories import (
    map_inputs,
)
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from timezone_experiments.maps import (
    build_plotly_figure,
    export_maps,
    result_feature_collection,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("shifts", "limit"),
    [
        ((-24, -2, 1.25, 1.25, None), 24),
        ((-2, -1, 0, 0.5, 1, 1.25, 1.5, 2, None), 2),
        ((1, 1, None), 1),
        ((-2, -1, None), 2),
        ((0, 0, None), 1),
        ((None, None), 1),
    ],
)
def test_plotly_shift_scale_is_continuous_and_centered_on_white(shifts: tuple[float | None, ...], limit: float) -> None:
    territories, results = map_inputs(shifts)
    figure = build_plotly_figure(territories, results, "timezone_shift_hours")
    trace = figure.data[0]
    assert tuple(trace.z) == shifts
    assert trace.colorbar.tickvals is None
    assert trace.zmin == -limit
    assert trace.zmax == limit
    assert trace.colorscale == ((0, "#2166ac"), (0.5, "#ffffff"), (1, "#b35806"))
    assert [row[3] for row in trace.customdata] == list(shifts)
    assert "%{customdata[3]}" in trace.hovertemplate
    if any(shift is not None for shift in shifts):
        assert trace.showscale
    else:
        assert not trace.showscale
        assert figure.layout.annotations[0].text == "No data available"


@pytest.mark.parametrize(
    ("shifts", "expected_values"),
    [
        ((-24, -2, 1.25, None), [-24, -2, 1.25]),
        ((-2, 0, 0.5, 1, 1.25, 1.5, 2), [-2, 0, 0.5, 1, 1.25, 1.5, 2]),
        ((1, 1, None), [1]),
        ((0, 0, None), [0]),
        ((None, None), []),
    ],
)
def test_static_shift_scale_is_continuous_while_offsets_stay_discrete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shifts: tuple[float | None, ...], expected_values: list[float]
) -> None:

    legends: dict[str, list[float]] = {}

    def capture(figure: Figure, path: Path, **kwargs: object) -> None:
        if path.stem not in ("proposed_utc_offset_hours", "timezone_shift_hours"):
            return
        axis = figure.axes[0]
        assert {np.asarray(line.get_xdata())[0] for line in axis.lines if line.get_linestyle() == "--"} == {
            longitude + 7.5 for longitude in range(-180, 180, 15)
        }
        assert {np.asarray(line.get_xdata())[0] for line in axis.lines if line.get_linestyle() == ":"} == set(
            range(-180, 181, 15)
        )
        assert {text.get_text() for text in axis.texts} >= {f"UTC{offset:+d}" for offset in range(-12, 13)}
        assert axis.get_xlim()[1] - axis.get_xlim()[0] < 2
        outlines = axis.collections[-1]
        assert isinstance(outlines, LineCollection)
        np.testing.assert_allclose(outlines.get_linewidth(), 0.6)
        legend = axis.get_legend()
        if path.name.startswith("timezone_shift_hours") and expected_values:
            collection = axis.collections[0]
            limit = max(map(abs, expected_values)) or 1
            assert collection.norm.vmin == -limit
            assert collection.norm.vmax == limit
            np.testing.assert_array_equal(collection.cmap(collection.norm(np.array([0])))[0], (1, 1, 1, 1))
            assert len(figure.axes) == 2
            assert figure.axes[1].get_ylabel() == "Timezone shift (hours)"
            observed = collection.get_array()
            assert observed is not None
            legends[path.name] = sorted(set(observed))
            if limit == 2:
                colors = collection.cmap(collection.norm(np.array([1, 1.25, 1.5])))
                np.testing.assert_allclose(colors[1], (colors[0] + colors[2]) / 2)
        elif legend is None:
            legends[path.name] = []
            assert "No data available" in [text.get_text() for text in axis.texts]
        else:
            legends[path.name] = [float(text.get_text()) for text in legend.get_texts() if text.get_text() != "No data"]
            if path.name.startswith("proposed_utc_offset_hours"):
                assert legends[path.name] == [1]

    monkeypatch.setattr(Figure, "savefig", capture)
    territories, results = map_inputs(shifts)
    export_maps(result_feature_collection(territories, results), tmp_path, kinds=("static",))
    assert legends["timezone_shift_hours.png"] == expected_values
    assert legends["timezone_shift_hours.svg"] == expected_values
