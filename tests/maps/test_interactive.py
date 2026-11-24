"""Interactive Plotly maps: hover text and borders."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pytest
from factories import (
    TOY_BOUNDARIES,
    daily_noon,
    make_territory,
    map_inputs,
)
from matplotlib.figure import Figure

from timezone_experiments.core.config import (
    ExperimentConfig,
)
from timezone_experiments.core.models import TerritorySample
from timezone_experiments.core.timezones import describe_timezone
from timezone_experiments.data.boundaries import load_territories
from timezone_experiments.experiments.runner import run_experiment
from timezone_experiments.maps import (
    MAP_FIELDS,
    build_plotly_figure,
    export_maps,
    result_feature_collection,
)
from timezone_experiments.maps.scales import clock
from timezone_experiments.optimization.assessment import (
    calculate_noon_statistics,
)
from timezone_experiments.optimization.optimizer import optimize_territory

if TYPE_CHECKING:
    from pathlib import Path


def test_plotly_hover_shows_actual_standard_and_dst_offsets() -> None:
    territories, results = map_inputs((0, None))
    for index, timezone in enumerate(("America/New_York", None)):
        results[index] = replace(
            results[index],
            current_timezone=timezone,
            noon=calculate_noon_statistics(
                replace(territories[index], current_timezone=timezone),
                results[index].proposed_utc_offset_hours,
                year=2025,
            ),
        )
    figure = build_plotly_figure(territories, results, MAP_FIELDS[0])
    trace = figure.data[0]
    year = 2025
    assert trace.customdata[0][0] == describe_timezone("America/New_York", year=year)
    assert "standard: UTC-05:00" in trace.customdata[0][0]
    assert "DST: UTC-04:00" in trace.customdata[0][0]
    assert trace.customdata[1][0] == "Not available"
    assert trace.customdata[0][1] == results[0].proposed_timezone
    assert trace.customdata[0][5] == year
    assert "Actual (%{customdata[5]}): %{customdata[0]}" in trace.hovertemplate
    assert "Proposed: %{customdata[1]}" in trace.hovertemplate
    details = trace.customdata[0][4]
    assert f"Annual noon extremes ({year})" in details
    assert "Standard time (UTC-05:00): earliest" in details
    assert "DST (UTC-04:00): earliest" in details
    assert "representative timezone across territory" in details
    assert "Current noon range unavailable" in trace.customdata[1][4]
    assert "%{customdata[4]}" in trace.hovertemplate


@pytest.mark.parametrize(
    ("west", "east", "offset"),
    [(-85, -82, -5), (170, -170, 12), (170, -170, -12), (-179, -178, 0)],
)
def test_noon_hover_uses_full_boundaries_and_dateline_frame(west: float, east: float, offset: float) -> None:
    geometry: dict[str, object] = {
        "type": "Polygon",
        "coordinates": [[[west, 0], [east, 0], [east, 1], [west, 1], [west, 0]]],
    }
    territory = make_territory(geometry=geometry, samples=(TerritorySample(0, 0),))
    result = optimize_territory(territory, ExperimentConfig(minimum_noon=None, utc_offsets=(offset,)))
    result = replace(
        result,
        current_timezone="UTC+05:45",
        noon=calculate_noon_statistics(replace(territory, current_timezone="UTC+05:45"), offset, year=result.noon.year),
    )
    figure = build_plotly_figure([territory], [result], MAP_FIELDS[0])
    details = figure.data[0].customdata[0][4]
    year = datetime.now(UTC).year
    unwrapped = [longitude + 360 * round((offset * 15 - longitude) / 360) for longitude in (west, east)]
    eastern = daily_noon((max(unwrapped) + 180) % 360 - 180, year, offset)
    western = daily_noon((min(unwrapped) + 180) % 360 - 180, year, offset)
    east_adjustment = -4 * (max(unwrapped) - ((max(unwrapped) + 180) % 360 - 180))
    west_adjustment = -4 * (min(unwrapped) - ((min(unwrapped) + 180) % 360 - 180))
    expected_earliest = clock(min(eastern) + east_adjustment)
    expected_latest = clock(max(western) + west_adjustment)
    assert f"Proposed: earliest {expected_earliest}; latest {expected_latest}" in details
    assert "Fixed offset UTC+05:45 (DST information unavailable)" in details
    if offset == 0:
        assert "(+1 day)" in details


def test_combined_country_maps_hide_internal_subdivision_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    config = ExperimentConfig(subdivision_rules={}, workers=1, sample_grid_size=2)
    territories = load_territories(TOY_BOUNDARIES, config)
    results = run_experiment(config, territories)
    assert len(territories) == len(results) == 1
    for field in MAP_FIELDS:
        figure = build_plotly_figure(territories, results, field)
        widths = figure.data[0].marker.line.width
        assert widths == 0 if field not in ("proposed_utc_offset_hours", "timezone_shift_hours") else set(widths) == {0}
        borders = figure.data[-1]
        assert not any(
            first[0] == second[0] == 8 and first[1] != second[1]
            for first, second in pairwise(zip(borders.lon, borders.lat, strict=True))
            if None not in (*first, *second)
        )
    saved = []

    def capture(figure: Figure, path: Path, **_kwargs: object) -> None:
        assert not np.asarray(figure.axes[0].collections[0].get_edgecolor())[:, 3].any()
        saved.append(path.name)

    monkeypatch.setattr(Figure, "savefig", capture)
    export_maps(result_feature_collection(territories, results), tmp_path, kinds=("static",))
    assert len(saved) == 2 * len(MAP_FIELDS)
