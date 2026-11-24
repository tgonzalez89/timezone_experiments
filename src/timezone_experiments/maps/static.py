"""Static PNG/SVG renderer (matplotlib/geopandas), driven by a MapSpec and its Scale."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import geopandas as gpd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure as MatplotlibFigure

from timezone_experiments.maps.registry import COUNTRY_BORDERS, TERRITORY_BORDERS
from timezone_experiments.maps.scales import TIMEZONE_BOUNDARIES, TIMEZONE_MERIDIANS
from timezone_experiments.maps.spec import drawn_features, observed_values, options_of

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from timezone_experiments.maps.spec import MapSpec

logger = logging.getLogger(__name__)
_MISSING = {"color": "lightgrey", "label": "No data"}


def export_static_map(collection: dict[str, Any], spec: MapSpec, output: Path) -> list[Path]:
    """Render one map once and save it as both PNG and SVG; returns the two paths."""
    options = options_of(collection)
    features = drawn_features(collection, spec)
    gradient = spec.layer is not None
    frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    # Gradient cells have no outlines (territory borders are drawn on top); subdivisions get thin grey edges.
    edges: str | list[str] = (
        "none"
        if gradient
        else ["none" if feature["properties"]["admin_level"] == 0 else "#999999" for feature in features]
    )
    shared = {"rasterized": options.rasterize_static_map_fills}
    figure = MatplotlibFigure(figsize=(16, 8))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    outputs = [output / f"{spec.field}.{extension}" for extension in ("png", "svg")]
    try:
        if not observed_values(features, spec.field):
            logger.warning("No values are available for map field %s; displaying uncolored territories.", spec.field)
            frame.plot(color="lightgrey", edgecolor=edges, linewidth=0.25, ax=axis, **shared)
            axis.text(0.5, 0.05, "No data available", transform=axis.transAxes, ha="center")
        else:
            scale = spec.scale(features)
            if scale.categorical:
                frame.plot(
                    column=spec.field,
                    categorical=True,
                    categories=list(scale.categories),
                    cmap="viridis",
                    legend=True,
                    legend_kwds={"title": scale.title, "loc": "upper left", "bbox_to_anchor": (1, 1)},
                    edgecolor=edges,
                    linewidth=0.25,
                    ax=axis,
                    missing_kwds=_MISSING,
                    **shared,
                )
            else:
                frame.plot(
                    column=spec.field,
                    cmap=LinearSegmentedColormap.from_list(spec.field, list(scale.colors), N=257),
                    norm=Normalize(vmin=scale.minimum, vmax=scale.maximum),
                    legend=True,
                    legend_kwds={"label": scale.title},
                    edgecolor=edges,
                    linewidth=0 if gradient else 0.25,
                    antialiased=not gradient,
                    ax=axis,
                    missing_kwds=_MISSING,
                    **shared,
                )
                if scale.ticks:
                    figure.axes[1].set_yticks(list(scale.ticks), list(scale.tick_labels))
        if gradient:
            axis.add_collection(
                LineCollection(collection[TERRITORY_BORDERS], colors="#999999", linewidths=0.25, zorder=2)
            )
        _add_timezone_guides(axis)
        axis.add_collection(LineCollection(collection[COUNTRY_BORDERS], colors="#333333", linewidths=0.6, zorder=3))
        footnote = spec.footnote(options)
        if footnote is not None:
            axis.text(0.5, -0.02, footnote, transform=axis.transAxes, ha="center", fontsize=8)
        if options.attribution:
            axis.text(
                1,
                -0.06,
                options.attribution,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="#555555",
            )
        axis.set_title(spec.title)
        axis.set_axis_off()
        for path in outputs:
            figure.savefig(path, bbox_inches="tight", dpi=180)
    finally:
        figure.clear()
    return outputs


def _add_timezone_guides(axis: Axes) -> None:
    limits = axis.get_xlim()
    for longitude in TIMEZONE_BOUNDARIES:
        axis.axvline(longitude, color="0.3", linewidth=0.6, linestyle="--", alpha=0.6)
    for index, longitude in enumerate(TIMEZONE_MERIDIANS):
        axis.axvline(longitude, color="0.3", linewidth=0.5, linestyle=":", alpha=0.4)
        axis.text(
            longitude,
            0.02 + 0.035 * (index % 2),
            f"UTC{longitude // 15:+d}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            fontsize=7,
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5},
        )
    axis.set_xlim(limits)
