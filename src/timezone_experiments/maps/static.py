"""Static PNG/SVG renderer (matplotlib/geopandas), driven by a MapSpec and its Scale."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from math import cos, radians
from typing import TYPE_CHECKING, Any

import geopandas as gpd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.figure import Figure as MatplotlibFigure

from timezone_experiments.maps.registry import COUNTRY_BORDERS, TERRITORY_BORDERS
from timezone_experiments.maps.scales import TIMEZONE_BOUNDARIES, TIMEZONE_MERIDIANS
from timezone_experiments.maps.spec import drawn_features, observed_values, options_of

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.axes import Axes

    from timezone_experiments.maps.spec import MapSpec

logger = logging.getLogger(__name__)
_MISSING = {"color": "lightgrey", "label": "No data"}
# Every map of a run shares one canvas size and one map box (a function of the territories' extent only), so the
# exported files line up and open at the same size; the colorbar or legend sits in the margin beside the map.
_FIGURE_WIDTH = 16.0  # inches
_MAP_LEFT = 0.3
_MAP_AREA_WIDTH = 12.2
_MAP_MAX_HEIGHT = 7.0
_MARGIN_TOP = 0.8
_MARGIN_BOTTOM = 1.1
_COLORBAR_GAP = 0.4
_COLORBAR_WIDTH = 0.25
# Border widths in points: countries are drawn dark and thin, borders inside countries lighter and thinner.
_COUNTRY_BORDER_WIDTH = 0.3
_TERRITORY_BORDER_WIDTH = 0.2
_BAND_OVERLAP_DEGREES = 0.05
_EXTENT_PADDING = 0.03  # of the extent, on each side; leaves room for the edge UTC labels
# The PNG is rendered 8K wide. The SVG opens at about 4K (its default display size is set in CSS pixels, at 96 per
# inch against 72 points per inch); its vector drawing is unchanged, so zooming in keeps revealing detail. Fills
# stored as images (``rasterize_static_map_fills``) are rendered at the PNG's resolution.
PNG_WIDTH_PIXELS = 7680
SVG_WIDTH_PIXELS = 3840
PNG_DPI = PNG_WIDTH_PIXELS / _FIGURE_WIDTH
SVG_DISPLAY_SCALE = SVG_WIDTH_PIXELS * 0.75 / (_FIGURE_WIDTH * 72)
_SVG_HEAD_BYTES = 4096
_FRACTION_DIGITS = re.compile(rb"(\.\d{3})\d+")  # digits past the third decimal


def export_static_map(
    collection: dict[str, Any], spec: MapSpec, output: Path, formats: Sequence[str] = ("png", "svg")
) -> list[Path]:
    """Render one map once and save it in each requested format (PNG and/or SVG); returns the paths."""
    options = options_of(collection)
    features = drawn_features(collection, spec)
    gradient = spec.layer is not None
    frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    # Gradient cells have no outlines, and neither do fills stored as images (they would turn the borders into
    # pixels too); in both cases the borders inside countries are drawn on top as vector lines. Otherwise the
    # subdivisions get thin grey edges.
    separate_borders = gradient or options.rasterize_static_map_fills
    edges: str | list[str] = (
        "none"
        if separate_borders
        else ["none" if feature["properties"]["admin_level"] == 0 else "#999999" for feature in features]
    )
    layout = _layout(collection)
    shared = {"rasterized": options.rasterize_static_map_fills, "aspect": layout.aspect}
    figure = MatplotlibFigure(figsize=(_FIGURE_WIDTH, layout.figure_height))
    FigureCanvasAgg(figure)
    axis = figure.add_axes(layout.map_rectangle())
    outputs = [output / f"{spec.field}.{extension}" for extension in formats]
    try:
        if not observed_values(features, spec.field):
            logger.warning("No values are available for map field %s; displaying uncolored territories.", spec.field)
            frame.plot(color="lightgrey", edgecolor=edges, linewidth=_TERRITORY_BORDER_WIDTH, ax=axis, **shared)
            axis.text(0.5, 0.05, "No data available", transform=axis.transAxes, ha="center")
        else:
            scale = spec.scale(features)
            for west, east, color in scale.bands:
                # Each band reaches slightly under the next one, so no hairline shows between bands.
                axis.axvspan(west, east + _BAND_OVERLAP_DEGREES, facecolor=color, linewidth=0, zorder=0)
            if scale.categorical:
                frame.plot(
                    column=spec.field,
                    categorical=True,
                    categories=list(scale.categories),
                    cmap=ListedColormap(list(scale.category_colors)),
                    edgecolor=edges,
                    linewidth=_TERRITORY_BORDER_WIDTH,
                    ax=axis,
                    missing_kwds=_MISSING,
                    **shared,
                )
                count = len(scale.categories)
                legend = figure.colorbar(
                    ScalarMappable(
                        norm=BoundaryNorm(range(count + 1), count), cmap=ListedColormap(list(scale.category_colors))
                    ),
                    cax=figure.add_axes(layout.colorbar_rectangle()),
                    ticks=[index + 0.5 for index in range(count)],
                )
                legend.ax.set_yticklabels(scale.legend_labels)
                legend.ax.tick_params(length=0)
                legend.outline.set_visible(False)
                legend.ax.set_title(scale.title, loc="left", fontsize="medium")
            else:
                frame.plot(
                    column=spec.field,
                    cmap=LinearSegmentedColormap.from_list(spec.field, list(scale.colors), N=257),
                    norm=Normalize(vmin=scale.minimum, vmax=scale.maximum),
                    legend=True,
                    legend_kwds={"label": scale.title},
                    cax=figure.add_axes(layout.colorbar_rectangle()),
                    edgecolor=edges,
                    linewidth=0 if gradient else _TERRITORY_BORDER_WIDTH,
                    antialiased=not gradient,
                    ax=axis,
                    missing_kwds=_MISSING,
                    **shared,
                )
                if scale.ticks:
                    figure.axes[1].set_yticks(list(scale.ticks), list(scale.tick_labels))
        if gradient:
            # The cells lie inside the axes, so skip the per-path clip attribute (about 50 bytes each).
            axis.collections[0].set_clip_on(False)
        if separate_borders:
            axis.add_collection(
                LineCollection(
                    collection[TERRITORY_BORDERS], colors="#999999", linewidths=_TERRITORY_BORDER_WIDTH, zorder=2
                )
            )
        _add_timezone_guides(axis)
        axis.add_collection(
            LineCollection(collection[COUNTRY_BORDERS], colors="black", linewidths=_COUNTRY_BORDER_WIDTH, zorder=3)
        )
        _add_notes(axis, spec.footnote(options), options.attribution)
        axis.set_title(spec.title)
        axis.set_xlim(layout.west, layout.east)
        axis.set_ylim(layout.south, layout.north)
        axis.set_axis_off()
        for path in outputs:
            figure.savefig(path, dpi=PNG_DPI)
            if path.suffix == ".svg":
                _finalize_svg(path)
    finally:
        figure.clear()
    return outputs


@dataclass(frozen=True, slots=True)
class _Layout:
    """The canvas and map box shared by every map of a run, derived only from the territories' extent.

    ``west``..``north`` are the (slightly padded) view limits; ``aspect`` stretches latitude as geopandas would.
    """

    west: float
    south: float
    east: float
    north: float
    aspect: float
    map_width: float
    map_height: float

    @property
    def figure_height(self) -> float:
        return self.map_height + _MARGIN_TOP + _MARGIN_BOTTOM

    def _rectangle(self, left: float, width: float) -> tuple[float, float, float, float]:
        """Return a figure-fraction rectangle spanning the map's height, from ``left`` inches and ``width`` wide."""
        return (
            left / _FIGURE_WIDTH,
            _MARGIN_BOTTOM / self.figure_height,
            width / _FIGURE_WIDTH,
            self.map_height / self.figure_height,
        )

    def map_rectangle(self) -> tuple[float, float, float, float]:
        return self._rectangle(_MAP_LEFT + (_MAP_AREA_WIDTH - self.map_width) / 2, self.map_width)

    def colorbar_rectangle(self) -> tuple[float, float, float, float]:
        return self._rectangle(_MAP_LEFT + _MAP_AREA_WIDTH + _COLORBAR_GAP, _COLORBAR_WIDTH)


def _layout(collection: dict[str, Any]) -> _Layout:
    """Return the shared layout: the map box is the territories' extent at equal-ish scale, as wide as fits."""
    bounds = gpd.GeoDataFrame.from_features(collection["features"], crs="EPSG:4326").total_bounds
    west, south, east, north = (float(value) for value in bounds)
    pad_x, pad_y = (east - west) * _EXTENT_PADDING, (north - south) * _EXTENT_PADDING
    west, east, south, north = west - pad_x, east + pad_x, south - pad_y, north + pad_y
    aspect = 1 / max(cos(radians((south + north) / 2)), 0.2)
    width = _MAP_AREA_WIDTH
    height = width * aspect * (north - south) / (east - west)
    if height > _MAP_MAX_HEIGHT:
        width, height = width * _MAP_MAX_HEIGHT / height, _MAP_MAX_HEIGHT
    return _Layout(west, south, east, north, aspect, width, height)


def _add_notes(axis: Axes, footnote: str | None, attribution: str) -> None:
    """Write the footnote and the data attribution below the map."""
    if footnote is not None:
        axis.text(0.5, -0.02, footnote, transform=axis.transAxes, ha="center", fontsize=8)
    if attribution:
        axis.text(1, -0.06, attribution, transform=axis.transAxes, ha="right", va="top", fontsize=7, color="#555555")


def _finalize_svg(path: Path) -> None:
    """Enlarge the SVG's default display size by ``SVG_DISPLAY_SCALE`` and trim its coordinates to ~0.001 pt.

    The drawing is otherwise unchanged. Matplotlib writes six decimals, far finer than the map's simplification;
    trimming them shrinks the file by about a fifth. Rounding is per number, so vertices that coincide still do.
    """
    if not path.is_file():  # tests stub out Figure.savefig, so nothing was written
        return
    data = path.read_bytes()
    head = data[:_SVG_HEAD_BYTES].decode("utf-8", errors="surrogateescape")

    def scaled(match: re.Match[str]) -> str:
        width, height = (float(match[name]) * SVG_DISPLAY_SCALE for name in ("width", "height"))
        return f'width="{width:.6f}pt" height="{height:.6f}pt"'

    head, sized = re.subn(r'width="(?P<width>[\d.]+)pt" height="(?P<height>[\d.]+)pt"', scaled, head, count=1)
    if not sized:
        logger.warning("Could not find the size in %s; leaving its display size unchanged.", path.name)
    resized = head.encode("utf-8", errors="surrogateescape") + data[_SVG_HEAD_BYTES:]
    path.write_bytes(_FRACTION_DIGITS.sub(rb"\1", resized))


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
