"""Interactive Plotly renderer: HTML exports and the dashboard figure, driven by a MapSpec and its Scale."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from timezone_experiments.core.files import write_text_atomic
from timezone_experiments.maps.features import result_feature_collection
from timezone_experiments.maps.geometry import plotly_geometry
from timezone_experiments.maps.registry import (
    COUNTRY_BORDERS,
    TERRITORY_BORDERS,
    spec_for,
    with_map_layers,
)
from timezone_experiments.maps.scales import TIMEZONE_BOUNDARIES, TIMEZONE_MERIDIANS
from timezone_experiments.maps.spec import MapOptions, drawn_features, options_of

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from plotly.graph_objects import Figure as PlotlyFigure

    from timezone_experiments.core.models import Territory
    from timezone_experiments.maps.geometry import BorderLine
    from timezone_experiments.maps.scales import Scale
    from timezone_experiments.maps.spec import MapSpec
    from timezone_experiments.optimization.optimizer import AssignmentResult

logger = logging.getLogger(__name__)
PLOTLY_BUNDLE = "plotly.min.js"
# Coordinates are written to 4 decimals (about 11 m), far below the map's simplification, which keeps the HTML small.
_COORDINATE_DECIMALS = 4
_POLE = 89.99  # band polygons stop just short of the poles, where spherical polygons degenerate


def _hover(prefix: str, suffix: str) -> str:
    """Return the hover template; ``prefix`` and ``suffix`` are the details text shared by every feature."""
    return (
        "%{text}<br>Actual (%{customdata[5]}): %{customdata[0]}<br>Proposed: %{customdata[1]}"
        f"<br>{prefix}%{{customdata[4]}}{suffix}<br>Compliance: %{{customdata[2]:.1%}}"
        "<br>Fallback below configured coverage: %{customdata[6]}"
    )


def _common_prefix(texts: list[str]) -> str:
    # The prefix shared by the smallest and largest text is shared by all of them.
    first, last = min(texts), max(texts)
    length = 0
    while length < min(len(first), len(last)) and first[length] == last[length]:
        length += 1
    return first[:length]


def _shared_text(texts: list[str]) -> tuple[str, str]:
    """Return the line-aligned prefix and suffix that every text starts and ends with.

    Moving them from each feature's hover data into the hover template, once, keeps the file small; both cut at
    ``<br>`` boundaries so the remainder still reads as whole lines. Text a template would misread is left alone.
    """
    if len(texts) < 2:  # noqa: PLR2004
        return "", ""
    prefix = _common_prefix(texts)
    prefix = prefix[: prefix.rfind("<br>") + 4] if "<br>" in prefix else ""
    suffix = _common_prefix([text[::-1] for text in texts])[::-1]
    suffix = suffix[suffix.find("<br>") :] if "<br>" in suffix else ""
    # The two parts must not overlap in the shortest text.
    if len(prefix) + len(suffix) > min(len(text) for text in texts) or any(
        marker in prefix + suffix for marker in ("%{", "<extra>")
    ):
        return "", ""
    return prefix, suffix


def _geojson(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the features reduced to what the map draws: id and geometry (hover data travels separately)."""
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": feature["id"], "geometry": _rounded(feature["geometry"])} for feature in features
        ],
    }


def _rounded(geometry: dict[str, Any]) -> dict[str, Any]:
    def ring(points: Sequence[Sequence[float]]) -> list[list[float]]:
        return [[round(x, _COORDINATE_DECIMALS), round(y, _COORDINATE_DECIMALS)] for x, y, *_ in points]

    if geometry["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [ring(points) for points in geometry["coordinates"]]}
    return {
        "type": geometry["type"],
        "coordinates": [[ring(points) for points in polygon] for polygon in geometry["coordinates"]],
    }


def build_plotly_figure(
    territories: list[Territory], results: list[AssignmentResult], field: str, options: MapOptions | None = None
) -> PlotlyFigure:
    """Build a Plotly choropleth for one result statistic, reusing cached map layers when options name a cache."""
    collection = {**result_feature_collection(territories, results), "options": options or MapOptions()}
    return plotly_figure(collection, spec_for(field))


def export_plotly_map(collection: dict[str, Any], spec: MapSpec, path: Path) -> Path:
    """Write one HTML map referencing the shared Plotly bundle, which the caller places beside it."""
    plotly_figure(collection, spec).write_html(path, include_plotlyjs=PLOTLY_BUNDLE, full_html=True)
    return path


def ensure_plotly_bundle(directory: Path) -> None:
    """Write plotly.js once per output directory instead of embedding ~4.6 MB in every HTML map."""
    bundle = get_plotlyjs()
    path = directory / PLOTLY_BUNDLE
    try:
        if path.read_text(encoding="utf-8") == bundle:
            return
    except OSError:
        pass
    directory.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, bundle)


def plotly_figure(collection: dict[str, Any], spec: MapSpec) -> PlotlyFigure:
    """Return the interactive map of one spec: fills, optional grey no-value trace, borders, and solar guides."""
    collection = with_map_layers(collection, [spec.field])
    features, rows = _plotly_features(collection, spec)
    values = [feature["properties"][spec.field] for feature in features]
    has_values = any(value is not None for value in values)
    figure = go.Figure(_choropleth(spec, features, rows, values, has_values=has_values), layout={"template": "plotly"})
    if spec.missing_label is not None:
        _add_missing_trace(figure, features, values, spec.missing_label)
    if spec.layer is not None:
        figure.add_trace(_lines(collection[TERRITORY_BORDERS], "#999999", 0.3, "Territory borders"))
    _add_timezone_guides(figure)
    figure.add_trace(_lines(collection[COUNTRY_BORDERS], "#000000", 0.9, "Country borders"))
    # The template paints Natural Earth land (#E5ECF6) and lakes under the map; drawn territories are all it shows.
    scale = spec.scale(features) if has_values else None
    if scale is not None and scale.bands:
        figure.add_trace(_bands(scale))
        figure.data = (figure.data[-1], *figure.data[:-1])  # traces draw in order: bands first, so under all
    figure.update_geos(
        showland=False,
        showlakes=False,
        showocean=False,
        showrivers=False,
        showsubunits=False,
        showcoastlines=False,
        showcountries=False,
        projection_type="natural earth",
        fitbounds=False,
    )
    figure.update_layout(title=spec.title, margin={"l": 0, "r": 0, "t": 40, "b": 0})
    if not has_values:
        logger.warning("No values are available for map field %s; displaying uncolored territories.", spec.field)
        figure.add_annotation(text="No data available", x=0.5, y=0.05, xref="paper", yref="paper", showarrow=False)
    options = options_of(collection)
    footnote = spec.footnote(options)
    if footnote is not None:
        figure.add_annotation(text=footnote, x=0.5, y=0, xref="paper", yref="paper", showarrow=False)
    if options.attribution:
        figure.update_layout(margin={"b": 30})
        figure.add_annotation(
            text=options.attribution,
            x=1,
            y=0,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="top",
            yshift=-8,
            showarrow=False,
            font={"size": 10, "color": "#555555"},
        )
    return figure


def _plotly_features(collection: dict[str, Any], spec: MapSpec) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the drawn features with Plotly-wound geometry, and each one's territory result row."""
    source_rows = [feature["properties"] for feature in collection["features"]]
    features = [
        {**feature, "geometry": plotly_geometry(feature["geometry"])} for feature in drawn_features(collection, spec)
    ]
    if spec.layer is None:
        return features, source_rows
    return features, [source_rows[feature["properties"]["territory_index"]] for feature in features]


def _choropleth(
    spec: MapSpec,
    features: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    values: list[float | None],
    *,
    has_values: bool,
) -> go.Choropleth:
    scale = spec.scale(features)
    details = [spec.details(feature["properties"], row) for feature, row in zip(features, rows, strict=True)]
    prefix, suffix = _shared_text(details)
    if scale.categorical:
        indices = {value: index for index, value in enumerate(scale.categories)}
        z: list[Any] = [indices.get(value) if value is not None else None for value in values]
        zmin, zmax, colorscale, colorbar = _categorical_scale(scale)
    else:
        z = values
        zmin, zmax = scale.minimum, scale.maximum
        colorscale = [[position, color] for position, color in scale.colors]
        colorbar: dict[str, Any] = {"title": scale.title.replace("\n", "<br>")}
        if scale.ticks:
            colorbar |= {"tickvals": list(scale.ticks), "ticktext": list(scale.tick_labels)}
    return go.Choropleth(
        geojson=_geojson(features),
        featureidkey="id",
        locations=[feature["id"] for feature in features],
        z=z,
        zmin=zmin,
        zmax=zmax,
        colorscale=colorscale or None,
        marker_line_color="#999999",
        marker_line_width=0 if spec.layer is not None else [0 if row["admin_level"] == 0 else 0.3 for row in rows],
        showscale=has_values,
        text=[f"{row['territory_name']} ({row['country_code']})" for row in rows],
        customdata=[
            [
                row["noon"]["current_timezone_description"],
                row["proposed_timezone"],
                row["compliance_rate"],
                spec.value_text(value),
                detail[len(prefix) : len(detail) - len(suffix)],
                row["noon"]["year"],
                row["fallback_used"],
            ]
            for detail, row, value in zip(details, rows, values, strict=True)
        ],
        hovertemplate=f"{_hover(prefix, suffix)}<br>{spec.hover_label}: %{{customdata[3]}}<extra></extra>",
        colorbar=colorbar,
    )


def _bands(scale: Scale) -> go.Choropleth:
    """Return the faded background bands as the bottom trace (traces draw in order, so everything covers them)."""
    count = len(scale.bands)
    ids = [str(index) for index in range(count)]
    polygons = [
        [[[west, -_POLE], [west, _POLE], [east, _POLE], [east, -_POLE], [west, -_POLE]]]  # clockwise, as Plotly expects
        for west, east, _ in scale.bands
    ]
    colors = [color for _, _, color in scale.bands]
    return go.Choropleth(
        geojson={
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "id": item, "geometry": {"type": "Polygon", "coordinates": polygon}}
                for item, polygon in zip(ids, polygons, strict=True)
            ],
        },
        featureidkey="id",
        locations=ids,
        z=list(range(count)),
        zmin=-0.5,
        zmax=count - 0.5,
        colorscale=[
            [position, color] for index, color in enumerate(colors) for position in (index / count, (index + 1) / count)
        ],
        showscale=False,
        marker_line_width=0,
        hoverinfo="skip",
        name="UTC zone bands",
    )


def _categorical_scale(scale: Scale) -> tuple[float, float, list[list[float | str]], dict[str, Any]]:
    """Return a stepped colorscale with one flat band and one labeled tick per category."""
    count = len(scale.categories)
    colors = scale.category_colors
    colorscale: list[list[float | str]] = [
        [position, color]
        for index, color in enumerate(colors)
        for position in (index / len(colors), (index + 1) / len(colors))
    ]
    colorbar = {
        "title": scale.title,
        "tickmode": "array",
        "tickvals": list(range(count)),
        "ticktext": scale.legend_labels,
    }
    return -0.5, max(count - 0.5, 0.5), colorscale, colorbar


def _add_missing_trace(
    figure: PlotlyFigure, features: list[dict[str, Any]], values: list[float | None], name: str
) -> None:
    """Draw features without a value as a separate grey trace that keeps their hover text."""
    missing = [index for index, value in enumerate(values) if value is None]
    if not missing:
        return
    trace = figure.data[0].to_plotly_json()
    figure.add_trace(
        go.Choropleth(
            **{
                **trace,
                "geojson": _geojson([features[index] for index in missing]),
                "locations": [features[index]["id"] for index in missing],
                "z": [0] * len(missing),
                "zmin": 0,
                "zmax": 1,
                "colorscale": [[0, "lightgrey"], [1, "lightgrey"]],
                "showscale": False,
                "text": [trace["text"][index] for index in missing],
                "customdata": [trace["customdata"][index] for index in missing],
                "name": name,
            }
        )
    )


def _lines(lines: list[BorderLine], color: str, width: float, name: str) -> go.Scattergeo:
    return go.Scattergeo(
        lon=[value for line in lines for value in (*[point[0] for point in line], None)],
        lat=[value for line in lines for value in (*[point[1] for point in line], None)],
        mode="lines",
        line={"color": color, "width": width},
        name=name,
        hoverinfo="skip",
        showlegend=False,
    )


def _add_timezone_guides(figure: PlotlyFigure) -> None:
    """Dotted solar-time meridians every 15 degrees, dashed ideal zone boundaries, and UTC labels."""
    for longitudes, dash, name in (
        (TIMEZONE_MERIDIANS, "dot", "Solar-time meridians"),
        (TIMEZONE_BOUNDARIES, "dash", "Solar-zone boundaries"),
    ):
        figure.add_trace(
            go.Scattergeo(
                lon=[value for longitude in longitudes for value in (longitude,) * 5 + (None,)],
                lat=[-90, -45, 0, 45, 90, None] * len(longitudes),
                mode="lines",
                line={"color": "rgba(70,70,70,0.5)", "width": 0.6, "dash": dash},
                name=name,
                hoverinfo="skip",
                showlegend=False,
            ),
        )
    figure.add_trace(
        go.Scattergeo(
            lon=TIMEZONE_MERIDIANS[:-1],
            lat=[-60 - 10 * (index % 2) for index in range(len(TIMEZONE_MERIDIANS) - 1)],
            text=["UTC-12/+12"] + [f"UTC{longitude // 15:+d}" for longitude in TIMEZONE_MERIDIANS[1:-1]],
            textposition=["middle left"] + ["middle center"] * (len(TIMEZONE_MERIDIANS) - 2),
            mode="text",
            textfont={"size": 9, "color": "black"},
            name="Solar UTC labels",
            hoverinfo="skip",
            showlegend=False,
        ),
    )
