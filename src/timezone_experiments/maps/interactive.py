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
    from pathlib import Path

    from plotly.graph_objects import Figure as PlotlyFigure

    from timezone_experiments.core.models import Territory
    from timezone_experiments.maps.geometry import BorderLine
    from timezone_experiments.maps.scales import Scale
    from timezone_experiments.maps.spec import MapSpec
    from timezone_experiments.optimization.optimizer import AssignmentResult

logger = logging.getLogger(__name__)
PLOTLY_BUNDLE = "plotly.min.js"
_HOVER = (
    "%{text}<br>Actual (%{customdata[5]}): %{customdata[0]}<br>Proposed: %{customdata[1]}"
    "<br>%{customdata[4]}<br>Compliance: %{customdata[2]:.1%}"
    "<br>Fallback below configured coverage: %{customdata[6]}"
)


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
    figure.add_trace(_lines(collection[COUNTRY_BORDERS], "#333333", 0.9, "Country borders"))
    figure.update_geos(showcoastlines=False, showcountries=False, projection_type="natural earth", fitbounds=False)
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
        geojson={"type": "FeatureCollection", "features": features},
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
                spec.details(feature["properties"], row),
                row["noon"]["year"],
                row["fallback_used"],
            ]
            for feature, row, value in zip(features, rows, values, strict=True)
        ],
        hovertemplate=f"{_HOVER}<br>{spec.hover_label}: %{{customdata[3]}}<extra></extra>",
        colorbar=colorbar,
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
        "ticktext": [f"{value:g}" for value in scale.categories],
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
                "geojson": {"type": "FeatureCollection", "features": [features[index] for index in missing]},
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
