"""Optional static and interactive map exporters."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from math import ceil, floor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from timezone_experiments.astronomy.events import SOLAR_EVENT_FIELDS, annual_solar_events
from timezone_experiments.data.geometry import _ring_winding, _unwrap_longitude, _unwrap_ring, iter_polygons
from timezone_experiments.progress import Progress

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize
    from plotly.graph_objects import Figure
    from shapely.geometry.base import BaseGeometry

    from timezone_experiments.data.geometry import Ring
    from timezone_experiments.data.models import Territory
    from timezone_experiments.optimization.optimizer import AssignmentResult

MAP_FIELDS = (
    "proposed_utc_offset",
    "timezone_shift_hours",
    "mean_solar_noon",
    *SOLAR_EVENT_FIELDS,
)
_TIMEZONE_MERIDIANS = tuple(range(-180, 181, 15))
_TIMEZONE_BOUNDARIES = tuple(longitude + 7.5 for longitude in _TIMEZONE_MERIDIANS[:-1])
_SHIFT_COLORS = ((0.0, "#2166ac"), (0.5, "#ffffff"), (1.0, "#b35806"))
_NOON_BAND_DEGREES = 0.25
_NOON_MINUTES = 12 * 60
_NOON_COLOR_LIMITS = (9 * 60, 15 * 60)
_EVENT_CELL_DEGREES = 0.5
_EVENT_COLORS = ((0.0, "#440154"), (0.25, "#3b528b"), (0.5, "#21918c"), (0.75, "#5ec962"), (1.0, "#fde725"))
logger = logging.getLogger(__name__)
type MapJob = tuple[str, Literal["static", "html"]]
type BorderLine = tuple[tuple[float, float], ...]
_worker_map_inputs: tuple[dict[str, Any], Path] | None = None


def export_maps(
    territories: list[Territory],
    results: list[AssignmentResult],
    output_directory: str | Path,
    *,
    workers: int = 1,
) -> list[Path]:
    """Export PNG/SVG and HTML maps together using the configured worker limit."""
    jobs: list[MapJob] = [(field, "static") for field in MAP_FIELDS]
    jobs.extend((field, "html") for field in MAP_FIELDS)
    return _export_maps(_result_feature_collection(territories, results), Path(output_directory), jobs, workers)


def export_static_maps(
    territories: list[Territory],
    results: list[AssignmentResult],
    output_directory: str | Path,
    *,
    workers: int = 1,
) -> list[Path]:
    """Write PNG and SVG maps for the main assignment and impact statistics."""
    jobs: list[MapJob] = [(field, "static") for field in MAP_FIELDS]
    return _export_maps(_result_feature_collection(territories, results), Path(output_directory), jobs, workers)


def export_result_maps(
    geometries: dict[str, dict[str, Any]],
    results: list[AssignmentResult],
    output_directory: str | Path,
    *,
    workers: int = 1,
) -> list[Path]:
    """Export saved results joined to boundary geometry, without evaluation samples."""
    jobs: list[MapJob] = [(field, "static") for field in MAP_FIELDS]
    jobs.extend((field, "html") for field in MAP_FIELDS)
    return _export_maps(_geometry_result_collection(geometries, results), Path(output_directory), jobs, workers)


def _export_maps(collection: dict[str, Any], output: Path, jobs: list[MapJob], workers: int) -> list[Path]:
    if workers < 1:
        msg = "Map export workers must be positive."
        raise ValueError(msg)
    collection = {**collection, "country_borders": _country_borders(collection, workers=workers)}
    if any(field == "mean_solar_noon" for field, _ in jobs):
        collection["mean_noon_features"] = _mean_noon_features(collection, workers=workers)
    if any(field in SOLAR_EVENT_FIELDS for field, _ in jobs):
        collection["solar_event_features"] = _solar_event_features(collection, workers=workers)
    if any(field == "mean_solar_noon" or field in SOLAR_EVENT_FIELDS for field, _ in jobs):
        collection["territory_borders"] = _country_borders(collection, workers=workers, group_by="territory_id")
    workers = min(workers, len(jobs))
    output.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting %d map jobs with %d workers to %s.", len(jobs), workers, output)
    results: dict[int, list[Path]] = {}
    with Progress("Exporting map jobs", len(jobs)) as progress:
        for completed, (position, paths) in enumerate(_map_results(collection, output, jobs, workers), start=1):
            results[position] = paths
            progress.update(completed, f"finished {', '.join(path.name for path in paths)}")
    return [path for position in range(len(jobs)) for path in results[position]]


def _map_results(
    collection: dict[str, Any], output: Path, jobs: list[MapJob], workers: int
) -> Iterator[tuple[int, list[Path]]]:
    if workers == 1:
        for position, job in enumerate(jobs):
            yield position, _export_map_job(collection, output, job)
        return
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_initialize_map_worker, initargs=(collection, output)
    ) as executor:
        futures = {executor.submit(_export_map_worker, job): position for position, job in enumerate(jobs)}
        for future in as_completed(futures):
            yield futures.pop(future), future.result()


def _initialize_map_worker(collection: dict[str, Any], output: Path) -> None:
    global _worker_map_inputs  # noqa: PLW0603
    _worker_map_inputs = collection, output


def _export_map_worker(job: MapJob) -> list[Path]:
    if _worker_map_inputs is None:
        msg = "Map worker has not been initialized."
        raise RuntimeError(msg)
    collection, output = _worker_map_inputs
    return _export_map_job(collection, output, job)


def _export_map_job(collection: dict[str, Any], output: Path, job: MapJob) -> list[Path]:
    field, kind = job
    if kind == "static":
        return _export_static_map(collection, field, output)
    return [_export_plotly_map(collection, field, output / f"{field}.html")]


def _export_static_map(collection: dict[str, Any], field: str, output: Path) -> list[Path]:
    try:
        import geopandas as gpd  # noqa: PLC0415
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
        from matplotlib.collections import LineCollection  # noqa: PLC0415
        from matplotlib.colors import LinearSegmentedColormap  # noqa: PLC0415
        from matplotlib.figure import Figure  # noqa: PLC0415
    except ImportError as error:
        msg = "Static maps require geopandas and matplotlib; install with `uv sync --extra visualization`."
        raise RuntimeError(msg) from error
    event = field in SOLAR_EVENT_FIELDS
    gradient = field == "mean_solar_noon" or event
    features = (
        collection["solar_event_features"]
        if event
        else collection["mean_noon_features"]
        if gradient
        else collection["features"]
    )
    frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    edges = (
        "none"
        if gradient
        else ["none" if feature["properties"]["admin_level"] == 0 else "#999999" for feature in features]
    )
    values = _observed_values({"features": features}, field)
    if not values:
        logger.warning("No values are available for map field %s; displaying uncolored territories.", field)
    figure = Figure(figsize=(16, 8))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    outputs = [output / f"{field}.{extension}" for extension in ("png", "svg")]
    try:
        if values and (field == "timezone_shift_hours" or gradient):
            minimum, maximum, colors, norm = _static_continuous_style(field, values, features)
            frame.plot(
                column=field,
                cmap=LinearSegmentedColormap.from_list(field, colors, N=257),
                norm=norm,
                legend=True,
                legend_kwds={"label": _map_title(field) if gradient else "Timezone shift (hours)"},
                edgecolor="none" if gradient else edges,
                linewidth=0 if gradient else 0.25,
                antialiased=not gradient,
                ax=axis,
                missing_kwds={"color": "lightgrey", "label": "No data"},
            )
            if gradient:
                ticks = _noon_ticks(minimum, maximum)
                figure.axes[1].set_yticks(ticks, [_noon_clock(value) for value in ticks])
        elif values:
            frame.plot(
                column=field,
                categorical=True,
                categories=values,
                cmap="viridis",
                legend=True,
                legend_kwds={"title": field.replace("_", " "), "loc": "upper left", "bbox_to_anchor": (1, 1)},
                edgecolor=edges,
                linewidth=0.25,
                ax=axis,
                missing_kwds={"color": "lightgrey", "label": "No data"},
            )
        else:
            frame.plot(color="lightgrey", edgecolor=edges, linewidth=0.25, ax=axis)
            axis.text(0.5, 0.05, "No data available", transform=axis.transAxes, ha="center")
        if gradient:
            axis.add_collection(
                LineCollection(collection["territory_borders"], colors="#999999", linewidths=0.25, zorder=2)
            )
        _add_static_timezone_guides(axis)
        axis.add_collection(LineCollection(collection["country_borders"], colors="#333333", linewidths=0.6, zorder=3))
        if event:
            axis.text(
                0.5,
                -0.02,
                "Approximate 0.5-degree cells; extrema over days with events; grey = no events all year",
                transform=axis.transAxes,
                ha="center",
                fontsize=8,
            )
        axis.set_title(_map_title(field))
        axis.set_axis_off()
        for path in outputs:
            figure.savefig(path, bbox_inches="tight", dpi=180)
    finally:
        figure.clear()
    return outputs


def _static_continuous_style(
    field: str, values: list[float], features: list[dict[str, Any]]
) -> tuple[float, float, list[tuple[float, str]], Normalize]:
    from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: PLC0415

    if field == "timezone_shift_hours":
        limit = max(map(abs, values)) or 1.0
        return -limit, limit, list(_SHIFT_COLORS), Normalize(vmin=-limit, vmax=limit)
    if field in SOLAR_EVENT_FIELDS:
        minimum, maximum, colors = _event_scale(_event_values(features, field))
    else:
        minimum, maximum, colors = _noon_scale(values)
        if minimum < _NOON_MINUTES < maximum:
            return (
                minimum,
                maximum,
                list(_SHIFT_COLORS),
                TwoSlopeNorm(vmin=minimum, vcenter=_NOON_MINUTES, vmax=maximum),
            )
    return minimum, maximum, colors, Normalize(vmin=minimum, vmax=maximum)


def _add_static_timezone_guides(axis: Axes) -> None:
    limits = axis.get_xlim()
    for longitude in _TIMEZONE_BOUNDARIES:
        axis.axvline(longitude, color="0.3", linewidth=0.6, linestyle="--", alpha=0.6)
    for index, longitude in enumerate(_TIMEZONE_MERIDIANS):
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


def export_plotly_map(
    territories: list[Territory],
    results: list[AssignmentResult],
    field: str,
    output_path: str | Path,
) -> Path:
    """Write a hoverable world choropleth as standalone HTML."""
    return _export_plotly_map(_result_feature_collection(territories, results), field, Path(output_path))


def _export_plotly_map(collection: dict[str, Any], field: str, path: Path) -> Path:
    figure = _build_plotly_figure(collection, field)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def build_plotly_figure(
    territories: list[Territory],
    results: list[AssignmentResult],
    field: str,
    *,
    workers: int = 1,
) -> Figure:
    """Build a Plotly choropleth for a selected result statistic."""
    collection = _result_feature_collection(territories, results)
    return _build_plotly_figure(collection, field, workers=workers)


def _build_plotly_figure(collection: dict[str, Any], field: str, *, workers: int = 1) -> Figure:
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as error:
        msg = "Interactive Plotly maps require plotly; install with `uv sync --extra visualization`."
        raise RuntimeError(msg) from error
    if field not in MAP_FIELDS:
        msg = f"Unsupported map statistic: {field!r}."
        raise ValueError(msg)
    collection = {
        **collection,
        "features": [
            {**feature, "geometry": _plotly_geometry(feature["geometry"])} for feature in collection["features"]
        ],
    }
    borders = collection.get("country_borders")
    if borders is None:
        borders = _country_borders(collection, workers=workers)
    event = field in SOLAR_EVENT_FIELDS
    gradient = field == "mean_solar_noon" or event
    features = collection["features"]
    source_rows = [feature["properties"] for feature in features]
    if gradient:
        features = collection.get("solar_event_features" if event else "mean_noon_features")
        if features is None:
            features = (
                _solar_event_features(collection, workers=workers)
                if event
                else _mean_noon_features(collection, workers=workers)
            )
        features = [{**feature, "geometry": _plotly_geometry(feature["geometry"])} for feature in features]
        rows = [source_rows[feature["properties"]["territory_index"]] for feature in features]
        map_values = [feature["properties"][field] for feature in features]
    else:
        rows = source_rows
        map_values = [row[field] for row in rows]
    values = sorted({value for value in map_values if value is not None})
    indices = {value: index for index, value in enumerate(values)}
    continuous = field == "timezone_shift_hours" or gradient
    zmin, zmax, colorscale, colorbar = _plotly_scale(field, _event_values(features, field) if event else values)
    figure = go.Figure(
        go.Choropleth(
            geojson={"type": "FeatureCollection", "features": features},
            featureidkey="id",
            locations=[feature["id"] for feature in features],
            z=map_values if continuous else [indices.get(value) for value in map_values],
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale or None,
            marker_line_color="#999999",
            marker_line_width=0 if gradient else [0 if row["admin_level"] == 0 else 0.3 for row in rows],
            showscale=bool(values),
            text=[f"{row['region']} ({row['country']})" for row in rows],
            customdata=[
                [
                    row["noon"]["current_timezone_description"],
                    row["proposed_timezone"],
                    row["compliance_rate"],
                    (
                        _noon_clock(map_values[index])
                        if gradient and map_values[index] is not None
                        else "Not applicable"
                        if event
                        else map_values[index]
                    ),
                    _event_hover(features[index]["properties"], field) if event else _noon_hover(row["noon"]),
                    row["noon"]["year"],
                ]
                for index, row in enumerate(rows)
            ],
            hovertemplate=(
                "%{text}<br>Actual (%{customdata[5]}): %{customdata[0]}<br>Proposed: %{customdata[1]}"
                "<br>%{customdata[4]}<br>Compliance: %{customdata[2]:.1%}"
                + (
                    f"<br>Annual {field.replace('_', ' ')} here (approx.): " + "%{customdata[3]}<extra></extra>"
                    if event
                    else "<br>Annual mean noon here (approx.): %{customdata[3]}<extra></extra>"
                    if gradient
                    else "<br>Value: %{customdata[3]}<extra></extra>"
                )
            ),
            colorbar=colorbar,
        ),
    )
    if event:
        _add_missing_event_trace(figure, features, map_values)
    if gradient:
        territory_borders = collection.get("territory_borders")
        if territory_borders is None:
            territory_borders = _country_borders(collection, workers=workers, group_by="territory_id")
        figure.add_trace(
            go.Scattergeo(
                lon=[value for line in territory_borders for value in (*[point[0] for point in line], None)],
                lat=[value for line in territory_borders for value in (*[point[1] for point in line], None)],
                mode="lines",
                line={"color": "#999999", "width": 0.3},
                name="Territory borders",
                hoverinfo="skip",
                showlegend=False,
            )
        )
    for longitudes, dash, name in (
        (_TIMEZONE_MERIDIANS, "dot", "Solar-time meridians"),
        (_TIMEZONE_BOUNDARIES, "dash", "Solar-zone boundaries"),
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
            lon=_TIMEZONE_MERIDIANS[:-1],
            lat=[-60 - 10 * (index % 2) for index in range(len(_TIMEZONE_MERIDIANS) - 1)],
            text=["UTC-12/+12"] + [f"UTC{longitude // 15:+d}" for longitude in _TIMEZONE_MERIDIANS[1:-1]],
            textposition=["middle left"] + ["middle center"] * (len(_TIMEZONE_MERIDIANS) - 2),
            mode="text",
            textfont={"size": 9, "color": "black"},
            name="Solar UTC labels",
            hoverinfo="skip",
            showlegend=False,
        ),
    )
    figure.add_trace(
        go.Scattergeo(
            lon=[value for line in borders for value in (*[point[0] for point in line], None)],
            lat=[value for line in borders for value in (*[point[1] for point in line], None)],
            mode="lines",
            line={"color": "#333333", "width": 0.9},
            name="Country borders",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.update_geos(showcoastlines=False, showcountries=False, projection_type="natural earth", fitbounds=False)
    figure.update_layout(
        title=_map_title(field),
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    _annotate_map(figure, field, has_values=bool(values))
    return figure


def _add_missing_event_trace(figure: Figure, features: list[dict[str, Any]], values: list[float | None]) -> None:
    import plotly.graph_objects as go  # noqa: PLC0415

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
                "name": "No sunrise/sunset events",
            }
        )
    )


def _annotate_map(figure: Figure, field: str, *, has_values: bool) -> None:
    if not has_values:
        logger.warning("No values are available for map field %s; displaying uncolored territories.", field)
        figure.add_annotation(text="No data available", x=0.5, y=0.05, xref="paper", yref="paper", showarrow=False)
    if field in SOLAR_EVENT_FIELDS:
        figure.add_annotation(
            text="Approximate 0.5-degree cells; extrema over days with events; grey = no events all year",
            x=0.5,
            y=0,
            xref="paper",
            yref="paper",
            showarrow=False,
        )


def _plotly_scale(field: str, values: list[float]) -> tuple[float, float, list[list[float | str]], dict[str, Any]]:
    from plotly.colors import sample_colorscale  # noqa: PLC0415

    if field == "mean_solar_noon" or field in SOLAR_EVENT_FIELDS:
        minimum, maximum, colors = _event_scale(values) if field in SOLAR_EVENT_FIELDS else _noon_scale(values)
        ticks = _noon_ticks(minimum, maximum)
        return (
            minimum,
            maximum,
            [[position, color] for position, color in colors],
            {
                "title": f"{field.replace('_', ' ').title()}<br>(proposed)"
                if field in SOLAR_EVENT_FIELDS
                else "Mean noon<br>(proposed)",
                "tickvals": ticks,
                "ticktext": [_noon_clock(value) for value in ticks],
            },
        )
    if field == "timezone_shift_hours":
        limit = max(map(abs, values), default=0.0) or 1.0
        return (
            -limit,
            limit,
            [[position, color] for position, color in _SHIFT_COLORS],
            {"title": "Timezone shift (hours)"},
        )
    colors = (
        sample_colorscale("Viridis", [index / max(len(values) - 1, 1) for index in range(len(values))])
        if values
        else []
    )
    colorscale = [
        [position, color]
        for index, color in enumerate(colors)
        for position in (index / len(colors), (index + 1) / len(colors))
    ]
    return (
        -0.5,
        max(len(values) - 0.5, 0.5),
        colorscale,
        {
            "title": field.replace("_", " "),
            "tickmode": "array",
            "tickvals": list(range(len(values))),
            "ticktext": [f"{value:g}" for value in values],
        },
    )


def _map_title(field: str) -> str:
    if field == "mean_solar_noon":
        return "Annual Mean Solar Noon (Proposed)"
    if field in SOLAR_EVENT_FIELDS:
        return f"Annual {field.replace('_', ' ').title()} (Proposed)"
    return field.replace("_", " ").title()


def _event_values(features: list[dict[str, Any]], field: str) -> list[float]:
    pair = [name for name in SOLAR_EVENT_FIELDS if name.endswith(field.rsplit("_", maxsplit=1)[-1])]
    return [
        feature["properties"][name] for feature in features for name in pair if feature["properties"][name] is not None
    ]


def _event_scale(values: list[float]) -> tuple[float, float, list[tuple[float, str]]]:
    minimum = floor(min(values, default=0) / 60) * 60
    maximum = ceil(max(values, default=1440) / 60) * 60
    if minimum == maximum:
        minimum, maximum = minimum - 60, maximum + 60
    return minimum, maximum, list(_EVENT_COLORS)


def _noon_scale(values: list[float]) -> tuple[float, float, list[tuple[float, str]]]:
    lower, upper = _NOON_COLOR_LIMITS
    minimum = floor(max(lower, min(upper, min(values, default=_NOON_MINUTES))) / 60) * 60
    maximum = ceil(max(lower, min(upper, max(values, default=_NOON_MINUTES))) / 60) * 60
    if minimum == maximum:
        minimum, maximum = max(lower, minimum - 60), min(upper, maximum + 60)
    blue, white, orange = (color for _, color in _SHIFT_COLORS)
    if maximum <= _NOON_MINUTES:
        colors = [(0.0, blue), (1.0, _blend_colors(blue, white, (maximum - minimum) / (_NOON_MINUTES - minimum)))]
    elif minimum >= _NOON_MINUTES:
        colors = [
            (0.0, _blend_colors(white, orange, (minimum - _NOON_MINUTES) / (maximum - _NOON_MINUTES))),
            (1.0, orange),
        ]
    else:
        colors = [(0.0, blue), ((_NOON_MINUTES - minimum) / (maximum - minimum), white), (1.0, orange)]
    return minimum, maximum, colors


def _blend_colors(start: str, end: str, fraction: float) -> str:
    first = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    last = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    return "#" + "".join(f"{round(a + fraction * (b - a)):02x}" for a, b in zip(first, last, strict=True))


def _noon_ticks(minimum: float, maximum: float) -> list[float]:
    target = (maximum - minimum) / 6
    step = next((value for value in (1, 5, 15, 30, 60, 120, 240) if value >= target), ceil(target / 60) * 60)
    ticks: list[float] = [index * step for index in range(ceil(minimum / step), floor(maximum / step) + 1)]
    ticks = sorted({minimum, *ticks, maximum})
    if minimum <= _NOON_MINUTES <= maximum:
        ticks = sorted({*ticks, _NOON_MINUTES})
    return ticks


def _mean_noon_features(collection: dict[str, Any], *, workers: int) -> list[dict[str, Any]]:
    features = collection["features"]
    logger.info("Preparing mean-noon gradients for %d territories with %d workers.", len(features), workers)
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        Progress("Preparing mean-noon gradients", len(features), enabled=len(features) > 1) as progress,
    ):
        bands: list[dict[str, Any]] = []
        for completed, slices in enumerate(executor.map(_mean_noon_territory, enumerate(features)), start=1):
            bands.extend(slices)
            progress.update(completed)
    return bands


def _mean_noon_territory(task: tuple[int, dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        import numpy as np  # noqa: PLC0415
        from shapely import box, get_parts, intersection, union_all  # noqa: PLC0415
        from shapely.affinity import translate  # noqa: PLC0415
        from shapely.geometry import Polygon  # noqa: PLC0415
    except ImportError as error:
        msg = "Mean-noon maps require numpy and shapely; install with `uv sync --extra visualization`."
        raise RuntimeError(msg) from error
    territory_index, feature = task
    properties = feature["properties"]
    reference = properties["proposed_utc_offset"] * 15
    merged = _gradient_geometry(feature)
    west, _, east, _ = merged.bounds
    indices = np.arange(floor(west / _NOON_BAND_DEGREES), ceil(east / _NOON_BAND_DEGREES))
    edges = indices * _NOON_BAND_DEGREES
    # Vectorized GEOS clipping releases the GIL for the shared worker pool.
    slices = intersection(merged, box(edges, -90, edges + _NOON_BAND_DEGREES, 90))
    features: list[dict[str, Any]] = []
    for index, clipped in zip(indices, slices, strict=True):
        parts = [part for part in get_parts(clipped) if isinstance(part, Polygon) and part.area > 0]
        if not parts:
            continue
        area = union_all(parts)
        west, _, east, _ = area.bounds
        longitude = (west + east) / 2
        mean = (
            720
            - 4 * _unwrap_longitude(longitude, reference)
            - properties["noon"]["mean_equation_of_time_minutes"]
            + 60 * properties["proposed_utc_offset"]
        )
        band = floor((longitude + 180) / 360)
        features.append(
            {
                "type": "Feature",
                "id": f"{feature['id']}:{index}",
                "properties": {
                    "territory_index": territory_index,
                    "mean_solar_noon": mean,
                    "longitude": longitude - 360 * band,
                },
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[list(point) for point in ring.coords] for ring in (part.exterior, *part.interiors)]
                        for part in get_parts(translate(area, xoff=-360 * band))
                    ],
                },
            }
        )
    return features


def _gradient_geometry(feature: dict[str, Any]) -> BaseGeometry:
    try:
        from shapely import get_parts, make_valid, union_all  # noqa: PLC0415
        from shapely.geometry import Polygon  # noqa: PLC0415
    except ImportError as error:
        msg = "Gradient maps require shapely; install with `uv sync --extra visualization`."
        raise RuntimeError(msg) from error
    reference = feature["properties"]["proposed_utc_offset"] * 15
    polygons = []
    for source in iter_polygons(_plotly_geometry(feature["geometry"])):
        outer = _unwrap_ring(source[0])
        anchor = _unwrap_longitude(outer[0][0], reference)
        shift = anchor - outer[0][0]
        rings = [
            [(x + shift, y) for x, y in outer],
            *[[(_unwrap_longitude(x, anchor), y) for x, y in hole] for hole in source[1:]],
        ]
        polygon = Polygon(rings[0], rings[1:])
        valid = polygon if polygon.is_valid else make_valid(polygon)
        polygons.extend(part for part in get_parts(valid) if isinstance(part, Polygon) and part.area > 0)
    return cast("BaseGeometry", union_all(polygons))


def _solar_event_features(collection: dict[str, Any], *, workers: int) -> list[dict[str, Any]]:
    features = collection["features"]
    logger.info("Preparing annual sunrise/sunset gradients for %d territories with %d workers.", len(features), workers)
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        Progress("Preparing sunrise/sunset gradients", len(features), enabled=len(features) > 1) as progress,
    ):
        cells: list[dict[str, Any]] = []
        for completed, slices in enumerate(executor.map(_solar_event_territory, enumerate(features)), start=1):
            cells.extend(slices)
            progress.update(completed)
    return cells


def _solar_event_territory(task: tuple[int, dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        import numpy as np  # noqa: PLC0415
        from shapely import box, get_parts, intersection, union_all  # noqa: PLC0415
        from shapely.affinity import translate  # noqa: PLC0415
    except ImportError as error:
        msg = "Solar-event maps require numpy and shapely; install with `uv sync --extra visualization`."
        raise RuntimeError(msg) from error
    territory_index, feature = task
    source = feature["properties"]
    merged = _gradient_geometry(feature)
    west, south, east, north = merged.bounds
    step = _EVENT_CELL_DEGREES
    columns = np.arange(floor(west / step), ceil(east / step))
    edges = columns * step
    features: list[dict[str, Any]] = []
    for row in range(floor(south / step), ceil(north / step)):
        slices = intersection(merged, box(edges, row * step, edges + step, (row + 1) * step))
        for column, clipped in zip(columns, slices, strict=True):
            parts = [part for part in get_parts(clipped) if part.geom_type == "Polygon" and part.area > 0]
            if not parts:
                continue
            area = union_all(parts)
            point = area.representative_point()
            longitude = (point.x + 180) % 360 - 180
            events = annual_solar_events(point.y, longitude, source["solar_year"], source["proposed_utc_offset"])
            properties: dict[str, Any] = {
                "territory_index": territory_index,
                "latitude": point.y,
                "longitude": longitude,
                "solar_year": source["solar_year"],
                "polar_day_days": events.polar_day_days,
                "polar_night_days": events.polar_night_days,
                "event_dates": {},
            }
            for field in SOLAR_EVENT_FIELDS:
                event = getattr(events, field)
                properties[field] = None if event is None else event.minutes
                properties["event_dates"][field] = None if event is None else event.date.isoformat()
            band = floor((point.x + 180) / 360)
            features.append(
                {
                    "type": "Feature",
                    "id": f"{feature['id']}:{row}:{column}",
                    "properties": properties,
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": [
                            [[list(point) for point in ring.coords] for ring in (part.exterior, *part.interiors)]
                            for part in get_parts(translate(area, xoff=-360 * band))
                        ],
                    },
                }
            )
    return features


def _event_hover(properties: dict[str, Any], field: str) -> str:
    solar_date = properties["event_dates"][field]
    return "<br>".join(
        [
            f"<b>Annual solar events ({properties['solar_year']}) | sampled location</b>",
            f"Location: {properties['latitude']:.3f}, {properties['longitude']:.3f}",
            f"Solar-cycle date: {solar_date}" if solar_date else "Not applicable: no sunrise/sunset events this year",
            f"Polar day: {properties['polar_day_days']} days; polar night: {properties['polar_night_days']} days",
            "Extrema use only days with events; day suffixes are relative to the solar-cycle date.",
            "Flat apparent horizon; standard refraction; no terrain/elevation correction.",
        ]
    )


def _country_borders(
    collection: dict[str, Any], *, workers: int, group_by: Literal["country", "territory_id"] = "country"
) -> list[BorderLine]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for feature in collection["features"]:
        groups.setdefault(feature["properties"][group_by], []).append(feature["geometry"])
    logger.info(
        "Preparing outlines for %d %s with %d workers.",
        len(groups),
        "countries" if group_by == "country" else "territories",
        workers,
    )
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        Progress(
            "Preparing country borders" if group_by == "country" else "Preparing territory borders",
            len(groups),
            enabled=len(groups) > 1,
        ) as progress,
    ):
        borders: list[BorderLine] = []
        # GEOS releases the GIL; map preserves country order across worker counts.
        for completed, lines in enumerate(executor.map(_country_lines, groups.values()), start=1):
            borders.extend(lines)
            progress.update(completed)
    return borders


def _country_lines(geometries: list[dict[str, Any]]) -> list[BorderLine]:
    try:
        from shapely import get_parts, make_valid, union_all  # noqa: PLC0415
        from shapely.geometry import Polygon, box, shape  # noqa: PLC0415
    except ImportError as error:
        msg = "Country outlines require shapely; install with `uv sync --extra visualization`."
        raise RuntimeError(msg) from error
    polygons = []
    reference = None
    for geometry in geometries:
        for part in get_parts(shape(geometry)):
            if not isinstance(part, Polygon) or part.is_empty:
                continue
            if reference is None:
                reference = part.exterior.coords[0][0]
            rings = [
                [(_unwrap_longitude(x, reference), y) for x, y in ring.coords]
                for ring in (part.exterior, *part.interiors)
            ]
            polygon = Polygon(rings[0], rings[1:])
            if polygon.area > 0:
                valid = polygon if polygon.is_valid else make_valid(polygon)
                polygons.extend(part for part in get_parts(valid) if part.geom_type in {"Polygon", "MultiPolygon"})
    if not polygons:
        msg = "Country outlines require nonzero-area polygon geometry."
        raise ValueError(msg)
    source_hole_edges = union_all(
        [
            ring
            for geometry in polygons
            for polygon in get_parts(geometry)
            if isinstance(polygon, Polygon)
            for ring in polygon.interiors
        ]
    )
    rings = []
    for polygon in get_parts(union_all(polygons)):
        rings.append(polygon.exterior)
        # Mismatched subdivision edges can enclose sliver gaps after merging.
        # Only original interior-ring edges can define genuine country holes.
        rings.extend(ring for ring in polygon.interiors if source_hole_edges.covers(ring))
    boundary = union_all(rings)
    west, _, east, _ = boundary.bounds
    lines: list[BorderLine] = []
    # Clip before wrapping, so dateline crossings never draw a line across the map.
    for band in range(floor((west + 180) / 360), floor((east + 180) / 360) + 1):
        clipped = boundary.intersection(box(-180 + 360 * band, -90, 180 + 360 * band, 90))
        lines.extend(
            tuple((x - 360 * band, y) for x, y in line.coords)
            for line in get_parts(clipped)
            if line.geom_type == "LineString"
        )
    return lines


def _noon_hover(noon: dict[str, Any]) -> str:
    earliest = noon["proposed_earliest_minutes"]
    latest = noon["proposed_latest_minutes"]
    details = [
        f"<b>Annual noon extremes ({noon['year']}) | whole territory</b>",
        f"Proposed: earliest {_noon_clock(earliest)}; latest {_noon_clock(latest)}",
    ]
    if not noon["current_ranges"]:
        details.append("Current noon range unavailable")
    else:
        details.append("Current baseline (representative timezone across territory):")
        for label, (earliest, latest) in noon["current_ranges"].items():
            details.append(f"{label}: earliest {_noon_clock(earliest)}; latest {_noon_clock(latest)}")
    return "<br>".join(details)


def _noon_clock(minutes: float) -> str:
    days, clock = divmod(round(minutes), 24 * 60)
    hours, minute = divmod(clock, 60)
    suffix = f" ({days:+d} day)" if days else ""
    return f"{hours:02d}:{minute:02d}{suffix}"


def _plotly_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    # Plotly's spherical renderer expects clockwise exteriors, unlike RFC 7946.
    polygons: list[list[Ring]] = []
    skipped = 0
    for polygon in iter_polygons(geometry):
        rings: list[Ring] = []
        for index, ring in enumerate(polygon):
            winding = _ring_winding(_unwrap_ring(ring))
            if winding == 0:
                skipped += 1
                if index == 0:
                    break
                continue
            closed = ring if ring[0] == ring[-1] else (*ring, ring[0])
            if (winding < 0) != (index == 0):
                closed = tuple(reversed(closed))
            rings.append(closed)
        if rings:
            polygons.append(rings)
    if not polygons:
        msg = "Interactive map geometry contains no nonzero-area polygon components."
        raise ValueError(msg)
    if skipped:
        logger.warning("Skipped %d zero-area rings in interactive map geometry.", skipped)
    coordinates = [[[list(point) for point in ring] for ring in polygon] for polygon in polygons]
    return {"type": geometry["type"], "coordinates": coordinates[0] if geometry["type"] == "Polygon" else coordinates}


def _observed_values(collection: dict[str, Any], field: str) -> list[float]:
    return sorted(
        {feature["properties"][field] for feature in collection["features"] if feature["properties"][field] is not None}
    )


def _result_feature_collection(
    territories: list[Territory],
    results: list[AssignmentResult],
) -> dict[str, Any]:
    assignments = {result.territory_id: result for result in results}
    geometries: dict[str, dict[str, Any]] = {}
    mapped_results: list[AssignmentResult] = []
    for territory in territories:
        assignment = assignments.get(territory.territory_id)
        if assignment is None:
            continue
        if territory.geometry is None:
            msg = f"Map export requires boundary geometry for {territory.territory_id!r}."
            raise ValueError(msg)
        geometries[territory.territory_id] = territory.geometry
        mapped_results.append(assignment)
    return _geometry_result_collection(geometries, mapped_results)


def _geometry_result_collection(
    geometries: dict[str, dict[str, Any]], results: list[AssignmentResult]
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for assignment in results:
        geometry = geometries.get(assignment.territory_id)
        if geometry is None:
            msg = (
                f"No boundary geometry matches result territory {assignment.territory_id!r}. "
                "Use the boundaries and subdivision settings from the original run."
            )
            raise ValueError(msg)
        properties = {
            "territory_id": assignment.territory_id,
            "country": assignment.country,
            "admin_level": assignment.admin_level,
            "region": assignment.region,
            "current_timezone": assignment.current_timezone,
            "proposed_timezone": assignment.proposed_timezone,
            "compliance_rate": assignment.compliance_rate,
            "population_covered": assignment.population_covered,
            "solar_noon_statistic": assignment.solar_noon_statistic,
            "proposed_utc_offset": assignment.proposed_utc_offset,
            "solar_year": assignment.solar_year,
            "timezone_shift_hours": assignment.timezone_shift_hours,
            "noon": asdict(assignment.noon),
        }
        features.append(
            {"type": "Feature", "id": assignment.territory_id, "properties": properties, "geometry": geometry}
        )
    return {"type": "FeatureCollection", "features": features}
