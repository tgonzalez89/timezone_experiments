"""What a map is: the statistics and output kinds, shared options, gradient layers, and the MapSpec protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from timezone_experiments.astronomy.events import SOLAR_EVENT_FIELDS
from timezone_experiments.core.constants import (
    DEFAULT_MAP_SIMPLIFICATION_DEGREES,
    EVENT_MAP_CELL_DEGREES,
    MAP_SIMPLIFICATION_DEGREES,
    require_positive_workers,
    validate_choice,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from timezone_experiments.core.config import ExperimentConfig
    from timezone_experiments.maps.scales import Scale

MAP_FIELDS = (
    "proposed_utc_offset_hours",
    "timezone_shift_hours",
    "median_solar_noon",
    *SOLAR_EVENT_FIELDS,
)
type MapKind = Literal["static", "html"]
MAP_KINDS: tuple[MapKind, ...] = ("static", "html")
type MapJob = tuple[str, MapKind]
_DEFAULT_EVENT_CELL_DEGREES = 0.5


@dataclass(frozen=True, slots=True, kw_only=True)
class MapOptions:
    """Rendering, caching, and parallelism settings shared by every map export."""

    workers: int = 1
    cache_directory: Path | None = None
    event_map_cell_degrees: float = _DEFAULT_EVENT_CELL_DEGREES
    map_geometry_simplification_degrees: float = DEFAULT_MAP_SIMPLIFICATION_DEGREES
    rasterize_static_map_fills: bool = False
    # Data attribution drawn under every map (e.g. "Data: Natural Earth v5.1.2; ..."); empty draws none.
    attribution: str = ""

    def __post_init__(self) -> None:
        """Validate worker count and the allowed resolution values."""
        require_positive_workers(self.workers, "Map export")
        validate_choice("event_map_cell_degrees", self.event_map_cell_degrees, EVENT_MAP_CELL_DEGREES)
        validate_choice(
            "map_geometry_simplification_degrees",
            self.map_geometry_simplification_degrees,
            MAP_SIMPLIFICATION_DEGREES,
        )

    @classmethod
    def from_config(
        cls, config: ExperimentConfig, *, cache_directory: Path | None = None, attribution: str = ""
    ) -> MapOptions:
        """Take the map settings of an experiment configuration, adding the data attribution to draw."""
        return cls(
            workers=config.workers,
            cache_directory=cache_directory,
            event_map_cell_degrees=config.event_map_cell_degrees,
            map_geometry_simplification_degrees=config.map_geometry_simplification_degrees,
            rasterize_static_map_fills=config.rasterize_static_map_fills,
            attribution=attribution,
        )

    def rendering_settings(self) -> dict[str, object]:
        """Return the settings that change rendered output (not workers or cache location)."""
        return {
            "event_map_cell_degrees": self.event_map_cell_degrees,
            "map_geometry_simplification_degrees": self.map_geometry_simplification_degrees,
            "rasterize_static_map_fills": self.rasterize_static_map_fills,
            "attribution": self.attribution,
        }


def options_of(collection: dict[str, Any]) -> MapOptions:
    """Return the options attached to a map collection (defaults when none are attached)."""
    options = collection.get("options")
    return options if isinstance(options, MapOptions) else MapOptions()


@dataclass(frozen=True, slots=True)
class GradientLayer:
    """Derived features (noon bands, event cells) drawn instead of whole territories, built once and cached.

    ``build`` turns a map collection into the derived features; ``dependency`` returns, per territory, the inputs
    beyond geometry, proposed offset, and simplification that the features depend on (they key the disk cache).
    """

    name: str
    key: str
    build: Callable[[dict[str, Any]], list[dict[str, Any]]]
    dependency: Callable[[dict[str, Any], MapOptions], dict[str, Any]]


class MapSpec(Protocol):
    """Everything specific to one map statistic; the static and interactive renderers are driven only by this.

    Map types subclass this explicitly. ``layer`` is None for maps that color whole territories. ``hover_label``
    names the per-feature value in the interactive hover, and ``missing_label`` (when set) names an extra grey
    trace for features without a value.
    """

    field: str
    title: str
    layer: GradientLayer | None
    hover_label: str
    missing_label: str | None

    def scale(self, features: list[dict[str, Any]]) -> Scale:
        """Return the color scale for the drawn features."""
        ...

    def value_text(self, value: float | None) -> object:
        """Return the hover text (or raw value) for one feature's value."""
        ...

    def details(self, properties: dict[str, Any], row: dict[str, Any]) -> str:
        """Return the hover detail block for a drawn feature and its territory's result row."""
        ...

    def footnote(self, options: MapOptions) -> str | None:
        """Return a note shown under the map, if any."""
        ...


def drawn_features(collection: dict[str, Any], spec: MapSpec) -> list[dict[str, Any]]:
    """Return the features a map draws: its gradient layer's features, or the territories themselves."""
    return collection[spec.layer.key] if spec.layer is not None else collection["features"]


def observed_values(features: list[dict[str, Any]], field: str) -> list[float]:
    """Return the sorted distinct non-missing values of a feature property."""
    return sorted({feature["properties"][field] for feature in features if feature["properties"][field] is not None})
