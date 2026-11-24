"""The map statistics, keyed by field, and assembly of the shared layers they draw."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from timezone_experiments.astronomy.events import SOLAR_EVENT_FIELDS
from timezone_experiments.maps.borders import cached_borders
from timezone_experiments.maps.cache import cached_layer, geometry_digests
from timezone_experiments.maps.median_noon import MedianNoonMap
from timezone_experiments.maps.offset import OffsetMap
from timezone_experiments.maps.shift import ShiftMap
from timezone_experiments.maps.solar_events import SolarEventMap

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from timezone_experiments.maps.spec import MapSpec

# In MAP_FIELDS order.
SPECS: dict[str, MapSpec] = {
    spec.field: spec
    for spec in (OffsetMap(), ShiftMap(), MedianNoonMap(), *(SolarEventMap(field) for field in SOLAR_EVENT_FIELDS))
}
TERRITORY_BORDERS = "territory_borders"
COUNTRY_BORDERS = "country_borders"


def spec_for(field: str) -> MapSpec:
    """Return the map spec of a statistic, rejecting unknown fields."""
    try:
        return SPECS[field]
    except KeyError:
        msg = f"Unsupported map statistic: {field!r}."
        raise ValueError(msg) from None


def map_title(field: str) -> str:
    """Return the human-readable title of a map statistic."""
    return spec_for(field).title


def layer_keys(spec: MapSpec) -> tuple[str, ...]:
    """Return the collection keys of the large per-map layers a spec draws (sent only with its own jobs)."""
    return () if spec.layer is None else (spec.layer.key, TERRITORY_BORDERS)


def with_map_layers(
    collection: dict[str, Any], fields: Sequence[str], digests: Sequence[str] | None = None
) -> dict[str, Any]:
    """Add the (disk-cached) border and gradient layers the given map fields draw, unless already present."""
    digests = digests or geometry_digests(collection["features"])
    wanted: dict[str, Callable[[], object]] = {
        COUNTRY_BORDERS: partial(cached_borders, collection, "country_code", digests)
    }
    for field in fields:
        layer = spec_for(field).layer
        if layer is not None:
            wanted[layer.key] = partial(cached_layer, collection, layer, digests)
            wanted[TERRITORY_BORDERS] = partial(cached_borders, collection, "territory_id", digests)
    return {**collection, **{key: build() for key, build in wanted.items() if key not in collection}}
