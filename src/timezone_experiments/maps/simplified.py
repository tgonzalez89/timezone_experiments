"""One jointly simplified geometry per territory, shared by every map's fills, borders, and gradient cells."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from timezone_experiments.core.fingerprint import json_digest
from timezone_experiments.maps.cache import cache_feature_data, read_feature_cache, write_feature_cache
from timezone_experiments.maps.geometry import SIMPLIFIED, gradient_geometries, wrapped_geometry
from timezone_experiments.maps.spec import options_of

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def simplified_collection(collection: dict[str, Any], digests: Sequence[str]) -> tuple[dict[str, Any], list[str]]:
    """Return the collection with every territory simplified together, and the matching geometry digests.

    Drawing fills, borders, and gradient cells from one simplification keeps them aligned at any zoom, and every
    layer then carries far fewer vertices. A tolerance of 0 leaves the geometry exact. The result is disk-cached
    (one entry covers all territories, since neighbors are simplified jointly), and its digests derive from the
    source digests and the tolerance, so downstream caches need not hash the new coordinates.
    """
    options = options_of(collection)
    tolerance = options.map_geometry_simplification_degrees
    marked = {**collection, SIMPLIFIED: True}
    if tolerance == 0:
        return marked, list(digests)
    features = collection["features"]
    joint = json_digest({"geometry": list(digests), "tolerance": tolerance, "offsets": _offsets(features)})
    cache = cache_feature_data(options.cache_directory, "simplified_geometry", {"joint": joint})
    cached = read_feature_cache(cache)
    if isinstance(cached, list) and len(cached) == len(features) and all(isinstance(item, dict) for item in cached):
        logger.info("Using cached simplified territory geometry.")
        geometries = cached
    else:
        logger.info("Simplifying %d territories together (tolerance %g degrees).", len(features), tolerance)
        areas = gradient_geometries(features, tolerance, options.workers)
        geometries = [
            wrapped_geometry(area, feature["geometry"]["type"]) for area, feature in zip(areas, features, strict=True)
        ]
        write_feature_cache(cache, geometries)
    simplified = [{**feature, "geometry": geometry} for feature, geometry in zip(features, geometries, strict=True)]
    return {**marked, "features": simplified}, [f"{joint}:{index}" for index in range(len(features))]


def _offsets(features: Sequence[dict[str, Any]]) -> list[Any]:
    # Territories are unwrapped around their proposed offset's meridian before simplifying.
    return [feature["properties"]["proposed_utc_offset_hours"] for feature in features]
