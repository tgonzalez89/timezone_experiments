"""GeoJSON boundary loaders for GADM, GeoBoundaries, and Natural Earth."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from timezone_experiments.core.files import read_json_dataset
from timezone_experiments.core.geometry import GeoJSONError, aggregate_samples, merge_geometries, sample_geometry
from timezone_experiments.core.models import Territory, TerritorySample, pack_base_samples, unpack_base_samples
from timezone_experiments.core.progress import Progress

if TYPE_CHECKING:
    from array import array
    from collections.abc import Iterator

    from timezone_experiments.core.config import ExperimentConfig

logger = logging.getLogger(__name__)
MIN_PARALLEL_SAMPLING_WORK = 1_000_000


class TerritoryKey(NamedTuple):
    """Identity of one configured territory; the boundary features sharing it are merged."""

    country_code: str
    admin_level: int
    local_id: str

    @property
    def territory_id(self) -> str:
        """Return the stable ``COUNTRY:local-id`` identifier used in results and maps."""
        return f"{self.country_code}:{self.local_id}"


@dataclass(frozen=True, slots=True)
class BoundaryGroupMeta:
    """Display name and source timezone label of a territory, read from its boundary features."""

    name: str
    timezone: str | None


def load_territories(
    path: str | Path,
    config: ExperimentConfig,
    *,
    grid_size: int | None = None,
) -> list[Territory]:
    """Load a GeoJSON boundary file into configured country or subdivision territories.

    GADM's GID/NAME fields and GeoBoundaries' shapeType are recognized
    automatically; features without level fields (Natural Earth) are countries.
    """
    file_path = Path(path)
    started = time.perf_counter()
    groups, metadata = _load_boundary_groups(file_path, config)
    resolution = config.sample_grid_size if grid_size is None else grid_size
    territories = _sample_groups(groups, metadata, resolution, config.workers, file_path)
    logger.info(
        "Loaded %d territories with %s sample points from %s in %.2f seconds.",
        len(territories),
        f"{sum(len(territory.samples) for territory in territories):,}",
        file_path,
        time.perf_counter() - started,
    )
    return sorted(territories, key=lambda territory: (territory.country_code, territory.name))


def load_territory_geometries(path: str | Path, config: ExperimentConfig) -> dict[str, dict[str, Any]]:
    """Load configured territory geometry without generating evaluation samples."""
    groups, _ = _load_boundary_groups(Path(path), config)
    geometries = {
        key.territory_id: merge_geometries([feature["geometry"] for feature in group]) for key, group in groups.items()
    }
    logger.info("Loaded boundary geometry for %d territories without sampling.", len(geometries))
    return geometries


def load_excluded_areas(path: str | Path) -> list[dict[str, Any]]:
    """Return the outlines (GeoJSON geometries) of areas merged countries leave out, kept only for drawing maps."""
    areas = read_feature_collection(Path(path)).get("excluded_areas", [])
    return [area for area in areas if isinstance(area, dict)] if isinstance(areas, list) else []


def read_feature_collection(path: Path) -> dict[str, Any]:
    """Read a GeoJSON FeatureCollection, naming the file and the fix when it is corrupt or malformed."""
    collection = read_json_dataset(path)
    if not isinstance(collection, dict) or collection.get("type") != "FeatureCollection":
        msg = f"Expected a GeoJSON FeatureCollection in {path}; if downloaded, refetch it with 'download --refresh'."
        raise GeoJSONError(msg)
    if not isinstance(collection.get("features", []), list):
        msg = f"GeoJSON features in {path} must be an array; if downloaded, refetch it with 'download --refresh'."
        raise GeoJSONError(msg)
    collection.setdefault("features", [])
    return collection


def _load_boundary_groups(
    file_path: Path, config: ExperimentConfig
) -> tuple[dict[TerritoryKey, list[dict[str, Any]]], dict[TerritoryKey, BoundaryGroupMeta]]:
    logger.info("Reading administrative boundaries: %s", file_path)
    features = read_feature_collection(file_path)["features"]
    logger.info("Grouping %d boundary features by the configured administrative levels.", len(features))
    groups: dict[TerritoryKey, list[dict[str, Any]]] = defaultdict(list)
    metadata: dict[TerritoryKey, BoundaryGroupMeta] = {}
    for feature in features:
        properties = feature.get("properties") or {}
        country_code = feature_country_code(properties)
        if country_code is None:
            logger.warning("Skipping feature without a recognized country code in %s.", file_path)
            continue
        level, local_id, name = _feature_level(properties)
        target_level = config.admin_level_for(country_code)
        if target_level > level:
            msg = (
                f"{country_code} is configured at administrative level {target_level}, but the input only "
                f"contains level {level} boundaries."
            )
            raise ValueError(msg)
        selected_level = min(target_level, level)
        key = TerritoryKey(country_code, selected_level, _ancestor_id(properties, selected_level, local_id))
        groups[key].append(feature)
        metadata[key] = BoundaryGroupMeta(
            _ancestor_name(properties, selected_level, name) or key.local_id, _timezone_label(properties) or None
        )
    if not groups:
        msg = f"No usable country-coded features were found in {file_path}."
        raise ValueError(msg)
    return groups, metadata


def _sample_groups(
    groups: dict[TerritoryKey, list[dict[str, Any]]],
    metadata: dict[TerritoryKey, BoundaryGroupMeta],
    resolution: int,
    workers: int,
    path: Path,
) -> list[Territory]:
    geometries = {key: merge_geometries([feature["geometry"] for feature in group]) for key, group in groups.items()}
    labels = {key: f"territory {key.territory_id} ({metadata[key].name}) in {path}" for key in groups}
    vertices = sum(
        len(ring)
        for geometry in geometries.values()
        for polygon in ([geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"])
        for ring in polygon
    )
    workers = min(workers, len(groups)) if vertices * resolution * resolution >= MIN_PARALLEL_SAMPLING_WORK else 1
    logger.info("Sampling %d territories (grid size %d) with %d workers.", len(groups), resolution, workers)
    if workers > 1:
        batches = _parallel_samples(geometries, labels, resolution, workers)
    else:
        batches = ((key, _sample_territory(geometry, resolution, labels[key])) for key, geometry in geometries.items())
    territories: list[Territory] = []
    with Progress("Sampling territory", len(groups)) as progress:
        for position, (key, samples) in enumerate(batches, start=1):
            meta = metadata[key]
            territories.append(
                Territory(
                    territory_id=key.territory_id,
                    country_code=key.country_code,
                    name=meta.name,
                    admin_level=key.admin_level,
                    samples=samples,
                    current_timezone=meta.timezone,
                    geometry=geometries[key],
                ),
            )
            progress.update(position, f"finished {key.territory_id} ({meta.name}), {len(samples):,} sample points")
    return territories


def _sample_territory(geometry: dict[str, Any], resolution: int, label: str) -> tuple[TerritorySample, ...]:
    try:
        return aggregate_samples(list(sample_geometry(geometry, resolution)))
    except ValueError as error:
        msg = f"Could not sample {label}: {error}"
        raise ValueError(msg) from error


def _sample_packed(geometry: dict[str, Any], resolution: int, label: str) -> array[float]:
    return pack_base_samples(_sample_territory(geometry, resolution, label))


def _parallel_samples(
    geometries: dict[TerritoryKey, dict[str, Any]],
    labels: dict[TerritoryKey, str],
    resolution: int,
    workers: int,
) -> Iterator[tuple[TerritoryKey, tuple[TerritorySample, ...]]]:
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_sample_packed, geometry, resolution, labels[key]): key
            for key, geometry in geometries.items()
        }
        for future in as_completed(futures):
            key = futures.pop(future)
            yield key, unpack_base_samples(future.result())


def feature_country_code(properties: dict[str, Any]) -> str | None:
    """Return a GeoJSON feature's upper-case country code from the GADM, Natural Earth, or GeoBoundaries fields."""
    for field in ("GID_0", "ISO_A3", "iso_a3", "ADM0_A3", "shapeGroup", "country_code"):
        value = properties.get(field)
        if value and str(value) not in {"-99", "None"}:
            return str(value).upper()
    return None


def _feature_level(properties: dict[str, Any]) -> tuple[int, str, str]:
    gid_levels = [
        int(field.split("_", maxsplit=1)[1])
        for field in properties
        if field.startswith("GID_") and field[4:].isdigit() and int(field[4:]) > 0
    ]
    explicit_level = properties.get("admin_level", properties.get("adminLevel"))
    shape_type = properties.get("shapeType")
    if explicit_level is not None:
        level = int(explicit_level)
    elif gid_levels:
        level = max(gid_levels)
    elif isinstance(shape_type, str) and shape_type.startswith("ADM"):
        level = int(shape_type[3:])
    else:
        level = 0
    local_id = _admin_value(properties, level, "GID") or _admin_value(properties, level, "ADM")
    if local_id is None:
        local_id = str(properties.get("shapeID") or properties.get("ISO_A3") or properties.get("GID_0") or "unknown")
    name = _admin_value(properties, level, "NAME") or _admin_value(properties, level, "ADM")
    if name is None:
        name = str(properties.get("shapeName") or properties.get("NAME_0") or local_id)
    return level, local_id, name


def _admin_value(properties: dict[str, Any], level: int, prefix: str) -> str | None:
    for key in (f"{prefix}_{level}", f"{prefix}{level}"):
        value = properties.get(key)
        if value:
            return str(value)
    return None


def _ancestor_id(properties: dict[str, Any], level: int, feature_id: str) -> str:
    if level == 0:
        return str(properties.get("GID_0") or properties.get("ISO_A3") or properties.get("shapeGroup") or feature_id)
    return _admin_value(properties, level, "GID") or _admin_value(properties, level, "ADM") or feature_id


def _ancestor_name(properties: dict[str, Any], level: int, feature_name: str) -> str:
    if level == 0:
        return str(properties.get("NAME_0") or properties.get("ADMIN") or properties.get("shapeGroup") or feature_name)
    return _admin_value(properties, level, "NAME") or _admin_value(properties, level, "ADM") or feature_name


def _timezone_label(properties: dict[str, Any]) -> str | None:
    for field in ("tzid", "TZID", "timezone", "TIMEZONE", "tz_name"):
        value = properties.get(field)
        if value:
            return str(value)
    return None


def load_timezone_boundaries(path: str | Path) -> list[tuple[str, dict[str, Any]]]:
    """Load timezone-boundary-builder GeoJSON features with IANA timezone IDs."""
    file_path = Path(path)
    logger.info("Reading current timezone polygons: %s", file_path)
    boundaries: list[tuple[str, dict[str, Any]]] = []
    for feature in read_feature_collection(file_path)["features"]:
        timezone = _timezone_label(feature.get("properties") or {})
        if timezone is not None:
            boundaries.append((timezone, feature["geometry"]))
    if not boundaries:
        msg = f"No timezone identifiers were found in {file_path}."
        raise ValueError(msg)
    logger.info("Loaded %d current timezone polygons.", len(boundaries))
    return boundaries
