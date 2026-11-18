"""Resolve configured datasets, downloading missing inputs and reusing caches."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError

from timezone_experiments.data.boundaries import (
    _country_code,
    _merge_geometries,
    load_territories,
    load_timezone_boundaries,
)
from timezone_experiments.data.download_boundaries import download
from timezone_experiments.data.download_data import download_timezones, download_world_boundaries, prepare_population
from timezone_experiments.data.population import attach_population_weights, load_population_csv
from timezone_experiments.data.timezone_index import TimezoneBoundaryIndex, annotate_current_timezones

if TYPE_CHECKING:
    from timezone_experiments.config import ExperimentConfig
    from timezone_experiments.data.models import Territory

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InputOptions:
    """Cache settings and optional expert-provided input overrides."""

    data_directory: Path = Path("data/raw")
    boundaries_path: Path | None = None
    population_path: Path | None = None
    timezones_path: Path | None = None
    source_admin_level: int | None = None
    refresh: bool = False


@dataclass(frozen=True, slots=True)
class PreparedInputs:
    """Local datasets ready to load into an experiment."""

    boundaries: Path
    population: Path | None = None
    timezones: Path | None = None
    source_admin_level: int | None = None


def prepare_inputs(config: ExperimentConfig, options: InputOptions | None = None) -> PreparedInputs:
    """Acquire exactly the datasets required by a configuration or overrides."""
    options = options or InputOptions()
    if options.source_admin_level is not None and options.source_admin_level < 0:
        msg = "source_admin_level must be non-negative."
        raise ValueError(msg)
    for path in (options.boundaries_path, options.population_path, options.timezones_path):
        if path is not None and not path.is_file():
            msg = f"Explicit input file does not exist: {path}. Omit the option to download it automatically."
            raise FileNotFoundError(msg)
    needs_population = options.population_path is not None or config.coverage_type == "population"
    needs_timezones = options.timezones_path is not None or config.compare_current_timezones
    total = 1 + needs_population + needs_timezones
    logger.info("Preparing %d input datasets (cache: %s).", total, options.data_directory)
    logger.info("Input 1/%d: administrative boundaries.", total)
    boundaries = prepare_boundaries(config, options)
    population = options.population_path
    if needs_population:
        logger.info("Input 2/%d: population grid.", total)
        if population is None:
            population = _prepare_population(config, options)
        else:
            logger.info("Using explicit population CSV: %s", population)
    timezones = options.timezones_path
    if needs_timezones:
        logger.info("Input %d/%d: current timezone boundaries.", total, total)
        if timezones is not None:
            logger.info("Using explicit timezone boundaries: %s", timezones)
        else:
            timezones = options.data_directory / "world_timezones.geojson"
            if options.refresh or not timezones.is_file():
                logger.info("Downloading current timezone boundaries (large archive).")
                timezones = download_timezones(options.data_directory)
            else:
                logger.info("Using cached timezone boundaries: %s", timezones)
    logger.info("Input datasets ready.")
    return PreparedInputs(boundaries, population, timezones, options.source_admin_level)


def load_experiment_territories(config: ExperimentConfig, options: InputOptions | None = None) -> list[Territory]:
    """Prepare data and load annotated, optionally population-weighted units."""
    inputs = prepare_inputs(config, options)
    logger.info("Loading and processing input data.")
    territories = load_territories(inputs.boundaries, config, source_admin_level=inputs.source_admin_level)
    if inputs.timezones is not None:
        index = TimezoneBoundaryIndex(load_timezone_boundaries(inputs.timezones))
        territories = annotate_current_timezones(territories, index, workers=config.workers)
    if inputs.population is not None:
        territories = attach_population_weights(
            territories, load_population_csv(inputs.population), workers=config.workers
        )
    logger.info("Input processing complete: %d territories ready for optimization.", len(territories))
    return territories


def _prepare_population(config: ExperimentConfig, options: InputOptions) -> Path:
    path = options.data_directory / "population_2025.csv"
    metadata_path = path.with_suffix(".source.json")
    if not options.refresh and path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            msg = f"Invalid population cache metadata in {metadata_path}; use --refresh to replace it."
            raise ValueError(msg)
        if metadata.get("cell_degrees") == config.population_cell_degrees:
            logger.info("Using cached population CSV: %s", path)
            return path
    logger.info("Preparing GHSL 2025 population at %s degrees.", config.population_cell_degrees)
    return prepare_population(
        options.data_directory,
        config.population_cell_degrees,
        refresh=options.refresh,
        workers=config.workers,
    )


def prepare_boundaries(config: ExperimentConfig, options: InputOptions) -> Path:
    """Resolve only administrative boundaries, without population or timezone inputs."""
    if options.source_admin_level is not None and options.source_admin_level < 0:
        msg = "source_admin_level must be non-negative."
        raise ValueError(msg)
    if options.boundaries_path is not None:
        if not options.boundaries_path.is_file():
            msg = f"Explicit input file does not exist: {options.boundaries_path}."
            raise FileNotFoundError(msg)
        logger.info("Using explicit administrative boundaries: %s", options.boundaries_path)
        return options.boundaries_path
    mixed = (
        config.default_admin_level > 0
        or any(level > 0 for level in config.subdivision_rules.values())
        or bool(config.country_boundary_levels)
    )
    settings = {
        "default_admin_level": config.default_admin_level,
        "subdivision_rules": config.subdivision_rules,
        "country_boundary_levels": config.country_boundary_levels,
        "include_antarctica": config.include_antarctica,
    }
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    base_name = "world_countries_with_antarctica.geojson" if config.include_antarctica else "world_countries.geojson"
    path = options.data_directory / (f"world_mixed_{fingerprint}.geojson" if mixed else base_name)
    if path.is_file() and not options.refresh:
        logger.info("Using cached administrative boundaries: %s", path)
        return path
    base_path = options.data_directory / base_name
    if options.refresh or not base_path.is_file():
        logger.info("Downloading Natural Earth world countries (Antarctica included: %s).", config.include_antarctica)
        base_path = download_world_boundaries(options.data_directory, include_antarctica=config.include_antarctica)
    else:
        logger.info("Using cached Natural Earth countries: %s", base_path)
    if not mixed:
        return base_path
    logger.info("Assembling configured administrative levels from %s.", base_path)
    collection = json.loads(base_path.read_text(encoding="utf-8"))
    collection["features"] = _mixed_features(collection["features"], config, options)
    logger.info("Writing %d administrative features to %s.", len(collection["features"]), path)
    path.write_text(json.dumps(collection), encoding="utf-8")
    return path


def _mixed_features(
    features: list[dict[str, Any]],
    config: ExperimentConfig,
    options: InputOptions,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    replaced: set[str] = set()
    countries = {_country_code(feature.get("properties") or {}) for feature in features}
    levels = {
        country: config.country_boundary_levels.get(country, config.admin_level_for(country))
        for country in countries
        if country is not None
    }
    total = sum(level > 0 for level in levels.values())
    for feature in features:
        country = _country_code(feature.get("properties") or {})
        if country is None:
            msg = "Downloaded world boundary feature has no recognized country code."
            raise ValueError(msg)
        level = levels[country]
        if level == 0:
            result.append(feature)
        elif country not in replaced:
            logger.info("Subdivision dataset %d/%d: %s ADM%d.", len(replaced) + 1, total, country, level)
            path = _subdivision_path(country, level, options)
            logger.info("Reading subdivisions from %s.", path)
            collection = json.loads(path.read_text(encoding="utf-8"))
            if collection.get("type") != "FeatureCollection" or not collection.get("features"):
                msg = f"Boundary dataset {path} returned no subdivision features for {country} ADM{level}."
                raise ValueError(msg)
            if country in config.country_boundary_levels:
                matching = [
                    item for item in collection["features"] if _country_code(item.get("properties") or {}) == country
                ]
                if not matching:
                    msg = f"Boundary dataset {path} returned no features belonging to {country}."
                    raise ValueError(msg)
                result.append({**feature, "geometry": _merge_geometries([item["geometry"] for item in matching])})
                result.extend(
                    item for item in collection["features"] if _country_code(item.get("properties") or {}) != country
                )
                logger.warning(
                    "%s uses combined ADM%d source geometry as one country; areas absent from %s are excluded "
                    "from calculations and maps.",
                    country,
                    level,
                    path,
                )
            else:
                result.extend(collection["features"])
            replaced.add(country)
    requested = {country for country, level in config.subdivision_rules.items() if level > 0}
    missing = (requested | config.country_boundary_levels.keys()) - replaced
    if missing:
        msg = f"Configured countries are absent from the world dataset: {', '.join(sorted(missing))}."
        raise ValueError(msg)
    return result


def _subdivision_path(country: str, level: int, options: InputOptions) -> Path:
    if not options.refresh:
        for source in ("gadm", "geoboundaries"):
            path = options.data_directory / f"boundaries_{source}_{country}_{level}.geojson"
            if path.is_file():
                logger.info("Using cached %s %s ADM%d: %s", source, country, level, path)
                return path
    try:
        logger.info("Downloading GADM %s ADM%d for the configured subdivisions.", country, level)
        return download("gadm", country, level, options.data_directory)
    except HTTPError as error:
        if error.code != HTTPStatus.NOT_FOUND:
            raise
        logger.warning(
            "GADM %s ADM%d is unavailable (HTTP 404); trying GeoBoundaries at the same administrative level.",
            country,
            level,
        )
    return download("geoboundaries", country, level, options.data_directory)
