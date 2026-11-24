"""Resolve configured datasets, downloading missing inputs and reusing caches."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from timezone_experiments.data.boundaries import (
    _country_code,
    _merge_geometries,
    load_territories,
    load_timezone_boundaries,
    read_feature_collection,
)
from timezone_experiments.data.download.http import DownloadError
from timezone_experiments.data.download_boundaries import download
from timezone_experiments.data.download_data import download_timezones, download_world_boundaries, prepare_population
from timezone_experiments.data.paths import (
    DEFAULT_DATA_DIRECTORY,
    TIMEZONES_FILENAME,
    boundary_filename,
    downloads_directory,
    world_filename,
)
from timezone_experiments.data.population import attach_population_weights, load_population_csv
from timezone_experiments.data.prepared_cache import (
    base_cache_path,
    population_cache_path,
    read_base_territories,
    read_population_layer,
    read_timezone_layer,
    timezone_cache_path,
    write_base_territories,
    write_population_layer,
    write_timezone_layer,
)
from timezone_experiments.data.timezone_index import TimezoneBoundaryIndex, annotate_current_timezones
from timezone_experiments.files import read_json_dataset, write_text_atomic
from timezone_experiments.fingerprint import json_digest

if TYPE_CHECKING:
    from pathlib import Path

    from timezone_experiments.config import ExperimentConfig
    from timezone_experiments.data.models import Territory

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InputOptions:
    """Cache settings and optional expert-provided population/timezone files."""

    data_directory: Path = DEFAULT_DATA_DIRECTORY
    population_path: Path | None = None
    timezones_path: Path | None = None
    refresh: bool = False

    @property
    def downloads(self) -> Path:
        """Directory holding downloaded inputs (DATA_DIRECTORY/downloads)."""
        return downloads_directory(self.data_directory)


@dataclass(frozen=True, slots=True)
class PreparedInputs:
    """Local datasets ready to load into an experiment."""

    boundaries: Path
    timezones: Path
    population: Path | None = None


def prepare_inputs(config: ExperimentConfig, options: InputOptions | None = None) -> PreparedInputs:
    """Acquire exactly the datasets required by a configuration or overrides."""
    options = options or InputOptions()
    for path in (options.population_path, options.timezones_path):
        if path is not None and not path.is_file():
            msg = f"Explicit input file does not exist: {path}. Omit the option to download it automatically."
            raise FileNotFoundError(msg)
    needs_population = options.population_path is not None or config.coverage_type == "population"
    total = 2 + needs_population
    logger.info("Preparing %d input datasets (data directory: %s).", total, options.data_directory)
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
    logger.info("Input %d/%d: current timezone boundaries.", total, total)
    if timezones is not None:
        logger.info("Using explicit timezone boundaries: %s", timezones)
    else:
        timezones = options.downloads / TIMEZONES_FILENAME
        if options.refresh or not timezones.is_file():
            logger.info("Downloading current timezone boundaries (large archive).")
            timezones = download_timezones(options.downloads)
        else:
            logger.info("Using cached timezone boundaries: %s", timezones)
    logger.info("Input datasets ready.")
    return PreparedInputs(boundaries=boundaries, timezones=timezones, population=population)


def load_experiment_territories(config: ExperimentConfig, options: InputOptions | None = None) -> list[Territory]:
    """Prepare data and load annotated, optionally population-weighted territories."""
    options = options or InputOptions()
    inputs = prepare_inputs(config, options)
    logger.info("Loading and processing input data.")
    base_cache = base_cache_path(options.data_directory, inputs.boundaries, config)
    territories = read_base_territories(base_cache)
    if territories is None:
        territories = load_territories(inputs.boundaries, config)
        write_base_territories(base_cache, territories)
    else:
        logger.info("Using cached prepared territory samples: %s", base_cache)
    timezone_cache = timezone_cache_path(options.data_directory, base_cache, inputs.timezones)
    annotated = read_timezone_layer(timezone_cache, territories)
    if annotated is None:
        index = TimezoneBoundaryIndex(load_timezone_boundaries(inputs.timezones))
        annotated = annotate_current_timezones(territories, index, workers=config.workers)
        write_timezone_layer(timezone_cache, annotated)
    else:
        logger.info("Using cached prepared timezone labels: %s", timezone_cache)
    territories = annotated
    if inputs.population is not None:
        population_cache = population_cache_path(options.data_directory, base_cache, inputs.population)
        weighted = read_population_layer(population_cache, territories)
        if weighted is None:
            base_territories = territories
            weighted = attach_population_weights(
                base_territories, load_population_csv(inputs.population), workers=config.workers
            )
            write_population_layer(population_cache, base_territories, weighted)
        else:
            logger.info("Using cached prepared population overlay: %s", population_cache)
        territories = weighted
    logger.info("Input processing complete: %d territories ready for optimization.", len(territories))
    return territories


def _prepare_population(config: ExperimentConfig, options: InputOptions) -> Path:
    path = options.downloads / "population_2025.csv"
    metadata_path = path.with_suffix(".source.json")
    if not options.refresh and path.is_file() and metadata_path.is_file():
        try:
            metadata = read_json_dataset(metadata_path)
        except ValueError as error:
            logger.warning("Regenerating population CSV: %s", error)
            metadata = None
        if isinstance(metadata, dict) and metadata.get("cell_degrees") == config.population_cell_degrees:
            logger.info("Using cached population CSV: %s", path)
            return path
    logger.info("Preparing GHSL 2025 population at %s degrees.", config.population_cell_degrees)
    return prepare_population(
        options.downloads,
        config.population_cell_degrees,
        refresh=options.refresh,
        workers=config.workers,
    )


def prepare_boundaries(config: ExperimentConfig, options: InputOptions) -> Path:
    """Resolve only administrative boundaries, without population or timezone inputs.

    Countries at level 0 come from Natural Earth. When ``countries`` restricts the selection or any country uses
    subdivisions, a configuration-specific file is assembled holding only the selected countries, with GADM (or
    GeoBoundaries) subdivisions downloaded for just those countries.
    """
    mixed = (
        config.default_admin_level > 0
        or any(level > 0 for level in config.subdivision_rules.values())
        or bool(config.merged_subdivision_levels)
    )
    settings = {
        "countries": config.countries,
        "default_admin_level": config.default_admin_level,
        "subdivision_rules": config.subdivision_rules,
        "merged_subdivision_levels": config.merged_subdivision_levels,
        "include_antarctica": config.include_antarctica,
    }
    base_name = world_filename(include_antarctica=config.include_antarctica)
    assembled = mixed or config.countries is not None
    path = options.downloads / (f"world_mixed_{json_digest(settings)[:12]}.geojson" if assembled else base_name)
    if path.is_file() and not options.refresh:
        logger.info("Using cached administrative boundaries: %s", path)
        return path
    base_path = options.downloads / base_name
    if options.refresh or not base_path.is_file():
        logger.info("Downloading Natural Earth world countries (Antarctica included: %s).", config.include_antarctica)
        base_path = download_world_boundaries(options.downloads, include_antarctica=config.include_antarctica)
    else:
        logger.info("Using cached Natural Earth countries: %s", base_path)
    if not assembled:
        return base_path
    logger.info("Assembling configured countries and administrative levels from %s.", base_path)
    collection = read_feature_collection(base_path)
    features = collection["features"]
    if config.countries is not None:
        features = _select_countries(features, config.countries)
    collection["features"] = _mixed_features(features, config, options) if mixed else features
    logger.info("Writing %d administrative features to %s.", len(collection["features"]), path)
    write_text_atomic(path, json.dumps(collection))
    return path


def _select_countries(features: list[dict[str, Any]], countries: tuple[str, ...]) -> list[dict[str, Any]]:
    selected = [feature for feature in features if _country_code(feature.get("properties") or {}) in countries]
    missing = set(countries) - {_country_code(feature.get("properties") or {}) for feature in selected}
    if missing:
        msg = f"countries lists codes absent from the world dataset: {', '.join(sorted(missing))}."
        raise ValueError(msg)
    logger.info("Selected %d configured countries.", len(countries))
    return selected


def _mixed_features(
    features: list[dict[str, Any]],
    config: ExperimentConfig,
    options: InputOptions,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    replaced: set[str] = set()
    countries = {_country_code(feature.get("properties") or {}) for feature in features}
    levels = {
        country: config.merged_subdivision_levels.get(country, config.admin_level_for(country))
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
            collection = read_feature_collection(path)
            if not collection["features"]:
                msg = f"Boundary dataset {path} returned no subdivision features for {country} ADM{level}."
                raise ValueError(msg)
            if country in config.merged_subdivision_levels:
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
    requested |= config.merged_subdivision_levels.keys()
    if config.countries is not None:
        requested &= set(config.countries)
    missing = requested - replaced
    if missing:
        msg = f"Configured countries are absent from the world dataset: {', '.join(sorted(missing))}."
        raise ValueError(msg)
    return result


def _subdivision_path(country: str, level: int, options: InputOptions) -> Path:
    if not options.refresh:
        for source in ("gadm", "geoboundaries"):
            path = options.downloads / boundary_filename(source, country, level)
            if path.is_file():
                logger.info("Using cached %s %s ADM%d: %s", source, country, level, path)
                return path
    try:
        logger.info("Downloading GADM %s ADM%d for the configured subdivisions.", country, level)
        return download("gadm", country, level, options.downloads)
    except DownloadError as error:
        if error.status != HTTPStatus.NOT_FOUND:
            raise
        logger.warning(
            "GADM %s ADM%d is unavailable (HTTP 404); trying GeoBoundaries at the same administrative level.",
            country,
            level,
        )
    return download("geoboundaries", country, level, options.downloads)
