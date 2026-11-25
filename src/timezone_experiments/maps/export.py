"""Batch export: every map statistic as static (PNG+SVG) and/or HTML files, in parallel, reusing unchanged outputs."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from timezone_experiments.core.progress import Progress
from timezone_experiments.maps.cache import (
    cached_map_paths,
    geometry_digests,
    map_cache_key,
    write_map_cache,
)
from timezone_experiments.maps.interactive import ensure_plotly_bundle, export_plotly_map
from timezone_experiments.maps.registry import SPECS, layer_keys, spec_for, with_map_layers
from timezone_experiments.maps.spec import MAP_FIELDS, MAP_KINDS, STATIC_FORMATS, MapOptions
from timezone_experiments.maps.static import export_static_map

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from timezone_experiments.maps.spec import MapJob, MapKind

logger = logging.getLogger(__name__)
# Large per-map layers; worker processes receive them only with the jobs that draw them.
_PER_JOB_LAYERS = frozenset(key for spec in SPECS.values() for key in layer_keys(spec))
# Set once per worker process by the pool initializer: the shared map collection and output directory.
_worker_state: dict[str, Any] = {}


def export_maps(
    collection: dict[str, Any],
    output_directory: str | Path,
    options: MapOptions | None = None,
    *,
    kinds: Sequence[MapKind] = MAP_KINDS,
) -> list[Path]:
    """Render every map statistic as each requested kind (PNG+SVG and/or HTML), reusing unchanged outputs.

    ``collection`` comes from ``result_feature_collection`` or ``geometry_feature_collection``.
    """
    options = options or MapOptions()
    output = Path(output_directory)
    jobs: list[MapJob] = [(field, kind) for kind in kinds for field in MAP_FIELDS]
    collection = {**collection, "options": options}
    digests = geometry_digests(collection["features"])
    cache_key = map_cache_key(collection, digests, jobs, options.rendering_settings())
    if "html" in kinds:
        ensure_plotly_bundle(output)
    cached = cached_map_paths(output, jobs, cache_key)
    if cached is not None:
        logger.info("Reused %d cached map files in %s.", len(cached), output)
        return cached
    collection = with_map_layers(collection, [field for field, _ in jobs], digests)
    workers = min(options.workers, len(jobs))
    output.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting %d map jobs with %d workers to %s.", len(jobs), workers, output)
    results: dict[int, list[Path]] = {}
    with Progress("Exporting map jobs", len(jobs)) as progress:
        for completed, (position, paths) in enumerate(_map_results(collection, output, jobs, workers), start=1):
            results[position] = paths
            progress.update(completed, f"finished {', '.join(path.name for path in paths)}")
    paths = [path for position in range(len(jobs)) for path in results[position]]
    write_map_cache(output, cache_key, paths)
    return paths


def _map_results(
    collection: dict[str, Any], output: Path, jobs: list[MapJob], workers: int
) -> Iterator[tuple[int, list[Path]]]:
    if workers == 1:
        for position, job in enumerate(jobs):
            yield position, export_map_job(collection, output, job)
        return
    # Workers receive the shared base once; the large gradient layers travel only with the jobs that draw them.
    base = {key: value for key, value in collection.items() if key not in _PER_JOB_LAYERS}
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker, initargs=(base, output)) as executor:
        futures = {
            executor.submit(_export_in_worker, job, _job_layers(collection, job)): position
            for position, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            yield futures.pop(future), future.result()


def _job_layers(collection: dict[str, Any], job: MapJob) -> dict[str, Any]:
    return {key: collection[key] for key in layer_keys(spec_for(job[0])) if key in collection}


def _initialize_worker(collection: dict[str, Any], output: Path) -> None:
    _worker_state.update(collection=collection, output=output)


def _export_in_worker(job: MapJob, layers: dict[str, Any]) -> list[Path]:
    if not _worker_state:
        msg = "Map worker has not been initialized."
        raise RuntimeError(msg)
    return export_map_job({**_worker_state["collection"], **layers}, _worker_state["output"], job)


def export_map_job(collection: dict[str, Any], output: Path, job: MapJob) -> list[Path]:
    """Render one (field, kind) job; the collection must already hold the layers the field draws."""
    field, kind = job
    spec = spec_for(field)
    if kind in STATIC_FORMATS:
        return export_static_map(collection, spec, output, STATIC_FORMATS[kind])
    return [export_plotly_map(collection, spec, output / f"{field}.html")]
