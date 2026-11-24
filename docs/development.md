# Development

Implementation notes for contributors. For how results are computed, see [methodology.md](methodology.md).

## Checks

```powershell
uv run pytest                     # full suite with coverage; add -m "not slow" to skip end-to-end tests
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pre-commit run --all-files
```

Tests set `TIMEZONE_EXPERIMENTS_NO_UPDATE_CHECK`, so they never query dataset releases.

## Package layout

The package is layered; a module may only import from its own layer or lower ones, which `tests/test_layering.py` enforces.

| Layer | Packages | Contents |
| --- | --- | --- |
| 0 | `core` | Models, configuration, constants, geometry, timezones, file and fingerprint helpers, progress output. |
| 1 | `astronomy` | Solar noon, sunrise, and sunset. |
| 2 | `data`, `optimization` | Downloads, boundary and population loading, caches; offset search and coverage. The two do not import each other. |
| 3 | `experiments` | Running an experiment and the result files. |
| 4 | `maps` | Map statistics, geometry, caches, and the static and interactive renderers. |
| 5 | `app` | CLI and Streamlit dashboard. |

## Dataset pins and provenance

`data/download/sources.json` pins each dataset's version and holds its URL template, license, and attribution. Each version is part of its download's file name, so changing a pin downloads the new release instead of reusing the old file. GeoBoundaries only serves its current build; its API build details are recorded instead of a pin.

- Every download gets a `<name>.source.json` file with the source, version, URL, retrieval time, SHA-256, and license. Assembled boundary files record the files they were built from.
- `run` writes `<results>.provenance.json`: input files with checksums and download records, package and Python versions, and the tz database version. It is separate from `<results>.config.json`, which `maps` compares against `--config`.
- Maps and the dashboard show a credits line built from those records.

Before downloading Natural Earth or timezone-boundary-builder data, `versions.warn_if_outdated` logs a warning when a newer GitHub release exists. It checks at most once a day (state in `downloads/.release-checks.json`), never fails, and is skipped when `TIMEZONE_EXPERIMENTS_NO_UPDATE_CHECK` is set.

To update the pins:

```powershell
uv run python scripts/update_data_sources.py          # report newer releases of every dataset
uv run python scripts/update_data_sources.py --write  # also update sources.json
```

`--write` only pins a release whose download URL exists. The script exits with 1 when updates are available but not written, or a check failed.

## Caches

Prepared inputs are cached under `DATA_DIRECTORY/cache/prepared/` in three layers. Each layer is one SQLite file named by a digest of everything it depends on:

| Layer | Contents | Keyed by |
| --- | --- | --- |
| `base` | Compressed geometry and sample grid per territory. | Boundary file content, territory and sampling settings, source of the code that builds them. |
| `timezones` | Representative timezone per territory. | Base layer, timezone file, code. |
| `population` | Population cells appended after each territory's base samples. | Base layer, population CSV, code. |

Samples are stored as zlib-compressed native-endian float64 rows (`latitude, longitude, area_weight, longitude_weight` and `latitude, longitude, population`), so the byte order is stored and checked. Solar, coverage, offset, worker, and map settings are not part of any key, so changing them reuses the prepared inputs. Unusable files are logged and rebuilt.

Map exports have two more caches:

- `OUTPUT_DIRECTORY/maps/.map-render-cache.json` fingerprints the territory geometry and results, the requested maps, rendering settings and source, and dependency and Python versions. On an exact repeat, the existing files are checksummed and reused.
- `DATA_DIRECTORY/cache/map-features/` holds intermediate map geometry: borders (keyed by geometry), median-noon bands (also by proposed offsets and the saved median equation of time), and sunrise/sunset cells (also by proposed offsets and solar year).

## Parallelism

`workers` caps each stage, and no stage uses more workers than it has independent jobs:

- Boundary sampling: processes across territories, only for large workloads.
- Timezone and population overlays: threads, since shapely's point-in-polygon tests release the GIL.
- Population raster conversion: threads, each with its own GDAL reader.
- Map exports: processes across the static and HTML rendering jobs, each receiving only the mapped geometry and result values.

Optimization is serial: it is vectorized, and a process pool spent more time pickling territories than it saved. A year's noon statistics come from one table of the equation of time sorted by phase (see `astronomy/solar.py`), so daily clock times are never built per sample and offset.

## Performance

On a 20-thread machine with the world GADM boundaries (599 territories, about 5.4 million samples): territory sampling about 11 s, timezone index 5 s, timezone overlay 5 s, population overlay 10 s, optimization 7 s. Cached runs skip everything except optimization.
