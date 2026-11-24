# timezone_experiments

Assigns each country or subdivision a permanent UTC offset chosen by configurable solar-noon rules, and compares it with the current timezone. It explores hypothetical timezones; it does not reproduce current timezone borders.

How the offsets are chosen, what the maps show, and the limitations are described in [docs/methodology.md](docs/methodology.md).

## Install

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.15 or newer. From the repository root:

```powershell
uv sync
```

## Usage

```powershell
uv run timezone_experiments run                                   # whole world, default settings
uv run timezone_experiments run --config examples\population.json --maps
uv run timezone_experiments maps                                  # redraw maps from the last results
uv run timezone_experiments dashboard
uv run timezone_experiments download --config examples\population.json
uv run timezone_experiments clean --cache
```

| Command | Purpose |
| --- | --- |
| `run` | Download or reuse the inputs, choose an offset per territory, write the result CSV (and maps with `--maps`). |
| `maps` | Redraw maps from a result CSV without recomputing it. |
| `dashboard` | Open the interactive dashboard; every control has a tooltip naming its configuration option. |
| `download` | Download the inputs a configuration needs, for offline use. |
| `clean` | Remove downloads, caches, or outputs. |

Run `uv run timezone_experiments <command> --help` for the options of each command. Add `--debug` before the command for full tracebacks.

The first run downloads the boundary data it needs (and, for population coverage, about 461 MiB of population data), which takes several minutes. Later runs reuse the downloads in `data/downloads/` and the prepared inputs in `data/cache/`.

## Configuration

A configuration is a JSON object passed with `--config`. Omitted options keep their defaults, and unknown options are rejected. [examples/](examples/) contains ready-to-run files; [examples/default.json](examples/default.json) lists every default except `workers`.

```json
{
  "countries": ["CAN", "MEX", "USA"],
  "subdivision_rules": {"USA": 1, "CAN": 1},
  "coverage_type": "population",
  "coverage_threshold": 0.95
}
```

Each territory gets the first offset in `utc_offsets`, in order of preference, whose solar noon (summarized per location by `solar_metric`) is within `minimum_noon` and `maximum_noon` for at least `coverage_threshold` of the territory, measured by `coverage_type`. With a minimum bound, the preferred offset is the one giving the earliest noon. If no offset qualifies, the one with the highest coverage is used and `fallback_used` is set in the results.

| Option | Valid values | Default | Description |
| --- | --- | --- | --- |
| `minimum_noon` | `"HH:MM"` (24-hour) or `null` | `"12:00"` | Earliest acceptable solar noon, inclusive. `null` removes the bound. |
| `maximum_noon` | `"HH:MM"` (24-hour) or `null`; not earlier than `minimum_noon` | `null` | Latest acceptable solar noon, inclusive. `null` removes the bound. |
| `solar_metric` | `"min"`, `"max"`, `"median"` | `"median"` | Statistic of each location's daily solar-noon times over `year` that is checked against the bounds. |
| `year` | Integer, 1900-2200 | `2026` | Year of the solar calculations and of the timezone rules used for comparison. |
| `coverage_type` | `"area"`, `"longitude"`, `"population"` | `"area"` | Weight used to measure coverage: land area, share of the longitude span, or population. `"population"` downloads population data. |
| `coverage_threshold` | Number, 0-1 | `0.9` | Share of the weight that must meet the bounds. |
| `utc_offsets` | Non-empty list of numbers in hours, each from -14 to 14 | Whole hours -12 to 14 | Candidate offsets. Fractions are allowed, such as `5.5` or `5.75`. |
| `countries` | List of three-letter ISO codes, or `null` | `null` (all) | Countries to include. Including `"ATA"` requires `include_antarctica`. |
| `default_admin_level` | Integer ≥ 0 | `0` | Administrative level for countries without a rule: 0 = country, 1 = states/provinces, 2 = districts, and so on. |
| `subdivision_rules` | Object mapping a country code to an integer ≥ 0 | 13 countries at level 1 (see below) | Countries assigned per subdivision at the given level. A given object replaces the default; `{}` removes it. |
| `merged_subdivision_levels` | Object mapping a country code to an integer ≥ 1; no overlap with positive `subdivision_rules` | `{"FRA": 1, "NLD": 1}` | Countries loaded at the given level but assigned as one territory, which leaves out areas absent from that level's data (such as overseas territories). `{}` removes it. |
| `include_antarctica` | `true`, `false` | `false` | Include Antarctica. It spans all longitudes, so it usually needs relaxed noon bounds. |
| `sample_grid_size` | Integer ≥ 1 | `12` | Grid size per polygon: up to N x N sample points. Higher values are more precise and slower. |
| `population_cell_degrees` | `0.1`, `0.25`, `0.5`, `1.0` | `1.0` | Cell size, in degrees, of the downloaded population grid. Smaller cells are more precise and slower. |
| `workers` | Integer ≥ 1 | Available CPU threads | Parallel workers for sampling, overlays, population conversion, and maps. |
| `event_map_cell_degrees` | `0.25`, `0.5`, `1.0`, `2.0` | `0.5` | Cell size, in degrees, of the sunrise and sunset maps. |
| `map_geometry_simplification_degrees` | `0`, `0.005`, `0.01`, `0.02` | `0.01` | Outline simplification of the noon, sunrise, and sunset maps; `0` keeps exact geometry. Simplification is skipped wherever it would change a shape's topology or its area by more than 0.1%. |
| `rasterize_static_map_fills` | `true`, `false` | `false` | Store the color fills of PNG/SVG maps as images (borders and labels stay vectors); makes SVG files smaller. |

The default `subdivision_rules` assign these countries per first-level subdivision: USA, CAN, RUS, CHN, AUS, BRA, IND, IDN, MEX, KAZ, MNG, KIR, GRL.

## Outputs

`run` writes, by default under `outputs/`:

- `csv/results.csv`: one row per territory with the proposed offset, the shift from the current offset, coverage, and noon statistics. Columns are described in [docs/methodology.md](docs/methodology.md#result-csv).
- `csv/results.config.json`: the configuration used; `maps` reuses it.
- `csv/results.provenance.json`: input files and dataset versions, and software and tz database versions.
- `maps/` (with `--maps`): PNG, SVG, and interactive HTML maps of the proposed offset, the shift, median solar noon, and earliest and latest sunrise and sunset. Keep `plotly.min.js` next to the HTML files.

## Data sources and licensing

Dataset versions are pinned in [sources.json](src/timezone_experiments/data/download/sources.json); each download is stored with a `.source.json` record of its source, version, and checksum. Maps and the dashboard show a credits line for the datasets used. Redistributed outputs must keep the attributions below.

| Dataset | Used for | License |
| --- | --- | --- |
| [Natural Earth](https://www.naturalearthdata.com/) 1:50m countries | Country boundaries | Public domain |
| [GADM](https://gadm.org/license.html) 4.1 | Subdivisions | Non-commercial use only; redistribution needs permission |
| [geoBoundaries](https://www.geoboundaries.org/) gbOpen | Subdivisions missing from GADM | CC BY 4.0 (some countries, e.g. Kiribati, are OpenStreetMap-derived, ODbL 1.0) |
| [timezone-boundary-builder](https://github.com/evansiroky/timezone-boundary-builder) | Current timezones | ODbL 1.0, © OpenStreetMap contributors |
| [GHSL GHS-POP R2023A](https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php), epoch 2025 | Population coverage | CC BY 4.0, European Union; cite the [dataset](https://doi.org/10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE) and [methodology](https://doi.org/10.1080/17538947.2024.2390454) and state that it was aggregated |

## Development

```powershell
uv run pytest
uv run ruff check .
uv run ty check
```

Package layout, caches, parallelism, and dataset updates are described in [docs/development.md](docs/development.md).
