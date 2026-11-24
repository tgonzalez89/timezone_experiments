# timezone_experiments

Explore hypothetical permanent UTC-offset assignments using configurable solar-noon constraints. This research framework does not attempt to reproduce present-day timezone borders.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run commands from the repository root. Python 3.15 or newer is required.

```powershell
uv sync
```

All dependencies, including those for population-raster conversion, maps, and the dashboard, are installed by `uv sync`. IANA timezone rules come from the `tzdata` package, so they work on Windows too.

## Run the whole world

```powershell
uv run timezone_experiments run
```

That's it: missing boundaries are downloaded automatically, then one permanent offset per selected country or subdivision is written to `outputs/csv/results.csv`, with the configuration used saved beside it as `outputs/csv/results.config.json`. Add `--maps` to also export maps. By default, the 13 countries and territories listed below are split into first-level subdivisions, while others stay at country level. France and the Netherlands use finer boundary datasets combined into one country each, excluding overseas areas absent from those sources. The worker count is detected automatically from the CPU threads available to Python. No external files or preparatory download command are required. Initial setup downloads the required administrative datasets and can take considerably longer than a cached run.

The defaults require each location's **median** annual solar noon to be at or after **12:00** across **90% of the territory's sampled land area**, using whole-hour offsets from UTC-12 through UTC+14 and reference year **2026**. Among feasible offsets, the optimizer chooses the **earliest solar noon** at or after `minimum_noon`; with only `maximum_noon` set it chooses the latest noon at or before it, and with neither bound the offset nearest the territory's solar meridian. There is no additional area-weighted alignment score. [examples/default.json](examples/default.json) lists the static defaults; it omits `workers` so the count can be detected on each machine.

**Antarctica is excluded by default:** it spans all longitudes, so one offset cannot meet the default solar-noon constraint throughout the territory. To include it, set `"include_antarctica": true` in your configuration and relax the noon constraint, for example `"minimum_noon": null`.

### Use a configuration

```powershell
uv run timezone_experiments run --config examples\default.json
uv run timezone_experiments run --config examples\full_area_coverage.json
```

The configuration determines which inputs are prepared:

| Setting | Automatic input preparation |
| --- | --- |
| Countries using administrative level 0 | Natural Earth 1:50m worldwide country boundaries. |
| `subdivision_rules` or `default_admin_level` above 0 | Download the required GADM administrative levels, using GeoBoundaries at the same level if GADM reports the dataset missing, and assemble a worldwide mixed-level GeoJSON. |
| `coverage_type: "population"` | Download GHSL population counts and convert to CSV. |
| Every run | Download or reuse the pinned timezone-boundary-builder land polygons. |

Downloaded inputs are stored under `data/downloads/`, while reusable prepared and map caches are stored under `data/cache/`. Existing matching inputs are reused, including converted population CSVs; changing population resolution regenerates the CSV from the cached source ZIP. Mixed administrative datasets have configuration-specific cache names. A GADM HTTP 404 triggers a logged attempt to obtain the same administrative level from GeoBoundaries; other download failures are reported directly. This supplies Kiribati's Gilbert, Line, and Phoenix island groups. Downloads fail explicitly if neither provider supplies the requested level; country boundaries are never substituted.

Sampled territories, timezone labels, and population overlays are also cached under `data/cache/prepared/`, so changing only optimization settings (solar metric, bounds, year, coverage, candidate offsets) skips input processing. Corrupt or outdated caches are rebuilt automatically; `clean --cache` discards them all. See [Caches](#caches) for details.

Use `--data-directory DIR` to choose another data directory; downloads then go to `DIR/downloads/` and caches to `DIR/cache/`, the same layout every command (including `clean`) uses. To download the required inputs again, run `download --refresh` (or `clean --downloads` and let the next run fetch them). Explicit custom input files are never replaced. Matching cached inputs work offline. Downloaded data and generated outputs are ignored by Git.

### Dataset versions and provenance

Dataset versions are pinned in [sources.json](src/timezone_experiments/data/download/sources.json). Each version is part of its download's file name, so downloading again returns the same data, and changing a pin fetches the new release instead of reusing the old file. GeoBoundaries only serves its current build, so the build details its API reports are recorded instead of a pin.

- Each download gets a `.source.json` file beside it recording the source, version, URL, retrieval time, SHA-256, and license. Assembled boundary files list the files they were built from.
- `run` writes `results.provenance.json` beside the result CSV: the input files with their checksums and download records, the package and Python versions, and the tz database version.
- Maps and the dashboard show a credits line naming the datasets and versions behind the results.

Before downloading Natural Earth or timezone-boundary-builder data, a check warns when a release newer than the pinned one exists. It runs at most once a day and never stops a run; set `TIMEZONE_EXPERIMENTS_NO_UPDATE_CHECK=1` to skip it. To update the pins:

```powershell
uv run python scripts/update_data_sources.py          # report newer releases of every dataset
uv run python scripts/update_data_sources.py --write  # also update sources.json
```

`--write` only pins a release whose download URL exists. The script exits with status 1 when updates are available but not written, so it can also run in CI.

### Progress output

Commands report the major steps: preparing datasets, reading/sampling boundaries, timezone and optional population overlays, optimization, and writing results/maps. Input preparation shows cache reuse or download decisions; subdivision datasets show **X/Y** and the current country. Sampling, population/timezone overlays, and optimization each use **one updating terminal line** with completed/total territories, remaining count, the most recently finished territory's ID/name, elapsed time, and ETA. Parallel completions update the counter without changing result order. Map exports use the same style to count completed rendering jobs.

Downloads and archive extraction use the same updating-line style, showing transferred bytes, total size when available, speed, elapsed time, and estimated time remaining. Population conversion uses it for raster rows processed.

Progress works without configuring separate extras. Output goes to stderr; when redirected to a file, all counters emit only their initial/final status instead of terminal control characters or per-item lines. Major-step logs remain visible. ETA is unavailable until some work has completed and is approximate, particularly when territory sizes differ.

### Population weighting

```powershell
uv run timezone_experiments run --config examples\population.json
```

This config enables population coverage; there is no need to obtain or pass a population CSV yourself. Its first download is about **461 MiB**, and extraction/conversion needs several GiB of temporary disk space. The converted CSV is reused, so the archive is downloaded and converted only once per cell size.

The source is **[GHSL GHS-POP R2023A](https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php), epoch 2025**, a projection distributed as a WGS84 30-arcsecond raster. Population epoch stays 2025 even if the configured solar year changes.

By default, native pixel counts are summed into **1-degree cells**, omitting no-data and zero-population cells. Set `"population_cell_degrees"` to `0.5`, `0.25`, or `0.1` for finer cells. Finer grids increase CSV size and experiment runtime considerably. Conversion processes raster strips rather than loading the full raster into memory.

Aggregation preserves the raster's population total, but these are coarse cell-center samples: cells may be assigned to the wrong side of coastlines or borders, and small islands may be missed. Cells retain the native raster's grid alignment. The CSV's `.source.json` file records the source, the aggregation, and the citations.

### Current-timezone baseline

Every run downloads or reuses the pinned [timezone-boundary-builder](https://github.com/evansiroky/timezone-boundary-builder/releases) release of the current (`timezones-now`) land-only polygons. The archive is large. A custom `--timezones PATH` file can replace the automatic dataset.

The baseline is the area-majority zone for each territory, not every existing zone within a country. The "now" dataset groups zones with equivalent current/future clocks under representative IANA names; it does not preserve every local name or historical borders for the configured solar year.

Only the area samples take part in this majority; population cells do not influence it. Ties go to the zone found first in the territory's sample order, and where timezone polygons overlap, the one listed first in the file wins. If no sample matches, the input territory's timezone attribute is retained, if present.

## All operations use one command

| Command | Purpose |
| --- | --- |
| `timezone_experiments run` | Prepare inputs, optimize territories, and write results. |
| `timezone_experiments maps` | Regenerate maps from saved results without rerunning calculations. |
| `timezone_experiments download` | Prepare/cache configured inputs without running calculations. |
| `timezone_experiments clean` | Remove selected downloads, caches, and outputs. |
| `timezone_experiments dashboard` | Launch the interactive dashboard. |

`run`, `maps`, `download`, and `dashboard` accept `--config` and `--data-directory` (the data root, default `data`), listed first in every `--help`. Omitting `--config` uses built-in defaults. `run` and `dashboard` share an **Optional input overrides** group (`--population` and `--timezones`), listed last. Which countries and administrative levels are used comes only from the configuration (`countries`, `default_admin_level`, `subdivision_rules`, `merged_subdivision_levels`). Options with a default show it in their help text. Errors print a short message; add `--debug` before the command (`timezone_experiments --debug run`) for the full traceback.

```powershell
uv run timezone_experiments --help
uv run timezone_experiments run --help
uv run timezone_experiments download --config examples\population.json
```

`download` prepares exactly what a `run` with the same configuration needs: administrative boundaries and timezones always, and the population CSV (at `population_cell_degrees`) when `coverage_type` is `"population"`. Add `--refresh` to download them again.

### Maps and dashboard

```powershell
uv run timezone_experiments run --maps
uv run timezone_experiments run --maps --maps-dir outputs\other-maps
uv run timezone_experiments dashboard
uv run timezone_experiments dashboard --config examples\population.json
```

The dashboard initializes controls from the configuration and has a control for every configuration setting except `rasterize_static_map_fills`, which only affects static PNG/SVG exports; each control's help names its configuration key. It uses the same automatic input preparation as the CLI when you click **Run experiment**. No uploads are required; your own population and timezone files can be uploaded under the optional input section. It shows a choropleth and results table and offers CSV download. Use `--port` to change the default local port 8501, or `--headless` to avoid opening a browser.

`run` writes only the result CSV (`--results`, default `outputs/csv/results.csv`) unless you add `--maps`, which also exports maps to `--maps-dir` (default `outputs/maps`); `--maps-dir` without `--maps` is an error. The maps are PNG/SVG and interactive HTML maps for the proposed UTC offset, the offset shift from the current timezone, median solar noon, earliest/latest sunrise, and earliest/latest sunset within territories. Each static map is saved as both PNG and SVG, so the seven statistics produce twenty-one files. HTML maps share one `plotly.min.js` written beside them, so keep it with the HTML files when moving them. Exports run in parallel up to `workers`; more workers need more memory.

To iterate on maps after an experiment, reuse its saved results:

```powershell
uv run timezone_experiments maps
uv run timezone_experiments maps --results outputs\csv\na-results.csv --maps-dir outputs\na-maps
```

`maps` uses the same `--results` and `--maps-dir` options as `run`: it reads `outputs/csv/results.csv` by default and writes the same twenty-one files under `outputs/maps/`, replacing existing maps there. CSVs do not contain geometry, so administrative boundaries are still needed: matching cached boundaries are reused, or downloaded if missing. Without `--config`, `maps` uses the configuration `run` saved beside the CSV (`results.config.json` for `results.csv`), so you do not need to repeat it. A `--config` may change map settings (such as `event_map_cell_degrees` or `map_geometry_simplification_degrees`), but its territory settings (`countries`, `default_admin_level`, `subdivision_rules`, `merged_subdivision_levels`, `include_antarctica`) must match the saved ones; otherwise `maps` stops before preparing boundaries and lists the differing settings. Noon, coverage, and offset settings never affect maps, because their results come from the CSV. A CSV without a saved configuration (for example a dashboard download) uses `--config` or the built-in defaults. Every result territory must match boundary geometry; mismatches are reported rather than silently omitted. Extra boundary territories are not drawn.

Maps are not rendered again when nothing changed: repeating `maps` or `run --maps` with the same results and settings reuses the existing files, and intermediate map geometry is cached for other changes (see [Caches](#caches)).

This command loads geometry and saved assignment values: **no optimization sampling, population processing, timezone overlay, or optimization**. Population, current-timezone descriptions, noon ranges, and the annual median equation of time come directly from the CSV, even if the configuration enables those inputs. Median-noon colors apply the saved median to the displayed longitude without recalculating noon astronomy. Sunrise/sunset maps calculate their own annual events on a separate display grid, using the saved experiment `solar_year` and proposed offsets. The `workers` setting controls the same parallel export as `run --maps`.

The proposed-offset map uses discrete colors and legend entries only for observed values. The timezone-shift map instead uses a continuous **blue-white-orange** gradient: negative shifts are blue, zero is exactly white, and positive shifts are orange. Color intensity is proportional to the shift in hours, including fractional hours. Its scale is symmetric around zero, extending to the largest absolute observed shift on both sides, so equal-magnitude advances and delays have comparable intensity. An all-zero dataset uses a neutral white fill with a -1 to +1 hour scale. Missing values stay light grey, and hover text shows the actual offset or shift. Real extreme shifts remain visible; they are not clipped. Both static and interactive maps use the same palette and limits.

The **`median_solar_noon`** map shows the annual median of daily solar-noon times at each longitude, under the territory's **proposed permanent UTC offset**. It is not a single territory-wide average; it uses the middle annual noon value independently for every longitude. A median is useful here because at least half of days fall on either side of it. Noon becomes four minutes earlier per degree east; latitude does not affect this model. Colors use a shared continuous blue-white-orange scale: **12:00 is white**, earlier noon blue, and later noon orange. The scale fits the displayed medians, rounding outward to full hours, but its limits stay within **09:00-15:00** (noon +/-3 hours). For example, 11:10-15:09 uses 11:00-15:00. Values outside the color limits receive the endpoint colors; actual calculations, CSV values, and hover times are not clamped. Both limits are labeled on the legend. When the limits span 12:00, each side reaches its strongest color at its own limit; 12:00 need not sit at the middle of the legend. Entirely earlier/later ranges use only the corresponding blue/orange palette. Constant or entirely out-of-range data gets a nonzero whole-hour display range within 09:00-15:00. Clock labels retain signed day suffixes when necessary. Equal times have equal colors across territories in the same map; a color jump at a boundary reflects different proposed offsets, not per-territory normalization. Current standard/DST gradients and min/max variants are not included.

The noon renderer clips 0.25-degree longitude bands to the full territory geometry, retaining holes and islands and splitting at the dateline. Each band is colored by its longitudinal midpoint, limiting the color-value approximation to **30 seconds**; band edges have no strokes. Hover shows the approximate median time for that band, rounded to minutes, alongside the existing territory details and annual extremes. Static and interactive maps use identical bands and scales.

The **`earliest_sunrise`**, **`latest_sunrise`**, **`earliest_sunset`**, and **`latest_sunset`** maps show annual event extremes at individual locations, not annual averages or a single territory-wide extreme. Each uses the territory's proposed permanent offset without DST, independently of solar-noon optimization settings. Events are evaluated daily over the saved experiment `solar_year`. Latitude controls seasonal day length; longitude shifts clock times. A two-dimensional 0.5-degree display grid is clipped to the complete territory, preserving holes, islands, and dateline splits. Each cell samples a point inside its clipped geometry; colors are approximate, especially near polar transition dates, and are not minute-accuracy guarantees across the cell.

The sunrise pair shares a continuous purple-to-yellow clock scale; the sunset pair shares another. Scales fit the observed pair's extrema, round outward to full hours, and are not subject to the noon map's 09:00-15:00 clamp. Static and interactive maps use identical cells, values, and scales. Hover includes the sampled coordinates, extreme's solar-cycle date, and counts of polar-day and polar-night days. Days without a horizon crossing are excluded rather than assigned midnight or noon. Locations without any modeled events are grey and explicitly not applicable, including in interactive hover. Signed day suffixes preserve events before or after their solar-cycle date; times are not wrapped before taking annual extrema.

Sunrise/sunset uses NOAA's approximate solar declination and equation of time with the conventional 90.833-degree zenith (solar radius plus standard atmospheric refraction). The horizon is flat, with no terrain, elevation, or weather correction. These are comparative estimates, not navigation-grade predictions; high-latitude transition dates are particularly sensitive.

Interactive hover text shows the proposed offset and the **actual timezone**, including its standard and DST UTC offsets when an IANA identifier is available. These details use the experiment's `year`, saved in `noon_year` and shown as `Actual (YEAR)`. Regenerating maps preserves the saved statistics, even with updated timezone rules. Rules are checked throughout the saved year, including short DST suspensions, fractional-hour changes, and negative-DST adjustments. Timezone data comes from the automatic current-timezone dataset or an explicit `--timezones` file. Missing timezone data is labeled `Not available`; fixed-offset labels show the offset with `DST information unavailable`, since a fixed offset does not identify seasonal rules. The baseline still represents the territory's area-majority zone, not every zone within it.

Hover text also shows **annual noon extremes across the whole territory**: the earliest annual noon at the eastern boundary and latest annual noon at the western boundary. These are independent of the optimization metric and coverage settings. Proposed and current-baseline ranges use the displayed saved year. Current ranges are separated by standard/DST state and UTC offset, using the timezone rules at each endpoint's solar-noon instant; only dates in each state contribute to its range. The current baseline applies the territory's representative timezone throughout, not each location's own timezone. Dateline coordinates are unwrapped around the proposed offset's meridian; times outside the reference day carry a signed day suffix rather than silently wrapping. Times are rounded to the nearest minute for display, while the CSV retains unrounded minutes.

Interactive maps normalize polygon winding for Plotly and skip zero-area components, preventing reversed polygons from coloring the ocean and covering other territories' borders. The initial view shows the full world instead of automatically rotating/zooming to fit the data. This affects rendering only: source boundaries, calculations, and saved results are unchanged.

Country outlines are dark, thicker solid lines; internal territory divisions are thin, light-grey solid lines. Country outlines are dissolved from the mapped territories grouped by country, rather than overlaying a different boundary dataset. Interior holes created by mismatched subdivision seams are excluded from the thick outlines; genuine holes supported entirely by the source interior-ring edges are retained, as are islands. This changes outlines only, not territory fills or calculations. Dateline-crossing outlines are joined before dissolving and split before display to avoid cross-map streaks. Dotted and dashed lines remain reserved for the solar-time guides.

All maps, including the dashboard, overlay dotted solar-time meridians every 15° with UTC labels, and dashed ideal zone boundaries 7.5° to either side of each meridian. These are geographic reference guides, not current political timezone borders. Labels run from UTC-12 to UTC+12; UTC+13 and UTC+14 share meridians with UTC-11 and UTC-10 on the next calendar day.

### Cleaning generated data

Use `clean` to remove selected runtime data without deleting source/configuration files:

```powershell
uv run timezone_experiments clean --csv
uv run timezone_experiments clean --maps --dry-run
uv run timezone_experiments clean --downloads --cache
uv run timezone_experiments clean --csv --maps
uv run timezone_experiments clean --all
```

Available targets are `--downloads` (`DATA_DIRECTORY/downloads`), `--cache` (`DATA_DIRECTORY/cache`), `--csv` (`OUTPUT_DIRECTORY/csv`), `--maps` (`OUTPUT_DIRECTORY/maps`), and `--all` (downloads, cache, and the whole output directory). Targets can be combined. `--data-directory` (default `data`) and `--output-directory` (default `outputs`) choose the two roots. `--dry-run` prints the selected paths without deleting.

## Selected countries

To experiment with a few countries instead of the whole world, list them in `countries`:

```powershell
uv run timezone_experiments run --config examples\north_america.json --results outputs\csv\na-results.csv
```

Only those countries are downloaded (at their configured levels), optimized, and mapped; `subdivision_rules` and `merged_subdivision_levels` entries for other countries are ignored. Administrative boundaries always come from the automatic Natural Earth/GADM/GeoBoundaries preparation, so results stay comparable across runs.

## Advanced: bring your own population or timezone files

Population and timezone paths are optional overrides, not required setup:

- `--timezones PATH`: custom current-timezone GeoJSON with IANA identifiers, replacing the automatic dataset.
- `--population PATH`: population CSV. Supplying a CSV alone does not switch coverage to population; set that in the configuration.

Custom population CSVs must contain WGS84 cell centers and non-negative **counts, not densities**; omit no-data cells:

```csv
latitude,longitude,population
51.5,-0.1,100
```

The built-in raster converter supports GHSL, not arbitrary provider formats. For territories without population samples, the calculation engine warns and reports `coverage_type_used=area`; automatic preparation does not silently skip failed population downloads.

## Experiment settings

Copy [examples/default.json](examples/default.json) or supply only settings you want to override. Omitted settings retain their defaults; unknown setting names are rejected. The example intentionally omits `workers` to retain automatic CPU detection; add `"workers": 4` if you want to pin it to four processes.

This section is the user-facing configuration reference. The example stays **standard JSON**, so you can run it directly: comments and extra keys such as `_description` are not supported. Defaults and validation are defined in [config.py](src/timezone_experiments/core/config.py).

```json
{
  "minimum_noon": "12:00",
  "maximum_noon": "13:00",
  "solar_metric": "min",
  "coverage_type": "population",
  "coverage_threshold": 0.95,
  "countries": ["CAN", "MEX", "USA"],
  "subdivision_rules": {"USA": 1, "CAN": 1, "MEX": 0}
}
```

Settings are grouped below in the same order as in [examples/default.json](examples/default.json): the solar-noon rule, coverage, candidate offsets, territories, accuracy and performance, and map exports.

### How the settings work together

For each country or subdivision, the optimizer:

1. Orders `utc_offsets` by preference: with `minimum_noon`, by mean solar noon at the territory's eastern edge, earliest first; with only `maximum_noon`, by noon at its western edge, latest first; with neither, by how close noon at its center is to 12:00. Offsets a whole day apart are ordered by closeness to the territory's meridian.
2. Calculates daily local solar-noon times for `year` at the sampled locations, then reduces each location's year to one value using `solar_metric`.
3. Checks those values against `minimum_noon` and `maximum_noon`. The fraction of the territory passing, measured using `coverage_type`, must reach `coverage_threshold`.
4. Stops at the first feasible offset; with the default minimum bound this is the **earliest feasible noon**. It does not reward later offsets for covering additional people or land once the threshold is met. If no offset is feasible, it uses the one with the highest coverage and sets `fallback_used` (see below).

**Time and space are separate:** `solar_metric` summarizes days at each location; `coverage_type` and `coverage_threshold` measure how much of a territory passes. For example, `"solar_metric": "min"`, `"minimum_noon": "12:00"`, and `"coverage_threshold": 0.95` require the earliest annual noon to be at least 12:00 across 95% of the chosen spatial weight, not merely on 95% of days.

### Solar-noon rule

Solar noon means the time the Sun crosses the local meridian, expressed on the proposed local clock, not necessarily 12:00. Clock settings use `"HH:MM"` in 24-hour time.

| Setting | Default | Meaning and accepted values |
| --- | --- | --- |
| `minimum_noon` | `"12:00"` | Inclusive lower bound on the selected noon statistic: noon must be at or after this time. Set `null` to remove the lower bound. |
| `maximum_noon` | `null` | Inclusive upper bound on the selected noon statistic: noon must be at or before this time. `null` means no upper bound. When both bounds are present, this must not be earlier than `minimum_noon`. |
| `solar_metric` | `"median"` | Statistic applied to a location's daily noon times over the year. Accepted values: `"min"` = earliest noon, `"max"` = latest, `"median"` = middle value (average of the two middle values for an even number of days). |
| `year` | `2026` | Integer reference year for daily solar calculations and winter comparison dates, including leap-year handling. Does not select a historical boundary release or change the population dataset's epoch. |

Bounds apply to the **chosen statistic**, not independently to every day. For example, with `"min"`, the lower bound controls the earliest noon, but the upper bound also tests that earliest noon; it does not cap the latest noon. Daily noon values are reduced to the clock range from 00:00 up to, but not including, 24:00 before calculating the statistic.

### Coverage

| Setting | Default | Meaning and accepted values |
| --- | --- | --- |
| `coverage_type` | `"area"` | How passing locations are weighted: `"longitude"` weights sampled longitude columns rather than land area; `"area"` approximates land-area coverage; `"population"` weights by population counts. |
| `coverage_threshold` | `0.9` | Minimum passing fraction from `0` to `1`, inclusive. The default permits 10% of the selected weight to fail; `1.0` requires all positive-weight samples to pass. `0` removes the coverage restriction. |

Coverage is estimated from samples, not an exact geometric or census guarantee. If population coverage has no usable population weights for a territory, the engine warns and uses area coverage; the result records the actual method in `coverage_type_used`.

The CLI and dashboard prepare population data automatically when `coverage_type` is `"population"`.

### Choosing between feasible offsets

| Setting | Default | Meaning and accepted values |
| --- | --- | --- |
| `utc_offsets` | Whole hours `-12` through `14` | Non-empty list of candidate offsets in hours relative to UTC. Every value must be between `-14` and `14`, inclusive. Decimal hours are allowed: `[5, 5.5, 5.75, 6]` includes UTC+05:30 and UTC+05:45. Only listed offsets are considered; they are permanent offsets without seasonal daylight-saving changes. |

With a minimum bound, the optimizer chooses the **earliest feasible noon**: it orders candidates by nominal noon at the easternmost analyzed longitude and stops at the first candidate satisfying the configured metric, bounds, coverage type, and threshold. Longitudes are unwrapped into one shared territory frame first, so dateline islands are ordered coherently. Using the eastern edge, rather than a land-area center, avoids mistaking late population noons for early ones when a candidate crosses midnight at the center. Candidate-list and sample order do not affect the result. For ordinary daytime noons this selects the smaller feasible UTC offset: if UTC+1 already passes, UTC+2 is not chosen merely for greater coverage or better land-area alignment. With only `maximum_noon`, it chooses the latest feasible noon instead, and with neither bound, the offset whose noon at the territory's center is closest to 12:00.

If no candidate reaches the configured coverage threshold, the optimizer prints a warning and uses the configured offset with the **highest achieved coverage**. Equal-coverage candidates keep their preference order, so the fallback is deterministic. This best-effort assignment does not satisfy the configured restriction; the CSV, dashboard, and map hover expose it with `fallback_used = true` and show the actual compliance rate. Normal assignments use the first feasible candidate and have `fallback_used = false`.

There is no additional area-weighted optimization score or `objective_score` output column. The reported `area_weighted_noon_minutes` remains an area-weighted summary of the chosen offset's per-location statistics; it does not select the offset.

Whenever the current offset is known, territories use a coherent longitude frame centered on the full territory's land area and shifted by whole rotations toward the current offset's meridian. Without a current offset, date-line-crossing territories use their land-area center; other territories use the first analyzed sample.

Increasing `coverage_threshold` makes feasibility stricter without changing the earliest-feasible selection rule. With only a minimum noon constraint and ordinary, non-wrapping noon times, 100% longitude coverage is governed by the easternmost positive-weight samples; a 50% threshold instead tests roughly the western half of the longitude weight. Area and population coverage apply the same fraction to their respective weights. These are hypothetical solar-rule assignments, not a reconstruction of current civil zones.

Clock-equivalent offsets can differ by a full calendar day: **UTC-11 and UTC+13 have the same clock reading**. The civil date line does not follow ±180°, so this matters even when a territory's geometry does not cross that meridian. Anchoring to the current offset resolves Tonga and Samoa to **UTC+13**, Wallis and Futuna to **UTC+13**, and Kiribati's Line Islands to **UTC+14**, rather than their negative-offset aliases. Using the full territory's land-area center also prevents small outlying islands from determining a date-line-crossing territory's calendar date. This changes neither clock readings nor noon compliance for day-equivalent offsets, imposes no hard shift cap, and still considers only the configured candidate offsets. Current-timezone data is always loaded to provide this civil-calendar reference.

### Territories

| Setting | Default | Meaning and accepted values |
| --- | --- | --- |
| `countries` | `null` | Three-letter ISO codes of the countries to include, such as `["CAN", "MEX", "USA"]`; `null` includes every country. Codes are case-insensitive and must exist in the world dataset. Only the listed countries are downloaded, optimized, and mapped. Including `ATA` requires `include_antarctica`. |
| `default_admin_level` | `0` | Administrative level used for countries without an override: `0` = one offset per country/territory, `1` = states/provinces, `2` = districts, and so on. Must be a non-negative integer. Higher levels require more detailed boundaries and produce more assignments; availability depends on the country. |
| `subdivision_rules` | Geographically spread countries/territories (below) | Per-country assignment overrides. By default, the 13 countries and territories below use first-level subdivisions. An explicitly supplied mapping **replaces**, rather than merges with, these defaults: `{"USA": 1}` subdivides only the USA. `{}` disables these subdivision overrides, but does not disable `merged_subdivision_levels`. Country codes are case-insensitive; levels are non-negative integers. This selects administrative units, not existing timezone borders. |
| `merged_subdivision_levels` | `{"FRA": 1, "NLD": 1}` | Per-country source levels for automatically prepared boundaries. Combine that country's finer geometry into **one country-level assignment**, retaining its original name and identity and hiding internal subdivision edges. These entries take precedence over `default_admin_level`; positive `subdivision_rules` for the same country are rejected. Keys are case-insensitive and levels are positive integers. An explicit mapping replaces these defaults; `{}` disables the overrides. All calculations, population/timezone overlays, noon statistics, and maps use the retained source footprint. This selects a dataset, not an automatic overseas filter. |
| `include_antarctica` | `false` | Include Antarctica in automatically prepared world boundaries. Its longitude span makes the default country-level noon constraint infeasible; relax the constraint if enabling this. |

The default subdivision rules come directly from `MAJOR_MULTIZONE_COUNTRIES`: **USA, CAN, RUS, CHN, AUS, BRA, IND, IDN, MEX, KAZ, MNG, KIR, GRL**, each at level `1`: the United States, Canada, Russia, China, Australia, Brazil, India, Indonesia, Mexico, Kazakhstan, Mongolia, Kiribati, and Greenland. These have wide landmass longitude spans, geographically separated territories, or island groups spanning many longitudes. France and the Netherlands instead use `merged_subdivision_levels`, with one assignment per country. Other countries use one country-level assignment by default. [examples/default.json](examples/default.json) spells out both mappings.

Boundary-source limitation: the current GADM level-1 datasets for France and the Netherlands omit their overseas territories; these are intentionally outside the calculation and map footprint and are not retained as separate assignments. Metropolitan France includes Corsica. Source exclusions are logged during assembly. Greenland's level-1 subdivisions still leave some median-noon outliers. Regenerate results after changing either boundary mapping; existing CSVs and maps are not updated automatically.

The dashboard labels `default_admin_level` as **Default administrative level**. For example, with fallback level `0` and `{"USA": 1}`, the USA uses state-level assignments and Argentina uses one country-level assignment. Setting the fallback to `2` would give Argentina district-level assignments, but the explicit USA override would remain `1` and France and the Netherlands would remain single countries under their default `merged_subdivision_levels`.

To reproduce a country-only run, use:

```json
{
  "default_admin_level": 0,
  "subdivision_rules": {},
  "merged_subdivision_levels": {}
}
```

### Accuracy and performance

| Setting | Default | Meaning and accepted values |
| --- | --- | --- |
| `sample_grid_size` | `12` | Positive integer giving the number of grid rows **and** columns per polygon bounding box. `12` gives up to 144 grid cell centers per polygon; centers outside it are discarded, and a polygon too small to contain any gets one interior point. Multipart territories can therefore have many more than 144 samples. Higher values refine spatial estimates but increase work roughly quadratically in the interior grid; they do not improve the source boundaries or population CSV resolution. |
| `population_cell_degrees` | `1.0` | Width and height, in degrees, of the aggregated GHSL population cells. Allowed values: `1.0`, `0.5`, `0.25`, `0.1`. Smaller cells improve spatial detail but increase CSV size and runtime. Used only when preparing population data; does not resample a custom CSV. The source population epoch remains 2025 regardless of `year`. |
| `workers` | Available CPU threads | Omit to use `os.process_cpu_count()`, which reports the logical CPUs available to the Python process; falls back to `1` if detection is unavailable and is capped at Python's process-pool limit of 61 on Windows. Set a positive integer to override, for example `4`, or `1` for sequential execution. It applies to boundary sampling, the timezone and population overlays, population raster conversion, and map exports (see [Parallelism](#parallelism)). Optimization, downloads, and file parsing are always serial. |

### Map exports

| Setting | Default | Meaning and accepted values |
| --- | --- | --- |
| `event_map_cell_degrees` | `0.5` | Width/height of the sunrise/sunset gradient cells, in degrees. Allowed values: `0.25`, `0.5`, `1`, `2`. This affects map generation only, not CSV optimization or coverage. Smaller cells are more detailed and slower; `1` is a good choice for quick drafts. |
| `map_geometry_simplification_degrees` | `0.01` | Topology-preserving simplification applied only to derived gradient-map geometry. Allowed values: `0`, `0.005`, `0.01`, `0.02`. `0` preserves exact source geometry; `0.01°` is the validated default and falls back to exact geometry whenever component topology or the 0.1% area-error limit would be exceeded. Country/territory outline overlays and CSV calculations remain unsimplified. |
| `rasterize_static_map_fills` | `false` | Rasterize all choropleth fill layers in static PNG/SVG exports, while retaining borders, guides, legends, and labels as vector elements. PNG appearance is unchanged; SVGs gain a bitmap image layer and can become much smaller. Interactive HTML maps are unchanged. |

## Outputs and limitations

The result CSV includes `territory_id`, `country_code`, `territory_name`, administrative level, current/proposed timezone, `proposed_utc_offset_hours`, offset shift, `area_weighted_noon_minutes` (the area-weighted mean of the chosen offset's per-location noon statistic), compliance, population coverage, coverage method, and whether the below-threshold best-coverage fallback was used.

Both CLI result files and dashboard CSV downloads persist the complete noon-hover data, calculated once after choosing each territory's offset:

| Column | Contents |
| --- | --- |
| `solar_year` | Configured experiment year, persisted for reproducible annual sunrise/sunset maps. |
| `noon_year` | Year of the descriptive noon statistics and current timezone rules (the configured `year`). |
| `proposed_earliest_noon_minutes` | Annual earliest noon at the eastern boundary, in unrounded minutes relative to midnight. |
| `proposed_latest_noon_minutes` | Annual latest noon at the western boundary, in the same units. |
| `current_timezone_description` | Saved timezone description, including standard/DST offsets or an explicit unavailable label. |
| `current_noon_ranges` | JSON object mapping each current standard/DST offset label to an `[earliest, latest]` minute pair. `{}` means no current timezone data. Multiple offset regimes are retained rather than collapsed into a single standard/DST pair. |
| `median_equation_of_time_minutes` | Unrounded annual median equation of time for `noon_year`, used to render median noon as `720 - 4 * longitude - median_equation_of_time_minutes + 60 * proposed_utc_offset_hours`. Longitude uses the proposed-offset meridian's dateline frame. |

Minute values may fall outside 0-1440 to retain the civil-day displacement. Noon map generation formats saved ranges and applies the saved median equation of time to longitude; it does not recalculate noon astronomy or query timezone rules. Sunrise/sunset gradients evaluate their own annual events using the persisted `solar_year`. CSVs missing these columns are rejected; rerun the experiment to create them.

Solar-noon calculations use NOAA's equation-of-time approximation (a Fourier series after Spencer, 1971), which is accurate to about **half a minute**, evaluated once per day at 12:00. They suit comparative experiments, not navigation or legal timekeeping.

Geometry sampling uses a regular longitude/latitude grid. Each sample's area weight is its cell's angular area times the cosine of the cell center's latitude: an equal-angle approximation of spherical area that is close for small cells and less so for tall cells far from the equator. Increase `sample_grid_size` to refine it. Population accuracy is limited by the CSV resolution.

Polygon rings, holes, multipolygons, and coordinate wrapping are handled, but this sampler is not a geodesic GIS engine. Natural Earth's generalized boundaries and GADM's detailed boundaries may differ, including on disputed borders and small islands. Setting a global administrative level above 0 can require many large downloads, and not every country has every level.

Some source polygons collapse to lines or points and enclose zero area. These degenerate components are skipped with a warning while valid components of the same territory are retained. A territory with no non-zero-area components still raises an error identifying the territory; it is not silently dropped.

### Data licensing and attribution

- Maps and the dashboard show a data credits line for the datasets behind each result.
- Natural Earth is [public domain](https://www.naturalearthdata.com/about/terms-of-use/).
- Automatically downloaded subdivisions use [GADM 4.1](https://gadm.org/license.html) or [GeoBoundaries](https://www.geoboundaries.org/) when unavailable from GADM. Review each dataset's license and attribution requirements: GADM restricts commercial use and redistribution; Kiribati's GeoBoundaries data is OpenStreetMap-derived under ODbL 1.0.
- Timezone-boundary-builder uses ODbL; retain upstream attribution and comply with its redistribution requirements.
- GHSL is CC BY 4.0, European Union. Retain the CSV's `.source.json` file, indicate aggregation, and cite the [dataset](https://doi.org/10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE) and [methodology](https://doi.org/10.1080/17538947.2024.2390454). See the [citation requirements](https://human-settlement.emergency.copernicus.eu/GHSLhowToCite.php).

## Development

```powershell
uv run pytest                     # full suite with coverage; add -m "not slow" to skip end-to-end tests
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pre-commit run --all-files
```

The package is layered: `core` (shared models, configuration, geometry) at the bottom, then `astronomy`, then `data` and `optimization` (which do not import each other), then `experiments`, `maps`, and `app` (CLI and dashboard) on top. A lower layer never imports a higher one; `tests/test_layering.py` enforces this.

### Caches

Prepared inputs are cached under `data/cache/prepared/` in three layers, each one SQLite file named by a digest of everything it depends on:

- `base`: each territory's compressed geometry and sample grid, keyed by the boundary file's content, the territory and sampling settings, and the source of the code that builds them.
- `timezones`: each territory's representative timezone, keyed by the base layer and the timezone file.
- `population`: only the population cells appended after each territory's base samples, keyed by the base layer and the population CSV.

Samples are stored as compressed native-endian float64 rows, so the byte order is recorded and checked. Solar, coverage, offset, worker, and map settings are deliberately not part of any key. Corrupt or incompatible files are logged and rebuilt.

Map exports have two more caches. A hidden `.map-render-cache.json` manifest in the output directory fingerprints the territory geometry and results, the requested maps, the rendering settings and source, and the dependency and Python versions; on an exact repeat, the existing files are checksummed and reused. Otherwise, intermediate map geometry is cached under `data/cache/map-features/`: borders depend only on boundary geometry, median-noon bands also on the proposed offsets and the saved median equation of time, and sunrise/sunset cells on the proposed offsets and solar year. Changing compliance, fallback, or hover-only values therefore reuses them.

### Parallelism

`workers` caps each stage, and no stage uses more workers than it has independent jobs:

- Boundary sampling uses processes across territories, but only for large workloads.
- The timezone and population overlays use threads, since shapely's point-in-polygon tests release the GIL.
- Population raster conversion uses threads, each with its own GDAL reader.
- Map exports use processes across the static and HTML rendering jobs, sending each only the mapped geometry and result values.

Optimization is serial: it is vectorized, and a process pool spent more time pickling territories than it saved. Each year's noon statistics come from one table of the equation of time sorted by phase, so no stage builds the daily clock times for every sample and offset.

### Performance

On a 20-thread machine with the world GADM boundaries (599 territories, about 5.4 million samples), input processing and optimization take roughly: territory sampling 11 s, timezone index 5 s, timezone overlay 5 s, population overlay 10 s, and about 7 s to optimize every territory (serial; the vectorized optimizer is faster without a process pool). Prepared inputs are cached, so repeated runs with the same inputs skip everything except optimization.
