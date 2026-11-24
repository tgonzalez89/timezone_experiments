# Methodology

How results are computed, what the maps show, and the known limitations. Configuration options are listed in the [README](../README.md#configuration).

## Territories and samples

Countries come from Natural Earth (1:50m). Countries with a positive level in `subdivision_rules` or `default_admin_level` are replaced by GADM subdivisions at that level; when GADM has no file for a country and level (HTTP 404), GeoBoundaries is used at the same level. Countries in `merged_subdivision_levels` are loaded at the finer level and merged back into one territory, so areas absent from that dataset (for example, overseas territories) are excluded.

Each polygon is sampled on a `sample_grid_size` x `sample_grid_size` grid over its bounding box; cell centers outside the polygon are discarded, and a polygon too small to contain any center gets one interior point. Each sample carries three weights:

- **Area:** the cell's angular area times the cosine of the cell center's latitude. This equal-angle approximation of spherical area is close for small cells and less so for tall cells far from the equator.
- **Longitude:** each grid column's width, shared evenly by its inside cells, so coverage measures the share of the territory's longitude span.
- **Population:** with population coverage, every population cell whose center lies in the territory becomes an extra sample weighted by its count. A cell inside overlapping territories counts for each.

Coverage is estimated from these samples. It is not an exact geometric or census measure.

## Solar noon

Solar noon is the time the Sun crosses the local meridian, expressed on the proposed clock. For a location at longitude λ (degrees east) on day *d*:

```text
noon(d) = 720 - 4 λ - EoT(d) + 60 × offset      (minutes after midnight)
```

720 is 12:00 UTC, the Sun crosses 1° of longitude every 4 minutes, and `EoT` is the equation of time from NOAA's "General Solar Position Calculations" (a Fourier series after Spencer, 1971), accurate to about half a minute and evaluated at 12:00 of each day. Latitude does not affect solar noon.

Each location's daily values for `year` are reduced to one value with `solar_metric` (`min`, `max`, or `median`, the median being the mean of the two middle values for an even number of days). Daily values are taken modulo 24 hours before the statistic.

## Choosing an offset

For each territory:

1. **Order the candidates** in `utc_offsets` by preference, using mean solar noon (without the equation of time):
   - with `minimum_noon`: earliest noon at the territory's eastern edge first;
   - with only `maximum_noon`: latest noon at the western edge first;
   - with neither: noon at the territory's center closest to 12:00 first.

   Offsets 24 hours apart (for example UTC-10 and UTC+14) show the same clock time; the one closer to the territory's meridian comes first.
2. **Evaluate** each candidate in that order: compute every sample's noon statistic and check it against `minimum_noon` and `maximum_noon` (both inclusive).
3. **Choose** the first candidate whose passing share of the `coverage_type` weight reaches `coverage_threshold`. With the default minimum bound this is the earliest feasible noon; later offsets are not preferred for covering more land or people.
4. **Fall back** when no candidate reaches the threshold: the candidate with the highest coverage is used (the first in preference order among equal ones), a warning is logged, and `fallback_used` is `true`.

If population coverage is configured but a territory has no usable population weights, area coverage is used and `coverage_type_used` records it.

### Date-line frame

Longitudes are measured in one continuous frame per territory, so territories crossing ±180° are ordered coherently. The frame is centered on the territory's area-weighted center and, when the current timezone is known, shifted by whole turns toward the current offset's meridian. This places Tonga, Samoa, and Wallis and Futuna at UTC+13 and Kiribati's Line Islands at UTC+14 rather than at their negative-offset aliases. It changes neither clock readings nor noon compliance.

## Current-timezone baseline

The current timezone of a territory is the timezone-boundary-builder zone (`timezones-now` release) covering the largest share of its area samples. Population cells do not take part. Ties go to the zone found first in the territory's sample order; where timezone polygons overlap, the one listed first in the file wins. The "now" dataset groups zones with equivalent current and future clocks under representative IANA names, so it does not preserve every local zone or historical borders.

The current standard offset is read at the local winter solstice (21 December in the northern hemisphere, 21 June in the southern), when daylight saving time is off. `timezone_shift_hours` is the proposed offset minus that standard offset.

## Maps

| Map | Shows |
| --- | --- |
| `proposed_utc_offset_hours` | Proposed offset, one color per observed value. |
| `timezone_shift_hours` | Proposed minus current standard offset. Diverging scale, symmetric around zero (white). |
| `median_solar_noon` | Annual median noon under the proposed offset, per 0.25° longitude band, colored at the band's midpoint (within 30 seconds). 12:00 is white; the scale is fitted to whole hours within 09:00-15:00, and values outside it take the end colors. |
| `earliest_sunrise`, `latest_sunrise`, `earliest_sunset`, `latest_sunset` | Annual extremes on an `event_map_cell_degrees` grid, under the proposed offset without daylight saving time. Each sunrise or sunset pair shares one scale. |

Sunrise and sunset use NOAA's declination series and a zenith of 90.833° (standard refraction plus the solar radius), with a flat horizon and no terrain, elevation, or weather. Days without sunrise or sunset (polar day or night) are excluded and counted in the hover text; locations with no events in the year are grey. These are comparative estimates; dates near polar transitions are the least accurate.

All maps draw dotted solar-time meridians every 15° and dashed ideal zone boundaries 7.5° on either side. Times outside 00:00-24:00 keep a signed day suffix rather than wrapping.

The `maps` command redraws maps from a result CSV: offsets, noon statistics, and timezone descriptions come from the CSV, and only sunrise and sunset are computed again.

## Result CSV

| Column | Contents |
| --- | --- |
| `territory_id`, `country_code`, `territory_name`, `admin_level` | Territory identity. |
| `current_timezone` | Current IANA zone, empty when unknown. |
| `proposed_timezone`, `proposed_utc_offset_hours` | Chosen offset, as a label (`UTC+01:00`) and in hours. |
| `timezone_shift_hours` | Proposed minus current standard offset, empty when the current zone is unknown. |
| `area_weighted_noon_minutes` | Area-weighted mean of the samples' noon statistic under the chosen offset, in minutes after midnight. |
| `compliance_rate` | Passing share of the coverage weight. |
| `population_covered` | Passing share of the territory's population, empty without population data. |
| `coverage_type_used` | Coverage actually used (`area` when population weights were missing). |
| `fallback_used` | `True` when no offset reached the threshold. |
| `solar_year`, `noon_year` | The configured `year`. |
| `proposed_earliest_noon_minutes`, `proposed_latest_noon_minutes` | Annual earliest noon at the eastern boundary and latest noon at the western boundary, unrounded minutes. |
| `current_timezone_description` | Current zone with its standard and daylight-saving offsets in `noon_year`. |
| `current_noon_ranges` | JSON object mapping each current offset label to an `[earliest, latest]` noon pair in minutes; `{}` without a current zone. |
| `median_equation_of_time_minutes` | Annual median equation of time in `noon_year`, used to draw the median-noon map. |

Minute values can fall outside 0-1440 to keep the displacement from the local day.

## Custom input files

- **Population CSV** (`--population`): columns `latitude`, `longitude`, `population`, with WGS84 cell centers and non-negative counts per cell (not densities). Supplying a CSV does not switch to population coverage; set `coverage_type` too.

  ```csv
  latitude,longitude,population
  51.5,-0.1,100
  ```

- **Timezone GeoJSON** (`--timezones`): polygons with an IANA timezone ID in a `tzid` property.

The built-in population download is GHS-POP R2023A, epoch 2025, summed from 30-arcsecond pixels into `population_cell_degrees` cells; no-data and zero-population cells are omitted. Cell centers can fall on the wrong side of coastlines or borders, and small islands can be missed.

## Limitations

- These are hypothetical solar-rule assignments, not a reconstruction of civil timezones.
- Sampling is planar on longitude and latitude, not geodesic. Results depend on `sample_grid_size` and the boundary sources; Natural Earth and GADM differ on disputed borders and small islands.
- The current GADM level-1 files for France and the Netherlands omit overseas territories, which are therefore outside the calculation and maps. Greenland's level-1 subdivisions leave some median-noon outliers.
- Polygon parts with zero area are skipped with a warning; a territory with no part of positive area is an error.
- A global `default_admin_level` above 0 requires many large downloads, and not every country has every level.
