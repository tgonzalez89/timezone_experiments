"""Point lookup against timezone-boundary-builder GeoJSON."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from timezone_experiments.constants import require_positive_workers
from timezone_experiments.data.geometry import PolygonIndex

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from timezone_experiments.data.models import Territory

logger = logging.getLogger(__name__)


class TimezoneBoundaryIndex:
    """Prepared timezone polygons; where polygons overlap, the feature listed first in the file wins."""

    def __init__(self, boundaries: Sequence[tuple[str, dict[str, object]]]) -> None:
        """Parse and prepare every timezone polygon once."""
        if not boundaries:
            msg = "At least one timezone boundary is required."
            raise ValueError(msg)
        logger.info("Preparing spatial index for %d timezone features.", len(boundaries))
        self.zones = tuple(zone for zone, _ in boundaries)
        self._polygons = PolygonIndex([geometry for _, geometry in boundaries])
        logger.info("Timezone spatial index ready (%d polygons).", len(self._polygons))

    def zone_indices(
        self,
        latitudes: NDArray[np.float64],
        longitudes: NDArray[np.float64],
        *,
        workers: int = 1,
        progress_label: str | None = None,
    ) -> NDArray[np.intp]:
        """Return each point's index into ``zones`` (-1 outside every timezone polygon)."""
        return self._polygons.first_owner(longitudes, latitudes, workers=workers, progress_label=progress_label)

    def timezone_at(self, latitude: float, longitude: float) -> str | None:
        """Return the timezone polygon containing a coordinate, if any."""
        (zone,) = self.zone_indices(np.array([latitude]), np.array([longitude])).tolist()
        return None if zone < 0 else self.zones[zone]


def annotate_current_timezones(
    territories: Sequence[Territory],
    index: TimezoneBoundaryIndex,
    *,
    workers: int = 1,
) -> list[Territory]:
    """Assign each territory the timezone covering the largest share of its positive-area samples.

    Territories with no matching sample keep the timezone attribute they already had (if any). Ties go to the
    zone that appears first in the territory's sample order.
    """
    require_positive_workers(workers, "Timezone overlay")
    started = time.perf_counter()
    positive = [[sample for sample in territory.samples if sample.area_weight > 0] for territory in territories]
    counts = [len(samples) for samples in positive]
    total = sum(counts)
    logger.info(
        "Looking up current timezones for %d territories (%s positive-area samples) with %d workers.",
        len(territories),
        f"{total:,}",
        workers,
    )
    latitudes = np.fromiter((sample.latitude for samples in positive for sample in samples), np.float64, total)
    longitudes = np.fromiter((sample.longitude for samples in positive for sample in samples), np.float64, total)
    weights = np.fromiter((sample.area_weight for samples in positive for sample in samples), np.float64, total)
    owners = np.repeat(np.arange(len(territories), dtype=np.intp), counts)
    zones = index.zone_indices(latitudes, longitudes, workers=workers, progress_label="Timezone overlay")
    majority = _majority_zones(owners, zones, weights, len(territories), len(index.zones))
    annotated = [
        territory if zone < 0 else replace(territory, current_timezone=index.zones[zone])
        for territory, zone in zip(territories, majority.tolist(), strict=True)
    ]
    logger.info(
        "Current timezone overlay complete for %d territories in %.2f seconds.",
        len(annotated),
        time.perf_counter() - started,
    )
    return annotated


def _majority_zones(
    owners: NDArray[np.intp],
    zones: NDArray[np.intp],
    weights: NDArray[np.float64],
    territory_count: int,
    zone_count: int,
) -> NDArray[np.intp]:
    """Return each territory's highest-weight zone (-1 when none matched), ties going to the first seen."""
    matched = zones >= 0
    keys = owners[matched] * zone_count + zones[matched]
    unique_keys, first_seen, inverse = np.unique(keys, return_index=True, return_inverse=True)
    # bincount adds in input order, matching a sequential per-territory sum of sample weights.
    totals = np.bincount(inverse, weights=weights[matched], minlength=len(unique_keys))
    key_owners, key_zones = np.divmod(unique_keys, zone_count)
    # Sort by territory, then descending weight, then first appearance; the first row per territory wins.
    order = np.lexsort((first_seen, -totals, key_owners))
    winners = order[np.r_[True, key_owners[order][1:] != key_owners[order][:-1]]] if len(order) else order
    majority = np.full(territory_count, -1, dtype=np.intp)
    majority[key_owners[winners]] = key_zones[winners]
    return majority
