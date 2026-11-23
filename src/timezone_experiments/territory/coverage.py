"""Weighted coverage evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from timezone_experiments.config import CoverageType
    from timezone_experiments.data.models import Territory


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """Fraction of territory/population satisfying a solar constraint."""

    compliance_rate: float
    population_covered: float | None


@dataclass(frozen=True, slots=True)
class CoverageWeights:
    """Per-sample weights extracted once per territory, so each candidate offset is just a masked sum."""

    weights: NDArray[np.float64]
    total: float
    population: NDArray[np.float64] | None
    population_total: float

    @classmethod
    def from_territory(cls, territory: Territory, coverage_type: CoverageType) -> CoverageWeights:
        """Collect the weights ``coverage_type`` measures, plus population weights when every sample has one."""
        samples = territory.samples
        if coverage_type == "longitude":
            weights = np.fromiter((sample.longitude_weight for sample in samples), np.float64, len(samples))
        elif coverage_type == "area":
            weights = np.fromiter((sample.area_weight for sample in samples), np.float64, len(samples))
        elif coverage_type == "population":
            if any(sample.population_weight is None for sample in samples):
                msg = (
                    f"Population coverage requested but territory {territory.territory_id!r} "
                    "has missing population weights."
                )
                raise ValueError(msg)
            weights = np.fromiter((sample.population_weight or 0.0 for sample in samples), np.float64, len(samples))
        else:
            msg = f"Unsupported coverage type: {coverage_type!r}."
            raise ValueError(msg)
        total = float(weights.sum())
        if total <= 0:
            msg = f"Territory {territory.territory_id!r} has no positive {coverage_type} weights."
            raise ValueError(msg)
        population = None
        if all(sample.population_weight is not None for sample in samples):
            population = np.fromiter((sample.population_weight or 0.0 for sample in samples), np.float64, len(samples))
        population_total = 0.0 if population is None else float(population.sum())
        return cls(weights, total, population, population_total)

    def evaluate(self, compliant: NDArray[np.bool_]) -> CoverageSummary:
        """Summarize coverage for a boolean mask aligned with the territory's samples."""
        if compliant.shape != self.weights.shape:
            msg = "Compliance values must correspond one-to-one with territory samples."
            raise ValueError(msg)
        population_covered = None
        if self.population is not None and self.population_total:
            population_covered = float(self.population[compliant].sum()) / self.population_total
        return CoverageSummary(float(self.weights[compliant].sum()) / self.total, population_covered)


def evaluate_coverage(
    territory: Territory,
    compliant: Sequence[bool],
    coverage_type: CoverageType,
) -> CoverageSummary:
    """Compute weighted compliance and optional population coverage."""
    if len(compliant) != len(territory.samples):
        msg = "Compliance values must correspond one-to-one with territory samples."
        raise ValueError(msg)
    return CoverageWeights.from_territory(territory, coverage_type).evaluate(np.asarray(compliant, dtype=np.bool_))
