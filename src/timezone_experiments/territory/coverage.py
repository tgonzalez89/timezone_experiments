"""Weighted coverage evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from timezone_experiments.config import CoverageType
    from timezone_experiments.data.models import Territory


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """Fraction of territory/population satisfying a solar constraint."""

    compliance_rate: float
    population_covered: float | None


def evaluate_coverage(
    territory: Territory,
    compliant: Sequence[bool],
    coverage_type: CoverageType,
) -> CoverageSummary:
    """Compute weighted compliance and optional population coverage."""
    if len(compliant) != len(territory.samples):
        msg = "Compliance values must correspond one-to-one with territory samples."
        raise ValueError(msg)
    weights = _coverage_weights(territory, coverage_type)
    total = sum(weights)
    if total <= 0:
        msg = f"Territory {territory.territory_id!r} has no positive {coverage_type} weights."
        raise ValueError(msg)
    compliance_rate = sum(weight for weight, passes in zip(weights, compliant, strict=True) if passes) / total
    population_weights = [sample.population_weight for sample in territory.samples]
    population_covered: float | None = None
    if all(weight is not None for weight in population_weights):
        population_total = sum(float(weight) for weight in population_weights if weight is not None)
        if population_total:
            population_covered = (
                sum(
                    float(weight)
                    for weight, passes in zip(population_weights, compliant, strict=True)
                    if passes and weight is not None
                )
                / population_total
            )
    return CoverageSummary(compliance_rate, population_covered)


def _coverage_weights(territory: Territory, coverage_type: CoverageType) -> list[float]:
    if coverage_type == "longitude":
        return [sample.longitude_weight for sample in territory.samples]
    if coverage_type == "area":
        return [sample.area_weight for sample in territory.samples]
    if coverage_type == "population":
        if any(sample.population_weight is None for sample in territory.samples):
            msg = (
                f"Population coverage requested but territory {territory.territory_id!r} "
                "has missing population weights."
            )
            raise ValueError(msg)
        return [float(sample.population_weight or 0) for sample in territory.samples]
    msg = f"Unsupported coverage type: {coverage_type!r}."
    raise ValueError(msg)
