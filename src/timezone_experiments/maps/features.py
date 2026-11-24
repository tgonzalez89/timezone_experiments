"""Map feature collections: optimization results joined to their territories' boundary geometry."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from timezone_experiments.core.models import Territory
    from timezone_experiments.optimization.optimizer import AssignmentResult


def result_feature_collection(
    territories: list[Territory],
    results: list[AssignmentResult],
) -> dict[str, Any]:
    """Join optimization results to their territories' boundary geometry as a map feature collection."""
    assignments = {result.territory_id: result for result in results}
    geometries: dict[str, dict[str, Any]] = {}
    mapped_results: list[AssignmentResult] = []
    for territory in territories:
        assignment = assignments.get(territory.territory_id)
        if assignment is None:
            continue
        if territory.geometry is None:
            msg = f"Map export requires boundary geometry for {territory.territory_id!r}."
            raise ValueError(msg)
        geometries[territory.territory_id] = territory.geometry
        mapped_results.append(assignment)
    return geometry_feature_collection(geometries, mapped_results)


def geometry_feature_collection(
    geometries: dict[str, dict[str, Any]], results: list[AssignmentResult]
) -> dict[str, Any]:
    """Join saved results to boundary geometry keyed by territory ID (no samples needed)."""
    features: list[dict[str, Any]] = []
    for assignment in results:
        geometry = geometries.get(assignment.territory_id)
        if geometry is None:
            msg = (
                f"No boundary geometry matches result territory {assignment.territory_id!r}. "
                "Use the boundaries and subdivision settings from the original run."
            )
            raise ValueError(msg)
        properties = {
            "territory_id": assignment.territory_id,
            "country_code": assignment.country_code,
            "admin_level": assignment.admin_level,
            "territory_name": assignment.territory_name,
            "current_timezone": assignment.current_timezone,
            "proposed_timezone": assignment.proposed_timezone,
            "compliance_rate": assignment.compliance_rate,
            "population_covered": assignment.population_covered,
            "fallback_used": assignment.fallback_used,
            "area_weighted_noon_minutes": assignment.area_weighted_noon_minutes,
            "proposed_utc_offset_hours": assignment.proposed_utc_offset_hours,
            "solar_year": assignment.solar_year,
            "timezone_shift_hours": assignment.timezone_shift_hours,
            "noon": asdict(assignment.noon),
        }
        features.append(
            {"type": "Feature", "id": assignment.territory_id, "properties": properties, "geometry": geometry}
        )
    return {"type": "FeatureCollection", "features": features}
