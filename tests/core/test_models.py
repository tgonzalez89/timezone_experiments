"""Territory and sample models, and the noon constraint."""

from __future__ import annotations

import pytest

from timezone_experiments.core.config import (
    ExperimentConfig,
)
from timezone_experiments.core.models import (
    NoonConstraint,
    Territory,
    TerritorySample,
    pack_base_samples,
    unpack_base_samples,
)


def test_constraint_bounds() -> None:
    assert ExperimentConfig(minimum_noon="12:00", maximum_noon="13:00").noon_constraint.accepts(750)
    assert not ExperimentConfig(minimum_noon=None, maximum_noon="13:00").noon_constraint.accepts(781)
    assert ExperimentConfig(minimum_noon=None).noon_constraint == NoonConstraint(None, None)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"latitude": 90.5}, "Latitude must be between -90 and 90"),
        ({"latitude": -91}, "Latitude must be between -90 and 90"),
        ({"latitude": float("nan")}, "Latitude must be between -90 and 90"),
        ({"longitude": 180.5}, "Longitude must be between -180 and 180"),
        ({"longitude": -181}, "Longitude must be between -180 and 180"),
        ({"area_weight": float("nan")}, "finite and non-negative"),
        ({"area_weight": -1.0}, "finite and non-negative"),
        ({"longitude_weight": float("inf")}, "finite and non-negative"),
        ({"longitude_weight": -1.0}, "finite and non-negative"),
        ({"population_weight": float("nan")}, "finite and non-negative"),
        ({"population_weight": -1.0}, "finite and non-negative"),
    ],
    ids=lambda value: value if isinstance(value, str) else ",".join(f"{k}={v}" for k, v in value.items()),
)
def test_samples_reject_invalid_coordinates_and_weights(fields: dict[str, float], message: str) -> None:
    values: dict[str, float] = {"latitude": 0, "longitude": 0, **fields}
    with pytest.raises(ValueError, match=message):
        TerritorySample(**values)


@pytest.mark.parametrize(
    ("latitude", "longitude", "weight", "population"),
    [(90, 180, 1, None), (-90, -180, 1, None), (0, 0, 0, 0), (45, 10, 0.5, 12.5)],
)
def test_samples_accept_boundary_coordinates_and_zero_weights(
    latitude: float, longitude: float, weight: float, population: float | None
) -> None:
    sample = TerritorySample(
        latitude, longitude, area_weight=weight, longitude_weight=weight, population_weight=population
    )
    assert (sample.latitude, sample.longitude, sample.population_weight) == (latitude, longitude, population)
    assert sample.with_zero_population().population_weight == 0


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"territory_id": ""}, "territory_id cannot be empty"),
        ({"country_code": ""}, "country_code cannot be empty"),
        ({"samples": ()}, "must contain at least one sample"),
    ],
    ids=["no-id", "no-country", "no-samples"],
)
def test_territories_require_an_identifier_a_country_and_samples(fields: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "territory_id": "TST:1",
        "country_code": "TST",
        "name": "Test",
        "admin_level": 0,
        "samples": (TerritorySample(0, 0),),
        **fields,
    }
    with pytest.raises(ValueError, match=message):
        Territory(**values)  # ty: ignore[invalid-argument-type]


def test_base_samples_round_trip_through_their_packed_form() -> None:
    samples = (
        TerritorySample(1.5, -2.25, area_weight=0.5, longitude_weight=0.25),
        TerritorySample(-89, 179, area_weight=0, longitude_weight=1),
    )
    assert unpack_base_samples(pack_base_samples(samples)) == samples
    with pytest.raises(ValueError, match="whole"):
        unpack_base_samples([1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("constraint", "minutes", "accepted"),
    [
        # Just after midnight is far too early under a minimum bound ...
        (NoonConstraint(720, None), 14, False),
        (NoonConstraint(720, None), 1430, True),
        # ... and far too late (24:14) under a maximum-only bound.
        (NoonConstraint(None, 780), 14, False),
        (NoonConstraint(None, 780), 779, True),
        (NoonConstraint(720, 780), 750, True),
        (NoonConstraint(None, None), 14, True),
    ],
)
def test_constraint_compares_clock_times_in_a_window_centered_on_its_bounds(
    constraint: NoonConstraint, minutes: float, *, accepted: bool
) -> None:
    assert constraint.accepts(minutes) is accepted
    assert constraint.accepts(minutes + 1440) is accepted
