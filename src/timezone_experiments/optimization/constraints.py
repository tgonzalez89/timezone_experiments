"""Solar-noon constraint helpers."""

from __future__ import annotations

from dataclasses import dataclass

from timezone_experiments.config import clock_minutes


@dataclass(frozen=True, slots=True)
class NoonConstraint:
    """Inclusive minimum and maximum local solar-noon bounds."""

    minimum_minutes: int | None
    maximum_minutes: int | None

    @classmethod
    def from_clocks(cls, minimum: str | None, maximum: str | None) -> NoonConstraint:
        """Create a constraint from optional HH:MM bounds."""
        lower = clock_minutes(minimum)
        upper = clock_minutes(maximum)
        if lower is not None and upper is not None and lower > upper:
            msg = "minimum_noon cannot be later than maximum_noon."
            raise ValueError(msg)
        return cls(lower, upper)

    def accepts(self, solar_noon_minutes: float) -> bool:
        """Return whether a local solar-noon clock time meets both bounds."""
        return (self.minimum_minutes is None or solar_noon_minutes >= self.minimum_minutes) and (
            self.maximum_minutes is None or solar_noon_minutes <= self.maximum_minutes
        )
