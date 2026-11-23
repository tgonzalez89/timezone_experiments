"""Solar-noon constraint helpers."""

from __future__ import annotations

from dataclasses import dataclass

from timezone_experiments.config import clock_minutes

_MINUTES_PER_DAY = 1440


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

    @property
    def window_start(self) -> float:
        """Start of the 24-hour clock window that noon times are compared in.

        Clock times repeat every day, so 00:14 could mean 14 minutes past midnight or 11h46 before noon
        tomorrow's frame. Centering the window on the active bound (half a day either side) makes "earlier"
        and "later" unambiguous: a noon just after midnight counts as very early under a minimum bound,
        and as very late (24:14) under a maximum-only bound. The default 12:00 minimum gives 00:00-24:00.
        """
        anchor = self.minimum_minutes if self.minimum_minutes is not None else self.maximum_minutes
        return 0.0 if anchor is None else anchor - _MINUTES_PER_DAY / 2

    def normalize(self, solar_noon_minutes: float) -> float:
        """Map a clock time (any day displacement) into this constraint's comparison window."""
        start = self.window_start
        return start + (solar_noon_minutes - start) % _MINUTES_PER_DAY

    def accepts(self, solar_noon_minutes: float) -> bool:
        """Return whether a local solar-noon clock time meets both bounds."""
        value = self.normalize(solar_noon_minutes)
        return (self.minimum_minutes is None or value >= self.minimum_minutes) and (
            self.maximum_minutes is None or value <= self.maximum_minutes
        )
