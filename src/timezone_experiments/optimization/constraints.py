"""Solar-noon constraint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

import numpy as np

from timezone_experiments.constants import MINUTES_PER_DAY

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class NoonConstraint:
    """Inclusive minimum and maximum local solar-noon bounds."""

    minimum_minutes: int | None
    maximum_minutes: int | None

    @property
    def window_start(self) -> float:
        """Start of the 24-hour clock window that noon times are compared in.

        Clock times repeat every day, so 00:14 could mean 14 minutes past midnight or 11h46 before noon
        tomorrow's frame. Centering the window on the active bound (half a day either side) makes "earlier"
        and "later" unambiguous: a noon just after midnight counts as very early under a minimum bound,
        and as very late (24:14) under a maximum-only bound. The default 12:00 minimum gives 00:00-24:00.
        """
        anchor = self.minimum_minutes if self.minimum_minutes is not None else self.maximum_minutes
        return 0.0 if anchor is None else anchor - MINUTES_PER_DAY / 2

    @overload
    def normalize(self, solar_noon_minutes: float) -> float: ...
    @overload
    def normalize(self, solar_noon_minutes: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def normalize(self, solar_noon_minutes: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Map clock times (any day displacement) into this constraint's comparison window."""
        start = self.window_start
        return start + (solar_noon_minutes - start) % MINUTES_PER_DAY

    def compliant(self, solar_noon_minutes: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Return which local solar-noon clock times meet both bounds."""
        values = self.normalize(solar_noon_minutes)
        accepted = np.ones(values.shape, dtype=np.bool_)
        if self.minimum_minutes is not None:
            accepted &= values >= self.minimum_minutes
        if self.maximum_minutes is not None:
            accepted &= values <= self.maximum_minutes
        return accepted

    def accepts(self, solar_noon_minutes: float) -> bool:
        """Return whether one local solar-noon clock time meets both bounds."""
        return bool(self.compliant(np.array([solar_noon_minutes], dtype=np.float64))[0])
