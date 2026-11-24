"""Timezone-shift map: a symmetric blue-white-orange ramp around zero, filling whole territories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from timezone_experiments.maps.scales import SHIFT_COLORS, Scale, noon_hover
from timezone_experiments.maps.spec import MapSpec, observed_values

if TYPE_CHECKING:
    from timezone_experiments.maps.spec import GradientLayer, MapOptions


class ShiftMap(MapSpec):
    """Continuous map of the proposed offset minus the current one.

    The scale is symmetric, extending to the largest absolute shift on both sides, so equal advances and delays
    have equal intensity; an all-zero map uses -1..+1 hours.
    """

    field = "timezone_shift_hours"
    title = "Timezone Shift Hours"
    layer: GradientLayer | None = None
    hover_label = "Value"
    missing_label: str | None = None

    @override
    def scale(self, features: list[dict[str, Any]]) -> Scale:
        limit = max(map(abs, observed_values(features, self.field)), default=0.0) or 1.0
        return Scale(title="Timezone shift (hours)", minimum=-limit, maximum=limit, colors=SHIFT_COLORS)

    @override
    def value_text(self, value: float | None) -> object:
        return value

    @override
    def details(self, properties: dict[str, Any], row: dict[str, Any]) -> str:
        return noon_hover(row["noon"])

    @override
    def footnote(self, options: MapOptions) -> str | None:
        return None
