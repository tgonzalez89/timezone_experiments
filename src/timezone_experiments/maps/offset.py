"""Proposed UTC offset map: one discrete color per observed offset, filling whole territories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from timezone_experiments.maps.scales import Scale, categorical_colors, noon_hover
from timezone_experiments.maps.spec import MapSpec, observed_values

if TYPE_CHECKING:
    from timezone_experiments.maps.spec import GradientLayer, MapOptions


class OffsetMap(MapSpec):
    """Categorical map of the proposed permanent offsets; only observed offsets get legend entries."""

    field = "proposed_utc_offset_hours"
    title = "Proposed Utc Offset Hours"
    layer: GradientLayer | None = None
    hover_label = "Value"
    missing_label: str | None = None

    @override
    def scale(self, features: list[dict[str, Any]]) -> Scale:
        categories = tuple(observed_values(features, self.field))
        return Scale(
            title=self.field.replace("_", " "),
            categories=categories,
            category_colors=categorical_colors(len(categories)),
        )

    @override
    def value_text(self, value: float | None) -> object:
        return value

    @override
    def details(self, properties: dict[str, Any], row: dict[str, Any]) -> str:
        return noon_hover(row["noon"])

    @override
    def footnote(self, options: MapOptions) -> str | None:
        return None
