"""Proposed UTC offset map: one discrete color per observed offset, filling whole territories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from timezone_experiments.maps.scales import Scale, blend_colors, categorical_colors, noon_hover
from timezone_experiments.maps.spec import MapSpec, observed_values

if TYPE_CHECKING:
    from timezone_experiments.maps.spec import GradientLayer, MapOptions


# Every offset a configuration can propose (UTC-12 to UTC+14). Colors are handed out by position in this list (plus any
# other offset that occurs), so an offset keeps its color whichever offsets a run assigns; only observed offsets are
# listed in the legend.
PALETTE_OFFSETS = tuple(float(hours) for hours in range(-12, 15))
_HOURS_TO_DEGREES = 15
_BAND_FADE = 0.25  # share of white mixed into a zone's color for its background band
_FIRST_ZONE, _LAST_ZONE = -12, 12


class OffsetMap(MapSpec):
    """Categorical map of the proposed permanent offsets, over faded bands showing each longitude's own zone.

    Only observed offsets get legend entries. The band of UTC+N spans the 15 degrees of longitude around N x 15
    (UTC-12 and UTC+12 are the two halves, at the west and east edges, of the zone the date line splits), in that
    offset's color mixed halfway with white. Territories assigned to a zone then sit on a band of their own color
    when they are at their zone's longitude, and on another zone's band when they are shifted.
    """

    field = "proposed_utc_offset_hours"
    title = "Proposed Utc Offset Hours"
    layer: GradientLayer | None = None
    hover_label = "Value"
    missing_label: str | None = None

    @override
    def scale(self, features: list[dict[str, Any]]) -> Scale:
        categories = tuple(observed_values(features, self.field))
        palette = sorted({*PALETTE_OFFSETS, *categories})
        colors = dict(zip(palette, categorical_colors(len(palette)), strict=True))
        bands = tuple(
            (
                max(-180.0, hours * _HOURS_TO_DEGREES - _HOURS_TO_DEGREES / 2),
                min(180.0, hours * _HOURS_TO_DEGREES + _HOURS_TO_DEGREES / 2),
                blend_colors(colors[float(hours)], "#ffffff", _BAND_FADE),
            )
            for hours in range(_FIRST_ZONE, _LAST_ZONE + 1)
        )
        return Scale(
            title=self.field.replace("_", " "),
            categories=categories,
            category_colors=tuple(colors[category] for category in categories),
            category_labels=tuple(f"{category:+g}" for category in categories),
            bands=bands,
        )

    @override
    def value_text(self, value: float | None) -> object:
        return value

    @override
    def details(self, properties: dict[str, Any], row: dict[str, Any]) -> str:
        return noon_hover(row["noon"])

    @override
    def footnote(self, options: MapOptions) -> str | None:
        return (
            "Faded background bands: the zone of each 15 degrees of longitude (UTC-12 to UTC+12), in that zone's color"
        )
