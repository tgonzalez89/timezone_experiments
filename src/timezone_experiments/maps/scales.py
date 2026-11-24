"""Renderer-neutral color scales, clock-time formatting, and the solar-time guide positions."""

from __future__ import annotations

import colorsys
import itertools
from dataclasses import dataclass
from math import ceil, floor
from typing import Any

from timezone_experiments.core.constants import MINUTES_PER_DAY

TIMEZONE_MERIDIANS = tuple(range(-180, 181, 15))
TIMEZONE_BOUNDARIES = tuple(longitude + 7.5 for longitude in TIMEZONE_MERIDIANS[:-1])
# Blue-white-orange diverging palette shared by the timezone-shift and median-noon maps.
SHIFT_COLORS = ((0.0, "#2166ac"), (0.5, "#ffffff"), (1.0, "#b35806"))
NOON_MINUTES = 12 * 60
GOLDEN_ANGLE_DEGREES = 137.50776405


@dataclass(frozen=True, slots=True)
class Scale:
    """Renderer-neutral color scale: continuous (``minimum``..``maximum``) or categorical (one color per category).

    ``title`` may contain line breaks, which each renderer translates. ``ticks`` and ``tick_labels`` are optional
    continuous-scale tick positions and their texts.
    """

    title: str
    minimum: float = 0.0
    maximum: float = 1.0
    colors: tuple[tuple[float, str], ...] = ()
    ticks: tuple[float, ...] = ()
    tick_labels: tuple[str, ...] = ()
    categories: tuple[float, ...] = ()
    category_colors: tuple[str, ...] = ()

    @property
    def categorical(self) -> bool:
        """Whether each observed value gets its own color instead of a continuous ramp."""
        return bool(self.categories)


def clock(minutes: float) -> str:
    """Format minutes after midnight as HH:MM, with a signed day suffix outside the reference day."""
    days, minute_of_day = divmod(round(minutes), MINUTES_PER_DAY)
    hours, minute = divmod(minute_of_day, 60)
    suffix = f" ({days:+d} day)" if days else ""
    return f"{hours:02d}:{minute:02d}{suffix}"


def clock_ticks(minimum: float, maximum: float) -> tuple[float, ...]:
    """Return readable clock tick positions for a scale, always including both limits (and noon when inside)."""
    target = (maximum - minimum) / 6
    step = next((value for value in (1, 5, 15, 30, 60, 120, 240) if value >= target), ceil(target / 60) * 60)
    ticks: list[float] = [index * step for index in range(ceil(minimum / step), floor(maximum / step) + 1)]
    ticks = sorted({minimum, *ticks, maximum})
    if minimum <= NOON_MINUTES <= maximum:
        ticks = sorted({*ticks, NOON_MINUTES})
    return tuple(ticks)


def clock_scale(
    title: str,
    minimum: float,
    maximum: float,
    colors: list[tuple[float, str]],
    clamped: tuple[bool, bool] = (False, False),
) -> Scale:
    """Return a continuous clock-time scale with clock ticks.

    ``clamped`` flags the low and high limits beyond which values are drawn with the endpoint color; such a limit's
    tick reads "≤ HH:MM" or "≥ HH:MM".
    """
    ticks = clock_ticks(minimum, maximum)
    labels = ["24:00" if value == MINUTES_PER_DAY else clock(value) for value in ticks]
    if clamped[0]:
        labels[0] = f"≤ {labels[0]}"
    if clamped[1]:
        labels[-1] = f"≥ {labels[-1]}"
    return Scale(
        title=title,
        minimum=minimum,
        maximum=maximum,
        colors=tuple(colors),
        ticks=ticks,
        tick_labels=tuple(labels),
    )


def ramp_window(stops: tuple[tuple[float, str], ...], minimum: float, maximum: float) -> list[tuple[float, str]]:
    """Return the part of a time-anchored color ramp within ``minimum``..``maximum`` as 0..1 color stops."""

    def color_at(time: float) -> str:
        for (start, first), (end, last) in itertools.pairwise(stops):
            if start <= time <= end:
                return blend_colors(first, last, (time - start) / (end - start))
        return stops[-1][1]

    inner = [(time, color) for time, color in stops if minimum < time < maximum]
    points = [(minimum, color_at(minimum)), *inner, (maximum, color_at(maximum))]
    return [((time - minimum) / (maximum - minimum), color) for time, color in points]


def categorical_colors(count: int) -> tuple[str, ...]:
    """Return ``count`` colors where consecutive entries differ strongly in both hue and lightness.

    Hue steps by the golden angle (about 137 degrees) and lightness alternates, so categories that are neighbors in
    sorted order (adjacent timezones) are easy to tell apart, for any number of categories. Deterministic.
    """
    colors: list[str] = []
    for index in range(count):
        hue = (index * GOLDEN_ANGLE_DEGREES % 360) / 360
        lightness = 0.40 if index % 2 == 0 else 0.68
        red, green, blue = colorsys.hls_to_rgb(hue, lightness, 0.75)
        colors.append(f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}")
    return tuple(colors)


def blend_colors(start: str, end: str, fraction: float) -> str:
    """Linearly interpolate between two ``#rrggbb`` colors."""
    first = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    last = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    return "#" + "".join(f"{round(a + fraction * (b - a)):02x}" for a, b in zip(first, last, strict=True))


def noon_hover(noon: dict[str, Any]) -> str:
    """Return the hover block with a territory's annual noon extremes, proposed and current."""
    earliest = noon["proposed_earliest_minutes"]
    latest = noon["proposed_latest_minutes"]
    details = [
        f"<b>Annual noon extremes ({noon['year']}) | whole territory</b>",
        f"Proposed: earliest {clock(earliest)}; latest {clock(latest)}",
    ]
    if not noon["current_ranges"]:
        details.append("Current noon range unavailable")
    else:
        details.append("Current baseline (representative timezone across territory):")
        for label, (earliest, latest) in noon["current_ranges"].items():
            details.append(f"{label}: earliest {clock(earliest)}; latest {clock(latest)}")
    return "<br>".join(details)
