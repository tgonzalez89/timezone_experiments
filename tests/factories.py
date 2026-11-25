"""Builders shared by tests in several packages (fixtures live in conftest.py)."""

from __future__ import annotations

import io
import json
import statistics
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Self

from timezone_experiments.astronomy import solar as solar_module
from timezone_experiments.core.config import (
    ExperimentConfig,
    SolarMetric,
)
from timezone_experiments.core.models import Territory, TerritorySample
from timezone_experiments.optimization.optimizer import AssignmentResult, optimize_territory

if TYPE_CHECKING:
    from collections.abc import Iterable

ROOT = Path(__file__).parents[1]
TOY_BOUNDARIES = ROOT / "tests" / "resources" / "toy_boundaries.geojson"


def make_territory(
    territory_id: str = "TST:1",
    longitude: float = 0,
    *,
    current_timezone: str | None = "UTC+00:00",
    geometry: dict[str, object] | None = None,
    samples: tuple[TerritorySample, ...] | None = None,
) -> Territory:
    """Return a territory with one sample point (or the given samples), named after its ID's country."""
    return Territory(
        territory_id=territory_id,
        country_code=territory_id.split(":", maxsplit=1)[0],
        name="Test",
        admin_level=0,
        samples=samples or (TerritorySample(latitude=0, longitude=longitude),),
        current_timezone=current_timezone,
        geometry=geometry,
    )


class DatasetResponse(io.BytesIO):
    def __init__(self, content: bytes, *, known_length: bool = True) -> None:
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content))} if known_length else {}
        self.status = 200


SQUARE: dict[str, Any] = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}


class FakeStreamlit(ModuleType):
    def __init__(self, *, has_upload: bool, run: bool, rules: str | None) -> None:
        super().__init__("streamlit")
        self.has_upload = has_upload
        self.run = run
        self.rules = rules
        self.errors: list[str] = []
        self.downloads: list[str] = []
        self.captions: list[str] = []
        self.warnings: list[str] = []
        self.number_inputs: dict[str, dict[str, object]] = {}
        self.session_state: dict[str, object] = {}

    def set_page_config(self, **kwargs: object) -> None:
        return None

    def title(self, value: str) -> None:
        return None

    def caption(self, value: str) -> None:
        self.captions.append(value)

    def file_uploader(self, label: str, **kwargs: object) -> UploadedTimezones | None:
        return UploadedTimezones() if "timezone" in label and self.has_upload else None

    def subheader(self, value: str) -> None:
        return None

    def selectbox(self, label: str, options: tuple[object, ...], *, index: int = 0, **kwargs: object) -> object:
        return options[index]

    def checkbox(self, label: str, *, value: bool, **kwargs: object) -> bool:
        return value

    def number_input(self, label: str, **kwargs: object) -> int:
        self.number_inputs[label] = kwargs
        value = kwargs.get("value", 0)
        return value if isinstance(value, int) else 0

    def text_area(self, label: str, **kwargs: object) -> str:
        return (
            self.rules
            if self.rules is not None and label.startswith("Subdivision rules")
            else str(kwargs.get("value", ""))
        )

    def slider(self, label: str, **kwargs: object) -> float:
        value = kwargs.get("value", 1.0)
        return float(value) if isinstance(value, (int, float)) else 1.0

    def text_input(self, label: str, **kwargs: object) -> str:
        return str(kwargs.get("value", ""))

    def columns(self, count: int) -> tuple[FakeStreamlit, ...]:
        return (self,) * count

    def expander(self, label: str, *, expanded: bool) -> FakeStreamlit:
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def info(self, message: str) -> None:
        return None

    def button(self, label: str, **kwargs: object) -> bool:
        return self.run

    def progress(self, value: float, **kwargs: object) -> FakeStreamlit:
        return self

    def dataframe(self, data: object, **kwargs: object) -> None:
        return None

    def plotly_chart(self, figure: object, **kwargs: object) -> None:
        return None

    def download_button(self, label: str, **kwargs: str) -> None:
        self.downloads.append(label)

    def success(self, message: str) -> None:
        return None

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class UploadedTimezones:
    name = "toy_timezones.geojson"

    @staticmethod
    def getvalue() -> bytes:
        return TOY_BOUNDARIES.read_bytes()


def daily_noon(longitude: float, year: int, offset: float = 0.0) -> list[float]:
    """Return reference daily local solar-noon clock times (not wrapped to a day), straight from the definition."""
    return [720 - 4 * longitude - equation + 60 * offset for equation in solar_module.equation_of_time_minutes(year)]


def html_map_data(path: Path) -> tuple[object, object]:
    arguments = path.read_text(encoding="utf-8").rsplit("Plotly.newPlot(", maxsplit=1)[1]
    payload = arguments.split(",", maxsplit=1)[1].lstrip()
    data, end = json.JSONDecoder().raw_decode(payload)
    layout, _ = json.JSONDecoder().raw_decode(payload[end:].lstrip().removeprefix(",").lstrip())
    return data, layout


def map_inputs(shifts: tuple[float | None, ...]) -> tuple[list[Territory], list[AssignmentResult]]:
    geometry: dict[str, object] = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    }
    territories = [make_territory(f"TST:{index}", geometry=geometry) for index in range(len(shifts))]
    config = ExperimentConfig(utc_offsets=(1,), minimum_noon=None)
    results = [
        replace(optimize_territory(territory, config), timezone_shift_hours=shift)
        for territory, shift in zip(territories, shifts, strict=True)
    ]
    results.append(replace(results[0], territory_id="unmapped", timezone_shift_hours=12))
    return territories, results


def statistic(values: Iterable[float], metric: SolarMetric) -> float:
    data = list(values)
    return min(data) if metric == "min" else max(data) if metric == "max" else float(statistics.median(data))


def map_source(geometry: dict[str, object], offset: float = 1) -> tuple[Territory, AssignmentResult]:
    territory = Territory(
        territory_id="TST:1",
        country_code="TST",
        name="Test",
        admin_level=0,
        samples=(TerritorySample(50, 0),),
        geometry=geometry,
    )
    result = optimize_territory(territory, ExperimentConfig(year=2024, minimum_noon=None, utc_offsets=(offset,)))
    return territory, result


def box_ring(west: float, south: float, east: float, north: float) -> list[list[float]]:
    return [[west, south], [east, south], [east, north], [west, north], [west, south]]


def territory_trace(figure: Any) -> Any:  # noqa: ANN401
    """Return a map figure's territory fills, skipping the faded UTC-zone bands drawn under them on the offset map."""
    return next(trace for trace in figure.data if trace.name != "UTC zone bands")
