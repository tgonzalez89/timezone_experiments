"""Static (PNG/SVG) map rendering."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from factories import (
    map_inputs,
)
from matplotlib.figure import Figure
from PIL import Image

from timezone_experiments.maps import (
    MAP_FIELDS,
    MapOptions,
    export_maps,
    result_feature_collection,
)
from timezone_experiments.maps.static import PNG_WIDTH_PIXELS, SVG_DISPLAY_SCALE

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_static_rasterization_applies_to_every_map_fill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    territories, results = map_inputs((0,))
    rasterized_fields: list[str] = []

    def capture(figure: Figure, path: Path, **_kwargs: object) -> None:
        axis = figure.axes[0]
        assert axis.collections[0].get_rasterized()
        # Edges would be rasterized with the fills, so every map has none and draws its subdivision borders as lines.
        assert (
            not len(axis.collections[0].get_edgecolor())
            or np.asarray(axis.collections[0].get_edgecolor())[:, 3].max() == 0
        )
        line_collections = [item for item in axis.collections[1:] if type(item).__name__ == "LineCollection"]
        assert len(line_collections) == 2  # subdivision borders and country borders, neither rasterized
        assert not any(item.get_rasterized() for item in line_collections)
        rasterized_fields.append(path.stem)

    monkeypatch.setattr(Figure, "savefig", capture)
    export_maps(
        result_feature_collection(territories, results),
        tmp_path,
        MapOptions(rasterize_static_map_fills=True),
        kinds=("static",),
    )
    assert set(rasterized_fields) == set(MAP_FIELDS)


def test_static_maps_draw_the_data_credits_only_when_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    territories, results = map_inputs((0,))
    credited: dict[str, bool] = {}

    def capture(figure: Figure, path: Path, **_kwargs: object) -> None:
        if path.suffix == ".png":
            texts = {text.get_text() for text in figure.axes[0].texts}
            credited[f"{path.parent.name}/{path.stem}"] = "Data: Natural Earth v5.1.2." in texts

    monkeypatch.setattr(Figure, "savefig", capture)
    for attribution, directory in (("Data: Natural Earth v5.1.2.", "credited"), ("", "plain")):
        export_maps(
            result_feature_collection(territories, results),
            tmp_path / directory,
            MapOptions(attribution=attribution),
            kinds=("static",),
        )
    assert credited == {
        f"{directory}/{field}": directory == "credited" for directory in ("credited", "plain") for field in MAP_FIELDS
    }


def test_data_credits_are_part_of_the_rendered_map_settings() -> None:
    assert MapOptions(attribution="Data: x.").rendering_settings() != MapOptions().rendering_settings()


def test_every_static_map_has_the_same_size_and_a_8k_png_and_4k_default_svg_size(tmp_path: Path) -> None:
    territories, results = map_inputs((0, 0, 0))
    export_maps(result_feature_collection(territories, results), tmp_path, kinds=("static",))
    sizes = {
        field: re.search(
            r'width="([\d.]+)pt" height="([\d.]+)pt"', (tmp_path / f"{field}.svg").read_text("utf-8")[:4096]
        )
        for field in MAP_FIELDS
    }
    matches = [match for match in sizes.values() if match is not None]
    assert len(matches) == len(sizes)
    assert len({match.groups() for match in matches}) == 1
    width = float(matches[0].group(1))
    assert width == 16 * 72 * SVG_DISPLAY_SCALE
    png_sizes = {Image.open(tmp_path / f"{field}.png").size for field in MAP_FIELDS}
    assert len(png_sizes) == 1
    assert next(iter(png_sizes))[0] == PNG_WIDTH_PIXELS  # 8K wide


def test_svg_coordinates_are_trimmed_to_three_decimals(tmp_path: Path) -> None:
    territories, results = map_inputs((0, 0, 0))
    export_maps(result_feature_collection(territories, results), tmp_path, kinds=("static",))
    text = (tmp_path / f"{MAP_FIELDS[0]}.svg").read_text("utf-8")
    assert re.search(r"[\d]\.[\d]{4,}", text.split("<defs>", 1)[-1]) is None
