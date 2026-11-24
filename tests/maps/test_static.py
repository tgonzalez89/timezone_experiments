"""Static (PNG/SVG) map rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from factories import (
    map_inputs,
)
from matplotlib.figure import Figure

from timezone_experiments.maps import (
    MAP_FIELDS,
    MapOptions,
    export_maps,
    result_feature_collection,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_static_rasterization_applies_to_every_map_fill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    territories, results = map_inputs((0,))
    rasterized_fields: list[str] = []

    def capture(figure: Figure, path: Path, **_kwargs: object) -> None:
        assert figure.axes[0].collections[0].get_rasterized()
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
