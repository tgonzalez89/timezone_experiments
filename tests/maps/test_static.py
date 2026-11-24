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
