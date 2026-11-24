"""GHSL population raster download and aggregation."""

from __future__ import annotations

import csv
import io
import json
from typing import TYPE_CHECKING
from zipfile import ZipFile

import numpy as np
import pytest
import rasterio
from factories import DatasetResponse
from rasterio.transform import from_origin

from timezone_experiments.data.download import http as http_module
from timezone_experiments.data.download import population as download_population
from timezone_experiments.data.download import sources as download_data
from timezone_experiments.data.population import load_population_csv

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def population_raster(tmp_path: Path) -> Path:
    path = tmp_path / "population.tif"
    values = np.array(
        [
            [1, 2, 0, 0, 10],
            [3, -200, 0, 0, 20],
            [0, 0, 4, 5, 0],
        ],
        dtype="float32",
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 2, 0.5, 0.5),
        nodata=-200,
    ) as target:
        target.write(values, 1)
    return path


def test_population_metadata_never_outlives_a_replaced_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "population_2025.csv"
    metadata_path = csv_path.with_suffix(".source.json")
    csv_path.write_text("latitude,longitude,population\n0.5,0.5,1\n", encoding="utf-8")
    metadata_path.write_text(json.dumps({"cell_degrees": 1.0}), encoding="utf-8")
    with ZipFile(tmp_path / "ghsl_population_2025.zip", "w") as archive:
        archive.writestr("population.tif", b"raster")

    def convert(_raster: Path, target: Path, **_options: object) -> tuple[int, float]:
        target.write_text("latitude,longitude,population\n0.25,0.25,1\n", encoding="utf-8")
        return 1, 1.0

    def crash(*_args: object, **_kwargs: object) -> str:
        msg = "simulated crash before metadata is written"
        raise RuntimeError(msg)

    monkeypatch.setattr(download_population, "population_raster_to_csv", convert)
    monkeypatch.setattr(download_population.json, "dumps", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        download_population.download_population(tmp_path, cell_degrees=0.5)
    assert "0.25,0.25" in csv_path.read_text(encoding="utf-8")
    assert not metadata_path.exists()


@pytest.mark.parametrize("workers", [1, 2])
def test_csv_sums_counts_and_preserves_partial_edge_cells(
    population_raster: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], workers: int
) -> None:
    csv_path = tmp_path / "population.csv"
    rows, total = download_population.population_raster_to_csv(population_raster, csv_path, workers=workers)
    points = load_population_csv(csv_path)
    assert rows == 3
    assert total == 45
    assert [(point.latitude, point.longitude, point.population) for point in points] == [
        (1.5, 0.5, 6),
        (1.5, 2.25, 30),
        (0.75, 1.5, 9),
    ]
    assert sum(point.population for point in points) == total
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    assert "Aggregating population raster rows: 0/3" in lines[0]
    assert "Aggregating population raster rows: 3/3 (100.0%; 0 left)" in lines[1]


def test_csv_rejects_upsampling_and_bad_grid_size(population_raster: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        download_population.population_raster_to_csv(population_raster, tmp_path / "bad.csv", cell_degrees=0.25)
    with pytest.raises(ValueError, match="cell size"):
        download_population.population_raster_to_csv(population_raster, tmp_path / "bad.csv", cell_degrees=0)


def test_csv_rejects_non_geographic_raster(population_raster: Path, tmp_path: Path) -> None:
    with rasterio.open(population_raster, "r+") as target:
        target.crs = rasterio.CRS.from_epsg(3857)
    with pytest.raises(ValueError, match="WGS84"):
        download_population.population_raster_to_csv(population_raster, tmp_path / "bad.csv")


def test_csv_handles_native_resolution_rounding_and_dateline_cells(population_raster: Path, tmp_path: Path) -> None:
    with rasterio.open(population_raster, "r+") as target:
        target.transform = from_origin(179.5, 2, 0.500000002, 0.500000002)
    csv_path = tmp_path / "rounded.csv"
    _, total = download_population.population_raster_to_csv(population_raster, csv_path)
    points = load_population_csv(csv_path)
    assert total == 45
    assert all(-180 <= point.longitude <= 180 for point in points)
    assert points[1].longitude == pytest.approx(-178.25)


@pytest.mark.parametrize("value", [-1, float("nan"), 0])
@pytest.mark.parametrize("workers", [1, 2])
def test_csv_rejects_invalid_or_empty_counts(
    population_raster: Path, tmp_path: Path, value: float, workers: int
) -> None:
    with rasterio.open(population_raster, "r+") as target:
        target.write(np.full((3, 5), value, dtype="float32"), 1)
    with pytest.raises(ValueError, match=r"negative|non-finite|no positive"):
        download_population.population_raster_to_csv(population_raster, tmp_path / "bad.csv", workers=workers)


def test_download_converts_and_reuses_cached_archive(
    tmp_path: Path,
    population_raster: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("source/population.tif", population_raster.read_bytes())
    calls: list[str] = []

    def download(url: str, destination: Path) -> None:
        calls.append(url)
        destination.write_bytes(buffer.getvalue())

    monkeypatch.setattr(download_population, "download_file", download)
    directory = tmp_path / "downloads"
    csv_path = download_data.prepare_population(directory, 1, workers=2)
    metadata = json.loads(csv_path.with_suffix(".source.json").read_text(encoding="utf-8"))
    assert metadata["population_total"] == 45
    assert metadata["license"] == "CC BY 4.0; European Union"
    assert download_population.download_population(directory, cell_degrees=0.5) == csv_path
    assert len(calls) == 1
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 7
    assert sum(float(row["population"]) for row in rows) == 45
    assert {path.name for path in directory.iterdir()} == {
        "population_2025.csv",
        "population_2025.source.json",
        "ghsl_population_2025.zip",
    }


def test_bad_archive_preserves_existing_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "population_2025.csv"
    csv_path.write_bytes(b"existing data")
    with ZipFile(tmp_path / "ghsl_population_2025.zip", "w") as archive:
        archive.writestr("readme.txt", "no raster")
    with pytest.raises(ValueError, match=r"exactly one .tif file"):
        download_population.download_population(tmp_path)
    assert csv_path.read_bytes() == b"existing data"


def test_an_interrupted_raster_download_caches_no_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def truncated(_request: object, _timeout: float) -> DatasetResponse:
        response = DatasetResponse(b"partial zip")
        response.headers = {"Content-Length": "1000"}
        return response

    monkeypatch.setattr(http_module, "_urlopen", truncated)
    monkeypatch.setattr(http_module.time, "sleep", lambda _seconds: None)
    directory = tmp_path / "downloads"
    with pytest.raises(http_module.DownloadError, match="Incomplete transfer"):
        download_population.download_population(directory)
    assert not any(directory.iterdir())
