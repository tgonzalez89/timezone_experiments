"""Pinned dataset catalog: sources.json, per-download metadata files, and data credits."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from timezone_experiments.core.fingerprint import file_digest
from timezone_experiments.data.download.catalog import (
    SOURCES,
    data_credits,
    input_record,
    load_sources,
    metadata_path,
    read_source_metadata,
    write_derived_metadata,
    write_source_metadata,
)
from timezone_experiments.data.download.http import validate_dataset_url

if TYPE_CHECKING:
    from pathlib import Path


def test_bundled_pins_cover_every_dataset_with_allowed_urls() -> None:
    assert list(SOURCES) == ["natural_earth", "gadm", "geoboundaries", "timezone_boundary_builder", "ghsl_population"]
    for source in SOURCES.values():
        assert source.license
        assert source.homepage.startswith("https://")
        validate_dataset_url(source.resolve(country="LUX", level=0))
    assert SOURCES["geoboundaries"].version is None


def test_urls_and_credits_use_the_pinned_or_requested_version() -> None:
    gadm = SOURCES["gadm"]
    assert gadm.resolve("4.1", country="FRA", level=2) == (
        "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_FRA_2.json"
    )
    assert "gadm5.0/json/gadm50_FRA_1" in gadm.resolve("5.0", country="FRA", level=1)
    assert gadm.credit() == f"GADM {gadm.version} (non-commercial use only)"
    assert SOURCES["natural_earth"].credit("v9.9.9") == "Natural Earth v9.9.9"


def test_pin_files_must_hold_complete_entries(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"natural_earth": {"name": "Natural Earth"}}), encoding="utf-8")
    with pytest.raises(TypeError):
        load_sources(path)


def test_source_metadata_records_the_download(tmp_path: Path) -> None:
    path = tmp_path / "countries.geojson"
    path.write_text("{}", encoding="utf-8")
    write_source_metadata(path, SOURCES["natural_earth"], "https://example.org/countries.geojson", level=0)
    assert metadata_path(path) == tmp_path / "countries.source.json"
    metadata = read_source_metadata(path)
    assert metadata is not None
    assert metadata["source"] == "natural_earth"
    assert metadata["version"] == SOURCES["natural_earth"].version
    assert metadata["url"] == "https://example.org/countries.geojson"
    assert metadata["sha256"] == file_digest(path)
    assert metadata["license"] == "Public domain"
    assert metadata["level"] == 0
    assert metadata["retrieved"]


@pytest.mark.parametrize("content", ["{damaged", "[1, 2]"], ids=["corrupt", "not-an-object"])
def test_unusable_metadata_reads_as_missing(tmp_path: Path, content: str) -> None:
    path = tmp_path / "countries.geojson"
    assert read_source_metadata(path) is None
    metadata_path(path).write_text(content, encoding="utf-8")
    assert read_source_metadata(path) is None


def test_input_records_reuse_recorded_checksums_and_hash_user_files(tmp_path: Path) -> None:
    downloaded = tmp_path / "downloaded.geojson"
    downloaded.write_text("{}", encoding="utf-8")
    metadata_path(downloaded).write_text(json.dumps({"source": "gadm", "sha256": "recorded"}), encoding="utf-8")
    record = input_record(downloaded)
    assert record["file"] == "downloaded.geojson"
    assert record["sha256"] == "recorded"
    assert record["metadata"]["source"] == "gadm"
    uploaded = tmp_path / "uploaded.csv"
    uploaded.write_text("latitude,longitude,population\n", encoding="utf-8")
    assert input_record(uploaded) == {"file": "uploaded.csv", "sha256": file_digest(uploaded), "metadata": None}


def test_credits_follow_assembled_files_to_every_source_at_its_recorded_version(tmp_path: Path) -> None:
    world, subdivisions, assembled = (tmp_path / f"{name}.geojson" for name in ("world", "gadm", "mixed"))
    for path in (world, subdivisions, assembled):
        path.write_text("{}", encoding="utf-8")
    write_source_metadata(world, SOURCES["natural_earth"], "https://example.org/world")
    metadata_path(subdivisions).write_text(json.dumps({"source": "gadm", "version": "3.6"}), encoding="utf-8")
    write_derived_metadata(assembled, [world, subdivisions])
    derived = read_source_metadata(assembled)
    assert derived is not None
    assert [component["file"] for component in derived["derived_from"]] == ["world.geojson", "gadm.geojson"]
    assert derived["sha256"] == file_digest(assembled)
    uploaded = tmp_path / "timezones.geojson"
    uploaded.write_text("{}", encoding="utf-8")
    credits_line = data_credits([input_record(assembled), input_record(uploaded), None])
    assert credits_line == (
        f"Data: Natural Earth {SOURCES['natural_earth'].version}; GADM 3.6 (non-commercial use only)."
    )


def test_credits_add_extra_sources_at_their_pinned_versions_and_are_empty_without_sources() -> None:
    version = SOURCES["timezone_boundary_builder"].version
    assert data_credits([], extra_sources=["timezone_boundary_builder"]).startswith(
        f"Data: timezone-boundary-builder {version}"
    )
    assert data_credits([{"file": "upload.csv", "metadata": None}]) == ""
