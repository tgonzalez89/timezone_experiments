"""Release detection for pinned datasets and the warn-only check before downloads."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from timezone_experiments.data.download import versions
from timezone_experiments.data.download.catalog import SOURCES
from timezone_experiments.data.download.http import DownloadError

if TYPE_CHECKING:
    from pathlib import Path

PINNED = SOURCES["natural_earth"].version


def test_github_releases_are_read_from_the_latest_release_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []

    def fetch(url: str) -> bytes:
        requested.append(url)
        return b'{"tag_name": "2030a"}'

    monkeypatch.setattr(versions, "fetch_metadata", fetch)
    assert versions.latest_version(SOURCES["timezone_boundary_builder"]) == "2030a"
    assert requested == ["https://api.github.com/repos/evansiroky/timezone-boundary-builder/releases/latest"]
    monkeypatch.setattr(versions, "fetch_metadata", lambda url: b'{"message": "Not Found"}')
    with pytest.raises(ValueError, match="latest release tag"):
        versions.latest_github_release("owner/repository")


def test_gadm_versions_are_found_by_probing_the_next_minor_and_major_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    published = {"4.2", "4.3", "5.0"}
    probed: list[str] = []

    def exists(url: str) -> bool:
        version = url.split("/gadm/gadm")[1].split("/", maxsplit=1)[0]
        probed.append(version)
        return version in published

    monkeypatch.setattr(versions, "url_exists", exists)
    gadm = replace(SOURCES["gadm"], version="4.1")
    assert versions.latest_version(gadm) == "5.0"
    assert probed == ["4.2", "4.3", "4.4", "5.0", "5.1", "6.0"]


def test_ghsl_releases_are_read_from_the_open_data_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = b'<a href="GHS_POP_GLOBE_R2022A/">x</a> <a href="GHS_POP_GLOBE_R2023A/">y</a> GHS_BUILT_S_GLOBE_R2030A'
    monkeypatch.setattr(versions, "fetch_metadata", lambda url: listing)
    assert versions.latest_version(SOURCES["ghsl_population"]) == "R2023A"
    monkeypatch.setattr(versions, "fetch_metadata", lambda url: b"<html></html>")
    with pytest.raises(ValueError, match="No GHS-POP releases"):
        versions.latest_ghsl_release()


def test_unpinned_or_unknown_sources_have_no_latest_version() -> None:
    assert versions.latest_version(SOURCES["geoboundaries"]) is None
    assert versions.latest_version(replace(SOURCES["gadm"], key="other")) is None


@pytest.fixture
def release_checks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Enable the pre-download check and answer it with a newer Natural Earth release; records the checks."""
    monkeypatch.delenv(versions.UPDATE_CHECK_ENV)
    checks: list[str] = []

    def latest(source: object) -> str:
        checks.append(str(source))
        return "v99.0.0"

    monkeypatch.setattr(versions, "latest_version", latest)
    return checks


def test_a_newer_release_is_warned_about_once_a_day(
    tmp_path: Path, release_checks: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        versions.warn_if_outdated("natural_earth", tmp_path)
        versions.warn_if_outdated("natural_earth", tmp_path)
    assert len(release_checks) == 1
    warning = (
        f"Natural Earth 1:50m admin-0 countries v99.0.0 is available (pinned: {PINNED}); "
        "run scripts/update_data_sources.py --write to update."
    )
    assert caplog.messages == [warning, warning]
    state = json.loads((tmp_path / ".release-checks.json").read_text(encoding="utf-8"))
    assert state["natural_earth"]["latest"] == "v99.0.0"
    assert state["natural_earth"]["pinned"] == PINNED


@pytest.mark.parametrize(
    "entry",
    [
        {"pinned": "v0.0.1", "latest": PINNED, "checked": datetime.now(UTC).isoformat()},
        {"pinned": PINNED, "latest": PINNED, "checked": (datetime.now(UTC) - timedelta(days=2)).isoformat()},
        {"pinned": PINNED, "latest": PINNED, "checked": "yesterday"},
        {"pinned": PINNED, "latest": PINNED, "checked": None},
    ],
    ids=["other-pin", "stale", "unreadable-date", "no-date"],
)
def test_remembered_checks_are_redone_when_stale_or_for_another_pin(
    tmp_path: Path, release_checks: list[str], entry: dict[str, object]
) -> None:
    (tmp_path / ".release-checks.json").write_text(json.dumps({"natural_earth": entry}), encoding="utf-8")
    versions.warn_if_outdated("natural_earth", tmp_path)
    assert len(release_checks) == 1


@pytest.mark.parametrize("content", ["{damaged", "[]"], ids=["corrupt", "not-an-object"])
def test_unreadable_check_state_is_replaced(tmp_path: Path, release_checks: list[str], content: str) -> None:
    state_path = tmp_path / ".release-checks.json"
    state_path.write_text(content, encoding="utf-8")
    versions.warn_if_outdated("natural_earth", tmp_path)
    assert len(release_checks) == 1
    assert "natural_earth" in json.loads(state_path.read_text(encoding="utf-8"))


def test_release_checks_never_fail_a_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv(versions.UPDATE_CHECK_ENV)

    def offline(source: object) -> str:
        url = "https://api.github.com"
        raise DownloadError(url, "offline")

    monkeypatch.setattr(versions, "latest_version", offline)
    with caplog.at_level(logging.WARNING):
        versions.warn_if_outdated("natural_earth", tmp_path)
    assert caplog.messages == []
    assert not (tmp_path / ".release-checks.json").exists()
    monkeypatch.setattr(versions, "latest_version", lambda source: "v99.0.0")
    with caplog.at_level(logging.WARNING):
        versions.warn_if_outdated("natural_earth", tmp_path / "missing" / "directory")
    assert len(caplog.messages) == 1


def test_release_checks_are_skipped_when_disabled_or_not_on_github(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(versions, "latest_version", pytest.fail)
    versions.warn_if_outdated("natural_earth", tmp_path)
    monkeypatch.delenv(versions.UPDATE_CHECK_ENV)
    versions.warn_if_outdated("gadm", tmp_path)
    assert list(tmp_path.iterdir()) == []
