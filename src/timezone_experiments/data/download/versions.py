"""Find the newest published release of each pinned dataset.

Used by ``scripts/update_data_sources.py`` (which reports and rewrites the pins) and, for the GitHub-hosted
datasets, by a warn-only check before downloading. Nothing here changes the pins.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from timezone_experiments.core.files import read_json_dataset, write_text_atomic
from timezone_experiments.data.download.catalog import SOURCES, DataSource
from timezone_experiments.data.download.http import DownloadError, fetch_metadata, url_exists

if TYPE_CHECKING:
    from pathlib import Path

# Set to any non-empty value to skip the pre-download release check (offline use, tests).
UPDATE_CHECK_ENV = "TIMEZONE_EXPERIMENTS_NO_UPDATE_CHECK"
_CHECK_INTERVAL = timedelta(days=1)
_CHECK_STATE = ".release-checks.json"
_GITHUB_REPOSITORIES = {
    "natural_earth": "nvkelso/natural-earth-vector",
    "timezone_boundary_builder": "evansiroky/timezone-boundary-builder",
}
_GHSL_LISTING = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
_GHSL_RELEASE = re.compile(r"GHS_POP_GLOBE_(R(\d{4})([A-Z]))")
# A small per-country GADM file that exists in every release, used to probe whether a version is published.
PROBE_FIELDS = {"country": "LUX", "level": 0}
logger = logging.getLogger(__name__)


def latest_github_release(repository: str) -> str:
    """Return the tag of a GitHub repository's latest release."""
    url = f"https://api.github.com/repos/{repository}/releases/latest"
    answer = json.loads(fetch_metadata(url))
    tag = answer.get("tag_name") if isinstance(answer, dict) else None
    if not isinstance(tag, str) or not tag:
        msg = f"GitHub did not report a latest release tag for {repository}."
        raise ValueError(msg)
    return tag


def latest_gadm_version(source: DataSource) -> str:
    """Return the newest published GADM version, probing the next minor and major versions (GADM has no API)."""
    current = source.version or ""
    major, minor = (int(part) for part in current.split("."))
    while True:
        for candidate in (f"{major}.{minor + 1}", f"{major + 1}.0"):
            if url_exists(source.resolve(candidate, **PROBE_FIELDS)):
                major, minor = (int(part) for part in candidate.split("."))
                break
        else:
            return f"{major}.{minor}"


def latest_ghsl_release() -> str:
    """Return the newest GHS-POP release listed in the JRC open-data folder (e.g. ``R2023A``)."""
    listing = fetch_metadata(_GHSL_LISTING).decode("utf-8", errors="replace")
    releases: set[tuple[int, str, str]] = {
        (int(match[2]), match[3], match[1]) for match in _GHSL_RELEASE.finditer(listing)
    }
    if not releases:
        msg = f"No GHS-POP releases are listed at {_GHSL_LISTING}."
        raise ValueError(msg)
    return max(releases)[2]


def latest_version(source: DataSource) -> str | None:
    """Return the newest release of a dataset, or None when it cannot be pinned (only a current build is served)."""
    if source.version is None:
        return None
    if source.key in _GITHUB_REPOSITORIES:
        return latest_github_release(_GITHUB_REPOSITORIES[source.key])
    if source.key == "gadm":
        return latest_gadm_version(source)
    if source.key == "ghsl_population":
        return latest_ghsl_release()
    return None


def warn_if_outdated(key: str, state_directory: Path) -> None:
    """Log a warning when a newer release than the pinned one exists; never fails and checks at most once a day.

    The result is remembered in ``state_directory`` (the downloads folder). Only GitHub-hosted datasets are
    checked here; ``scripts/update_data_sources.py`` checks every dataset.
    """
    if os.environ.get(UPDATE_CHECK_ENV, "") != "" or key not in _GITHUB_REPOSITORIES:
        return
    source = SOURCES[key]
    state_path = state_directory / _CHECK_STATE
    state = _read_state(state_path)
    entry = state.get(key)
    now = datetime.now(UTC)
    if isinstance(entry, dict) and entry.get("pinned") == source.version and _recent(entry.get("checked"), now):
        latest = entry.get("latest")
    else:
        try:
            latest = latest_version(source)
        except (DownloadError, ValueError) as error:
            logger.debug("Could not check for a newer %s release: %s", source.name, error)
            return
        state[key] = {"pinned": source.version, "latest": latest, "checked": now.isoformat(timespec="seconds")}
        try:
            write_text_atomic(state_path, json.dumps(state, indent=2) + "\n")
        except OSError as error:
            logger.debug("Could not remember the release check in %s: %s", state_path, error)
    if isinstance(latest, str) and latest != source.version:
        logger.warning(
            "%s %s is available (pinned: %s); run scripts/update_data_sources.py --write to update.",
            source.name,
            latest,
            source.version,
        )


def _read_state(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        state = read_json_dataset(path)
    except OSError, ValueError:
        return {}
    return {str(key): value for key, value in state.items()} if isinstance(state, dict) else {}


def _recent(checked: object, now: datetime) -> bool:
    if not isinstance(checked, str):
        return False
    try:
        return now - datetime.fromisoformat(checked) < _CHECK_INTERVAL
    except ValueError:
        return False
