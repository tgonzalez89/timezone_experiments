"""The pinned datasets (sources.json), the metadata file written beside each download, and data credits.

Versions are pinned so a re-download returns the same data; ``scripts/update_data_sources.py`` reports newer
releases and rewrites sources.json. Each version is part of its download's file name, so changing a pin downloads
the new release instead of reusing the old file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from timezone_experiments.core.files import read_json_dataset, write_text_atomic
from timezone_experiments.core.fingerprint import file_digest

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

SOURCES_PATH = Path(__file__).with_name("sources.json")
METADATA_SUFFIX = ".source.json"
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DataSource:
    """One downloadable dataset: its pinned version (None when the provider only serves its current build)."""

    key: str
    name: str
    version: str | None
    url: str
    homepage: str
    license: str
    attribution: str

    def resolve(self, version: str | None = None, **fields: object) -> str:
        """Return the download URL for ``version`` (default: the pinned one), filling ``{country}`` etc."""
        chosen = version or self.version or ""
        return self.url.format(version=chosen, compact=chosen.replace(".", ""), **fields)

    def credit(self, version: str | None = None) -> str:
        """Return the attribution text shown on outputs, for ``version`` (default: the pinned one)."""
        return self.attribution.format(version=version or self.version or "")


def load_sources(path: Path = SOURCES_PATH) -> dict[str, DataSource]:
    """Read the dataset catalog, in its file order (which is also the order of credits)."""
    with path.open(encoding="utf-8") as file:
        entries: dict[str, dict[str, Any]] = json.load(file)
    return {key: DataSource(key=key, **entry) for key, entry in entries.items()}


SOURCES = load_sources()


def metadata_path(path: Path) -> Path:
    """Return where a downloaded (or assembled) dataset's metadata is stored: ``x.geojson`` -> ``x.source.json``."""
    return path.with_suffix(METADATA_SUFFIX)


def write_source_metadata(path: Path, source: DataSource, url: str, **details: object) -> None:
    """Record what a downloaded file is: source, version, URL, retrieval time, checksum, and licence."""
    metadata = {
        "source": source.key,
        "name": source.name,
        "version": source.version,
        "url": url,
        "retrieved": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": file_digest(path),
        "license": source.license,
        "homepage": source.homepage,
        **details,
    }
    write_text_atomic(metadata_path(path), json.dumps(metadata, indent=2) + "\n")


def write_derived_metadata(path: Path, components: Iterable[Path]) -> None:
    """Record a file assembled from other datasets (e.g. selected countries plus subdivisions)."""
    metadata = {
        "derived_from": [input_record(component) for component in components],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": file_digest(path),
    }
    write_text_atomic(metadata_path(path), json.dumps(metadata, indent=2) + "\n")


def read_source_metadata(path: Path) -> dict[str, Any] | None:
    """Return a dataset's metadata, or None when it has none (user-provided files) or it is unreadable."""
    sidecar = metadata_path(path)
    if not sidecar.is_file():
        return None
    try:
        metadata = read_json_dataset(sidecar)
    except (OSError, ValueError) as error:
        logger.warning("Ignoring unreadable dataset metadata %s: %s", sidecar, error)
        return None
    return metadata if isinstance(metadata, dict) else None


def input_record(path: Path) -> dict[str, Any]:
    """Describe one input file for provenance: name, checksum, and its metadata (None for user-provided files).

    The checksum recorded at download time is reused, so large inputs are not re-hashed on every run.
    """
    metadata = read_source_metadata(path)
    recorded = None if metadata is None else metadata.get("sha256")
    return {
        "file": path.name,
        "sha256": recorded if isinstance(recorded, str) else file_digest(path),
        "metadata": metadata,
    }


def data_credits(records: Iterable[Mapping[str, Any] | None], extra_sources: Iterable[str] = ()) -> str:
    """Return the data credits line for the sources behind the given input records (plus ``extra_sources``).

    Each source is credited at the version recorded when its file was downloaded; user-provided files carry no
    metadata and are not credited.
    """
    used: dict[str, str | None] = {}
    for key in extra_sources:
        used[key] = None
    pending = [record for record in records if record is not None]
    while pending:
        metadata = pending.pop().get("metadata") or {}
        key, version = metadata.get("source"), metadata.get("version")
        if isinstance(key, str) and key in SOURCES:
            used[key] = version if isinstance(version, str) else None
        pending.extend(component for component in metadata.get("derived_from", []) if isinstance(component, dict))
    parts = [source.credit(used[key]) for key, source in SOURCES.items() if key in used]
    return f"Data: {'; '.join(parts)}." if parts else ""
