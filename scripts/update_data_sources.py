"""Report newer releases of the pinned datasets and, with ``--write``, update the pins in sources.json.

A developer tool; the package never runs it. Usage::

    uv run python scripts/update_data_sources.py            # report only
    uv run python scripts/update_data_sources.py --write    # also rewrite the pinned versions

Before a new version is written, its download URL is checked to exist, so a provider that changed its file layout
is reported instead of pinned to a broken URL. Exit status: 0 when every pin is current (or was updated), 1 when
updates are available but not written or any check failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from timezone_experiments.data.download.catalog import SOURCES_PATH, DataSource, load_sources
from timezone_experiments.data.download.http import DownloadError, url_exists
from timezone_experiments.data.download.versions import PROBE_FIELDS, latest_version


def main(argv: list[str] | None = None) -> int:
    """Check every pinned dataset, print one line per dataset, and optionally write the newer versions."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="update sources.json with the newer versions")
    parser.add_argument("--sources", type=Path, default=SOURCES_PATH, help="pin file to check (default: %(default)s)")
    args = parser.parse_args(argv)
    updates: dict[str, str] = {}
    failed = False
    for key, source in load_sources(args.sources).items():
        status, latest, check_failed = _check(source)
        print(f"{source.name}: {status}")
        failed |= check_failed
        if latest is not None:
            updates[key] = latest
    if not updates:
        return int(failed)
    if not args.write:
        print("Newer releases are available; run with --write to update the pins.")
        return 1
    pins = json.loads(args.sources.read_text(encoding="utf-8"))
    for key, version in updates.items():
        pins[key]["version"] = version
    args.sources.write_text(json.dumps(pins, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Updated {args.sources}. The next run downloads the new releases; review the changes before committing.")
    return int(failed)


def _check(source: DataSource) -> tuple[str, str | None, bool]:
    """Return a dataset's status line, the version to update to (None: nothing to write), and whether it failed."""
    try:
        latest = latest_version(source)
        if latest is None:
            return "not pinnable (the provider only serves its current build)", None, False
        if latest == source.version:
            return f"up to date ({source.version})", None, False
        url = source.resolve(latest, **PROBE_FIELDS)
        if not url_exists(url):
            return f"check failed: {latest} is released but {url} does not exist; update the URL by hand", None, True
    except (DownloadError, ValueError) as error:
        return f"check failed ({error})", None, True
    return f"newer release {latest} (pinned: {source.version})", latest, False


if __name__ == "__main__":
    sys.exit(main())
