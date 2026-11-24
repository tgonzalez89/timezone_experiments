"""Crash-safe file writes and dataset reads with actionable errors."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def atomic_path(path: Path) -> Generator[Path]:
    """Yield a unique temporary sibling of ``path`` and move it into place only if the block succeeds.

    Readers never see a half-written file, an interrupted write leaves the previous file untouched, and
    concurrent writers (CLI and dashboard) each get their own temporary file instead of colliding.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text so the target is either the old or the complete new content."""
    with atomic_path(path) as temporary:
        temporary.write_text(text, encoding="utf-8")


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write bytes so the target is either the old or the complete new content."""
    with atomic_path(path) as temporary:
        temporary.write_bytes(data)


def read_json_dataset(path: Path) -> object:
    """Decode a JSON/GeoJSON dataset (callers validate its structure), naming the file and the fix if corrupt."""
    try:
        with path.open(encoding="utf-8-sig") as file:
            return json.load(file)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = (
            f"Corrupt or unreadable dataset {path}: {error}. "
            "If it was downloaded, run 'download --refresh' (or delete it) to fetch it again."
        )
        raise ValueError(msg) from error
