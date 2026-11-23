"""Errors shown to users as messages rather than tracebacks."""

from __future__ import annotations

import sqlite3
from zipfile import BadZipFile

# Failures caused by inputs, configuration, the network, or the disk. Anything else is a bug and should
# surface with a traceback.
USER_ERRORS: tuple[type[Exception], ...] = (OSError, ValueError, BadZipFile, sqlite3.Error)


def describe_error(error: BaseException) -> str:
    """Return an error's message followed by any context notes (such as the failing territory)."""
    return "\n".join([str(error), *getattr(error, "__notes__", ())])
