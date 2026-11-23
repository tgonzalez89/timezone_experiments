"""Dependency-free, throttled console progress and binary stream copying."""

from __future__ import annotations

import shutil
import sys
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from types import TracebackType
    from typing import IO

BYTES_PER_KIB = 1024


class Progress:
    """Update one terminal line, showing only start/end states when redirected by default."""

    def __init__(
        self,
        label: str,
        total: int | None,
        *,
        unit: str = "items",
        enabled: bool = True,
        log_interval: float | None = None,
    ) -> None:
        """Create a counter; set ``log_interval`` to opt into redirected snapshots."""
        if total is not None and total < 0:
            msg = "Progress total cannot be negative."
            raise ValueError(msg)
        self.label = label
        self.total = total
        self.unit = unit
        self.enabled = enabled
        self.log_interval = log_interval
        self.completed = 0
        self.detail = ""
        self._stream = sys.stderr
        self._terminal = self._stream.isatty()
        self._started = time.monotonic()
        self._last_render = self._started
        self._width = 0

    def __enter__(self) -> Self:
        """Display the initial state."""
        self._render()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """End the line, retaining the actual completed count on failure."""
        self._render(final=True, failed=exception_type is not None)

    def update(self, completed: int, detail: str = "") -> None:
        """Record completed work; redraw no more often than the display interval."""
        self.completed = completed
        self.detail = detail
        interval = 0.2 if self._terminal else self.log_interval
        if interval is not None and time.monotonic() - self._last_render >= interval:
            self._render()

    def _render(self, *, final: bool = False, failed: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        elapsed = now - self._started
        count = self._amount(self.completed)
        if self.total is None:
            count += f" {self.unit} (total unknown)" if self.unit != "bytes" else " (total unknown)"
        else:
            percent = self.completed / self.total * 100 if self.total else 100.0
            count += f"/{self._amount(self.total)} ({percent:.1f}%; {self._amount(self.total - self.completed)} left)"
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.completed) / rate if self.total is not None and rate > 0 else None
        timing = f"elapsed {_duration(elapsed)} | ETA {_duration(eta) if eta is not None else '?'}"
        if rate > 0:
            timing += f" | {self._amount(rate)}/s"
        detail = f" | {self.detail}" if self.detail else ""
        status = " [failed]" if failed else ""
        text = f"{self.label}{status}: {count}{detail} | {timing}"
        if self._terminal:
            columns = shutil.get_terminal_size(fallback=(120, 24)).columns
            text = text[: max(1, columns - 1)]
            self._stream.write("\r" + text.ljust(self._width) + ("\n" if final else ""))
            self._stream.flush()
            self._width = len(text)
        else:
            print(text, file=self._stream, flush=True)
        self._last_render = now

    def _amount(self, value: float) -> str:
        if self.unit != "bytes":
            return f"{value:,.0f}"
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        for unit in units[:-1]:
            if abs(value) < BYTES_PER_KIB:
                return f"{value:.1f} {unit}"
            value /= BYTES_PER_KIB
        return f"{value:.1f} {units[-1]}"


def _duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, round(seconds))))


def copy_with_progress(source: IO[bytes], target: IO[bytes], *, label: str, total: int | None) -> None:
    """Copy bytes with throughput reporting and detect incomplete transfers."""
    completed = 0
    with Progress(label, total, unit="bytes") as progress:
        while True:
            block = source.read(1024 * 1024)
            if not isinstance(block, bytes):
                msg = "Binary input stream returned non-binary data."
                raise TypeError(msg)
            if not block:
                break
            target.write(block)
            completed += len(block)
            progress.update(completed)
        if total is not None and completed != total:
            msg = f"Incomplete transfer ({label}): expected {total} bytes, received {completed}."
            raise ConnectionError(msg)
