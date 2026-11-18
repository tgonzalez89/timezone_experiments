"""Verify terminal progress, redirected snapshots, and interrupted transfers."""

from __future__ import annotations

import io
import os
from typing import override

import pytest

from timezone_experiments import progress as console


class Terminal(io.StringIO):
    @override
    def isatty(self) -> bool:
        return True


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [0.0]
    monkeypatch.setattr(console.time, "monotonic", lambda: now[0])
    return now


def test_terminal_updates_are_throttled_and_finish_on_one_line(
    monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    output = Terminal()
    monkeypatch.setattr(console.sys, "stderr", output)
    monkeypatch.setattr(console.shutil, "get_terminal_size", lambda **kwargs: os.terminal_size((180, 24)))
    with console.Progress("Optimizing", 3) as progress:
        progress.update(1, "too soon")
        clock[0] = 0.25
        progress.update(2, "finished TST:2")
        progress.update(3, "finished TST:3")
    text = output.getvalue()
    assert text.count("\n") == 1
    assert text.count("\r") == 3
    assert "0/3" in text
    assert "2/3" in text
    assert "3/3 (100.0%; 0 left)" in text
    assert "finished TST:3" in text
    assert "too soon" not in text
    assert "elapsed" in text
    assert "ETA" in text


def test_redirected_transfers_emit_only_start_and_end_for_unknown_totals(
    capsys: pytest.CaptureFixture[str], clock: list[float]
) -> None:
    with console.Progress("Downloading", None, unit="bytes") as progress:
        clock[0] = 4
        progress.update(1024 * 1024)
        clock[0] = 6
        progress.update(2 * 1024 * 1024)
        clock[0] = 6.1
        progress.update(3 * 1024 * 1024)
    text = capsys.readouterr().err
    assert "\r" not in text
    assert len(text.splitlines()) == 2
    assert "1.0 MiB" not in text
    assert "2.0 MiB" not in text
    assert "3.0 MiB (total unknown)" in text
    assert "ETA ?" in text


def test_redirected_optimization_has_no_per_item_lines(capsys: pytest.CaptureFixture[str], clock: list[float]) -> None:
    with console.Progress("Optimizing", 2) as progress:
        clock[0] = 10
        progress.update(1, "TST:1")
        clock[0] = 20
        progress.update(2, "TST:2")
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    assert "0/2" in lines[0]
    assert "2/2" in lines[1]
    assert "TST:1" not in "".join(lines)


def test_incomplete_transfer_reports_failure_without_claiming_completion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = io.BytesIO()
    with pytest.raises(OSError, match="expected 10 bytes, received 7"):
        console.copy_with_progress(io.BytesIO(b"partial"), target, label="Downloading", total=10)
    text = capsys.readouterr().err
    assert "[failed]: 7.0 B/10.0 B (70.0%" in text
    assert "100.0%" not in text
    assert text.endswith("\n")
