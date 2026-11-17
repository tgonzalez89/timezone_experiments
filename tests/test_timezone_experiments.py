"""Tests for timezone_experiments package."""

from __future__ import annotations

import timezone_experiments


def test_version_is_string() -> None:
    assert isinstance(timezone_experiments.__version__, str)


def test_version_has_three_parts() -> None:
    parts = timezone_experiments.__version__.split(".")
    assert len(parts) == 3
