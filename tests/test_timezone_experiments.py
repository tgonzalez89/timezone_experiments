"""Tests for your_actual_project_name package."""

from __future__ import annotations

import your_actual_project_name


def test_version_is_string() -> None:
    assert isinstance(your_actual_project_name.__version__, str)


def test_version_has_three_parts() -> None:
    parts = your_actual_project_name.__version__.split(".")
    assert len(parts) == 3
