"""User-facing error descriptions."""

from __future__ import annotations

from timezone_experiments.app.errors import describe_error


def test_describe_error_appends_notes() -> None:
    error = ValueError("broken")
    error.add_note("While doing something.")
    assert describe_error(error) == "broken\nWhile doing something."
