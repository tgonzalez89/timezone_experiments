"""Cache fingerprints of source files, package versions, and JSON payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING

from timezone_experiments.core import fingerprint
from timezone_experiments.core.fingerprint import json_digest, source_fingerprint

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_changing_a_source_file_changes_the_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fingerprint, "_PACKAGE_ROOT", tmp_path)
    source = tmp_path / "module.py"
    source.write_bytes(b"VALUE = 1\n")
    before = json_digest(source_fingerprint(["module.py"]))
    assert json_digest(source_fingerprint(["module.py"])) == before
    source.write_bytes(b"VALUE = 2\n")
    assert json_digest(source_fingerprint(["module.py"])) != before


def test_package_versions_are_part_of_the_fingerprint() -> None:
    packages = source_fingerprint([], ["numpy", "not-an-installed-distribution"])["packages"]
    assert isinstance(packages, dict)
    assert set(packages) == {"numpy", "not-an-installed-distribution"}
    assert packages["numpy"]
    assert packages["not-an-installed-distribution"] == ""


def test_json_digest_ignores_key_order() -> None:
    assert json_digest({"a": 1, "b": [1, 2]}) == json_digest({"b": [1, 2], "a": 1})
    assert json_digest({"a": 1}) != json_digest({"a": 2})
