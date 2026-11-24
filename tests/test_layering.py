"""Keep the package layered: each subpackage imports only from the layers below it (or from itself)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = "timezone_experiments"
SOURCE = Path(__file__).parents[1] / "src" / PACKAGE
# Lowest first. Packages of equal rank are peers and must not import each other.
LAYERS = {
    "core": 0,
    "astronomy": 1,
    "data": 2,
    "optimization": 2,
    "experiments": 3,
    "maps": 4,
    "app": 5,
}


def _imported_packages(path: Path) -> set[str]:
    """Return the subpackages a module imports from, including imports used only for type checking."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            # ``from timezone_experiments import maps`` names the subpackage in the imported names.
            modules = (
                [f"{node.module}.{alias.name}" for alias in node.names] if node.module == PACKAGE else [node.module]
            )
        else:
            continue
        for module in modules:
            parts = module.split(".")
            if parts[0] == PACKAGE and len(parts) > 1 and parts[1] in LAYERS:
                imported.add(parts[1])
    return imported


def test_every_subpackage_has_a_layer() -> None:
    packages = {path.parent.name for path in SOURCE.glob("*/__init__.py")}
    assert packages == set(LAYERS)
    assert not [path.name for path in SOURCE.glob("*.py") if path.name != "__init__.py"]


@pytest.mark.parametrize("path", sorted(SOURCE.glob("*/**/*.py")), ids=lambda path: path.relative_to(SOURCE).as_posix())
def test_modules_import_only_from_lower_layers(path: Path) -> None:
    package = path.relative_to(SOURCE).parts[0]
    upward = {
        imported for imported in _imported_packages(path) if imported != package and LAYERS[imported] >= LAYERS[package]
    }
    assert not upward, f"{package} (layer {LAYERS[package]}) must not import {sorted(upward)}"
