"""The scripts in tools/ (DECISIONS.md §14, §18).

They mirror main() and import from the demonstrator, so a rename there breaks
them without any test noticing. Importing them catches that in milliseconds;
running them needs the checkpoint and is left to their own use.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.support import ROOT


@pytest.mark.parametrize("name", ["benchmark", "make_screenshot", "make_sample"])
def test_tool_imports(name: str) -> None:
    path: Path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tools_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
