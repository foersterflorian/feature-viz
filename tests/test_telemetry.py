"""ultralytics' telemetry stays off, whatever the entry point (DECISIONS.md §19).

feature_viz sets YOLO_OFFLINE=1 in its package __init__; ultralytics reads it
once, at its first import. The failure mode is an import order in which
ultralytics comes first - silent, since nothing else changes. Each check runs
in a fresh interpreter that pretends DNS works, so ultralytics would consider
itself online if the switch arrived too late; without that, an offline test
machine would pass for the wrong reason.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.support import ROOT

PRETEND_ONLINE: str = "import socket; socket.getaddrinfo = lambda *a, **k: [('fake',)]\n"
REPORT: str = "\nfrom ultralytics.utils import ONLINE; print(ONLINE)"


def load_tool(name: str) -> str:
    """Executes a tool's module body (its imports), not its main()."""
    path: str = str(ROOT / "tools" / f"{name}.py")
    return (
        "import importlib.util as u\n"
        f"s = u.spec_from_file_location('t', {path!r})\n"
        "s.loader.exec_module(u.module_from_spec(s))"
    )


ENTRY_POINTS: dict[str, str] = {
    "demonstrator": "import feature_viz.demonstrator",
    "benchmark": load_tool("benchmark"),
    "make_screenshot": load_tool("make_screenshot"),
}


def ultralytics_online(entry: str, **env: str) -> str:
    # The test process has imported feature_viz itself, so YOLO_OFFLINE is
    # already in os.environ; the child must not inherit it.
    base: dict[str, str] = {k: v for k, v in os.environ.items() if k != "YOLO_OFFLINE"}
    proc: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", PRETEND_ONLINE + entry + REPORT],
        env={**base, **env},
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("entry", ENTRY_POINTS.values(), ids=ENTRY_POINTS.keys())
def test_ultralytics_starts_offline(entry: str) -> None:
    assert ultralytics_online(entry) == "False"


def test_explicit_opt_in_is_respected() -> None:
    """Also the control for the test above: with the switch off, the faked
    DNS does make ultralytics believe it is online."""
    assert ultralytics_online(ENTRY_POINTS["demonstrator"], YOLO_OFFLINE="0") == "True"
