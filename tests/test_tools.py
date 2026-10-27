"""The scripts and host files in tools/ (DECISIONS.md §9, §14, §18).

They mirror main() and import from the demonstrator, so a rename there breaks
them without any test noticing. Importing them catches that in milliseconds;
running them needs the checkpoint and is left to their own use.
"""

from __future__ import annotations

import importlib.util
import re
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


def test_camera_unit_starts_the_container_compose_creates() -> None:
    """§9: the udev rule wants the unit, and the unit starts the container
    by the name Compose gives it - project, service, replica. Renaming the
    project or the service in compose.yaml would otherwise leave the unit
    starting a container that no longer exists, unnoticed until a camera
    drops off the bus on the stand."""
    compose: str = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    project = re.search(r"^name: (\S+)$", compose, re.MULTILINE)
    service = re.search(r"^services:\n  (\S+):$", compose, re.MULTILINE)
    assert project is not None and service is not None
    container: str = f"{project.group(1)}-{service.group(1)}-1"

    unit_file: Path = ROOT / "tools" / "systemd" / "feature-viz-camera.service"
    unit: str = unit_file.read_text(encoding="utf-8")
    assert f"ExecStart=/usr/bin/docker start {container}\n" in unit

    rule: str = (ROOT / "tools" / "udev" / "70-feature-viz-camera.rules").read_text()
    assert f'ENV{{SYSTEMD_WANTS}}+="{unit_file.name}"' in rule
    assert 'ENV{ID_V4L_CAPABILITIES}==":capture:"' in rule
