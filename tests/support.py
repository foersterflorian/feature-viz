"""Constants and helpers shared by the tests. Kept out of conftest.py, which
pytest loads as a plugin and which is not meant to be imported from."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn
from ultralytics import YOLO

from feature_viz import demonstrator as demo

ROOT: Path = Path(__file__).resolve().parent.parent

# Every variable build_config() reads. Cleared per test so that a developer's
# shell (e.g. FORCE_CPU=1 left exported) cannot change what a test sees.
ENV_VARS: tuple[str, ...] = ("FORCE_CPU", "WEIGHTS", "SOURCE", "PORT", "DISPLAY_MODE")

# Where the real checkpoint is looked for by the `weights` tests: $WEIGHTS if
# set, otherwise the repository root, which is where a bare filename lands
# (DECISIONS.md §11).
WEIGHTS: Path = Path(os.getenv("WEIGHTS", str(ROOT / "yolo26n.pt"))).resolve()

# Block type per target index, as verified by DUMP_STRUCTURE (DECISIONS.md §3).
EXPECTED_TARGETS: dict[int, str] = {
    2: "C3k2",
    4: "C3k2",
    9: "SPPF",
    10: "C2PSA",
    16: "C3k2",
    22: "C3k2",
}


def forward(model: YOLO, size: int = 160, seed: int = 0) -> object:
    """One forward pass on a fixed random image; fires any registered hooks."""
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    x: torch.Tensor = torch.rand(1, 3, size, size, generator=gen)
    net: nn.Module = cast(nn.Module, model.model)
    with torch.no_grad():
        return net(x)


def bgr(h: int, w: int, value: int = 0) -> demo.BGRImage:
    return np.full((h, w, 3), value, dtype=np.uint8)
