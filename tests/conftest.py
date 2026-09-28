"""Shared fixtures. The default suite needs no weights, no GPU and no network:
the model is built from the architecture YAML that ships inside ultralytics,
which yields the same module chain as `yolo26n.pt` with random weights.
See DECISIONS.md §18.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest
import torch
from torch import nn
from ultralytics import YOLO

from feature_viz import demonstrator as demo

from tests.support import ENV_VARS, WEIGHTS


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip, rather than fail, what this machine cannot run."""
    no_weights = pytest.mark.skip(reason=f"checkpoint not found: {WEIGHTS}")
    no_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "weights" in item.keywords and not WEIGHTS.is_file():
            item.add_marker(no_weights)
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(no_gpu)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def cpu_cfg(monkeypatch: pytest.MonkeyPatch) -> demo.Config:
    """The reduced profile, regardless of the machine the tests run on."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return demo.build_config()


@pytest.fixture
def gpu_cfg(monkeypatch: pytest.MonkeyPatch) -> demo.Config:
    """The presentation profile. Building it touches no GPU, so this works
    on any machine; only running a model on it needs CUDA."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return demo.build_config()


@pytest.fixture(scope="session")
def yaml_model() -> YOLO:
    """yolo26n architecture with random weights, built offline in about 1 s."""
    model: YOLO = YOLO("yolo26n.yaml")
    cast(nn.Module, model.model).eval()
    return model


@pytest.fixture
def tap(yaml_model: YOLO, cpu_cfg: demo.Config) -> Iterator[demo.FeatureTap]:
    """A tap on the session model. Closed afterwards, so that hooks never leak
    from one test into the next."""
    t: demo.FeatureTap = demo.FeatureTap(yaml_model, cpu_cfg.targets)
    yield t
    t.close()
