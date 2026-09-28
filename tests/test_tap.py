"""Feature extraction and the tile grid (DECISIONS.md §2, §3, §3.1).

Runs on the yolo26n architecture with random weights. That checks the module
chain and the plumbing; the real checkpoint is checked in test_e2e.py.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn
from ultralytics import YOLO

from feature_viz import demonstrator as demo
from tests.support import EXPECTED_TARGETS, forward


def tensors(out: object) -> Iterator[torch.Tensor]:
    """Every tensor in a nested model output, in a stable order."""
    if isinstance(out, torch.Tensor):
        yield out
    elif isinstance(out, dict):
        for key in sorted(out):
            yield from tensors(out[key])
    elif isinstance(out, (list, tuple)):
        for item in out:
            yield from tensors(item)


def hook_count(model: YOLO) -> int:
    net: nn.Module = cast(nn.Module, model.model)
    return sum(len(m._forward_hooks) for m in net.modules())


# --------------------------------------------------------------------------
# Layer selection (§3)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("profile", ["cpu_cfg", "gpu_cfg"])
def test_targets_resolve_to_the_documented_block_types(
    yaml_model: YOLO, profile: str, request: pytest.FixtureRequest
) -> None:
    cfg: demo.Config = request.getfixturevalue(profile)
    tap: demo.FeatureTap = demo.FeatureTap(yaml_model, cfg.targets)
    try:
        assert {i: tap.label(i) for i in tap.targets} == {
            i: EXPECTED_TARGETS[i] for i in cfg.targets
        }
    finally:
        tap.close()


def test_detect_head_is_not_a_target(yaml_model: YOLO, gpu_cfg: demo.Config) -> None:
    tap: demo.FeatureTap = demo.FeatureTap(yaml_model, [])
    last: int = len(tap.layers) - 1
    assert tap.label(last) == "Detect"
    assert last not in gpu_cfg.targets


def test_out_of_range_targets_are_dropped_with_a_warning(
    yaml_model: YOLO, capsys: pytest.CaptureFixture[str]
) -> None:
    tap: demo.FeatureTap = demo.FeatureTap(yaml_model, [4, 99, -1])
    try:
        assert tap.targets == [4]
        assert "[99, -1]" in capsys.readouterr().err
    finally:
        tap.close()


def test_dump_marks_exactly_the_targets(
    tap: demo.FeatureTap, capsys: pytest.CaptureFixture[str]
) -> None:
    tap.dump_structure()
    marked: list[int] = [
        int(line.split()[0]) for line in capsys.readouterr().out.splitlines() if "<--" in line
    ]
    assert marked == tap.targets


# --------------------------------------------------------------------------
# Hooks leave the model alone (§2)
# --------------------------------------------------------------------------
def test_tap_does_not_change_the_model_output(yaml_model: YOLO, cpu_cfg: demo.Config) -> None:
    """§2: extraction observes, it never alters the forward pass."""
    before: list[torch.Tensor] = list(tensors(forward(yaml_model)))
    tap: demo.FeatureTap = demo.FeatureTap(yaml_model, cpu_cfg.targets)
    try:
        after: list[torch.Tensor] = list(tensors(forward(yaml_model)))
    finally:
        tap.close()
    assert before and len(before) == len(after)
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_close_removes_every_hook(yaml_model: YOLO, gpu_cfg: demo.Config) -> None:
    baseline: int = hook_count(yaml_model)
    tap: demo.FeatureTap = demo.FeatureTap(yaml_model, gpu_cfg.targets)
    assert hook_count(yaml_model) == baseline + len(gpu_cfg.targets)
    tap.close()
    assert hook_count(yaml_model) == baseline


def test_hooks_capture_detached_activations(tap: demo.FeatureTap, yaml_model: YOLO) -> None:
    forward(yaml_model)
    assert sorted(tap.activations) == tap.targets
    for t in tap.activations.values():
        assert t.ndim == 4 and not t.requires_grad


# --------------------------------------------------------------------------
# Grid (§3.1)
# --------------------------------------------------------------------------
def test_render_produces_one_uint8_bgr_mosaic(
    tap: demo.FeatureTap, yaml_model: YOLO, cpu_cfg: demo.Config
) -> None:
    forward(yaml_model)
    renderer: demo.GridRenderer = demo.GridRenderer(cpu_cfg, tap)
    renderer.calibrate()
    grid: demo.BGRImage = renderer.render()
    assert grid.dtype == np.uint8 and grid.ndim == 3 and grid.shape[2] == 3
    # Same input, same grid: nothing random in the render path.
    assert np.array_equal(renderer.render(), grid)


def test_calibrate_caps_channels_at_max_channels(
    tap: demo.FeatureTap, yaml_model: YOLO, cpu_cfg: demo.Config
) -> None:
    """§3.1: at most max_channels tiles per layer, chosen by energy."""
    forward(yaml_model)
    renderer: demo.GridRenderer = demo.GridRenderer(cpu_cfg, tap)
    renderer.calibrate()
    for i, chosen in renderer.channels.items():
        available: int = int(tap.activations[i].shape[1])
        assert len(chosen) == min(cpu_cfg.max_channels, available)
        assert len(set(chosen.tolist())) == len(chosen)


def test_channel_selection_is_fixed_after_calibration(
    tap: demo.FeatureTap, yaml_model: YOLO, cpu_cfg: demo.Config
) -> None:
    """calibrate() runs once; later frames must not reshuffle the tiles."""
    forward(yaml_model, seed=0)
    renderer: demo.GridRenderer = demo.GridRenderer(cpu_cfg, tap)
    renderer.calibrate()
    chosen: dict[int, list[int]] = {i: c.tolist() for i, c in renderer.channels.items()}
    forward(yaml_model, seed=1)
    renderer.render()
    assert {i: c.tolist() for i, c in renderer.channels.items()} == chosen


def test_grid_pads_incomplete_panels_and_rows(yaml_model: YOLO, cpu_cfg: demo.Config) -> None:
    """Neither padding path runs with today's profiles: every target has at
    least 64 channels (an 8x8 panel) and fills its row. A change of
    max_channels or targets reaches them, and must not break the grid:
    10 tiles in a 4x3 panel, 4 panels in rows of 3."""
    cpu_cfg.max_channels = 10
    cpu_cfg.targets = [2, 4, 9, 16]
    tap: demo.FeatureTap = demo.FeatureTap(yaml_model, cpu_cfg.targets)
    try:
        forward(yaml_model)
        renderer: demo.GridRenderer = demo.GridRenderer(cpu_cfg, tap)
        renderer.calibrate()
        panel: demo.BGRImage = renderer.panel(2)
        assert panel.shape[1] == 4 * cpu_cfg.tile
        grid: demo.BGRImage = renderer.render()
        # Two rows of three equal cells: the empty cell is padded, not dropped.
        cell_w: int = (grid.shape[1] - 2 * 8) // 3
        assert grid.shape[1] == 3 * cell_w + 2 * 8
        assert grid.shape[0] % 2 == 0
    finally:
        tap.close()


@pytest.mark.gpu
def test_same_code_path_on_cuda(yaml_model: YOLO, gpu_cfg: demo.Config) -> None:
    """§5: the GPU profile runs the identical tap and renderer."""
    net: nn.Module = cast(nn.Module, yaml_model.model)
    net.to("cuda")
    tap: demo.FeatureTap = demo.FeatureTap(yaml_model, gpu_cfg.targets)
    try:
        with torch.no_grad():
            net(torch.rand(1, 3, 160, 160, device="cuda"))
        renderer: demo.GridRenderer = demo.GridRenderer(gpu_cfg, tap)
        renderer.calibrate()
        grid: demo.BGRImage = renderer.render()
        assert grid.dtype == np.uint8 and grid.shape[2] == 3
    finally:
        tap.close()
        net.to("cpu")
