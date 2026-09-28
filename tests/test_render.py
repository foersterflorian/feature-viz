"""Normalisation and canvas layout (DECISIONS.md §4, §15, §17)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_viz import demonstrator as demo
from tests.support import bgr

LINES: list[demo.CaptionLine] = [
    ("12.3 FPS", 1.1, (120, 255, 120)),
    ("yolo26n.pt | CPU | 416px", 0.75, (210, 210, 210)),
]


# --------------------------------------------------------------------------
# Scale (§4)
# --------------------------------------------------------------------------
def test_scale_starts_on_the_first_frame_instead_of_easing_in(cpu_cfg: demo.Config) -> None:
    """§8 relies on this: a single still is correctly normalised."""
    t: torch.Tensor = torch.linspace(0.0, 10.0, 1001)
    lo, hi = demo.Scale(cpu_cfg).get(t)
    q_lo, q_hi = cpu_cfg.quantiles
    assert lo == pytest.approx(torch.quantile(t, q_lo).item())
    assert hi == pytest.approx(torch.quantile(t, q_hi).item())


def test_scale_recomputes_only_every_update_every_frames(cpu_cfg: demo.Config) -> None:
    scale: demo.Scale = demo.Scale(cpu_cfg)
    first: tuple[float, float] = scale.get(torch.zeros(100) + torch.arange(100.0))
    for _ in range(cpu_cfg.update_every - 1):
        assert scale.get(torch.full((100,), 1e6)) == first
    assert scale.get(torch.full((100,), 1e6)) != first


def test_scale_follows_by_ema(cpu_cfg: demo.Config) -> None:
    cpu_cfg.update_every = 1
    scale: demo.Scale = demo.Scale(cpu_cfg)
    lo0, hi0 = scale.get(torch.arange(100.0))
    lo1, hi1 = scale.get(torch.arange(100.0) + 100.0)
    assert lo1 == pytest.approx(lo0 + cpu_cfg.alpha * 100.0)
    assert hi1 == pytest.approx(hi0 + cpu_cfg.alpha * 100.0)


def test_scale_never_divides_by_zero(cpu_cfg: demo.Config) -> None:
    lo, hi = demo.Scale(cpu_cfg).get(torch.full((50,), 3.0))
    assert hi > lo


def test_scale_accepts_float16(cpu_cfg: demo.Config) -> None:
    """§10: torch.quantile rejects float16 on CUDA; Scale casts first."""
    t: torch.Tensor = torch.linspace(0.0, 1.0, 500)
    assert demo.Scale(cpu_cfg).get(t.half()) == pytest.approx(
        demo.Scale(cpu_cfg).get(t), abs=1e-3
    )


# --------------------------------------------------------------------------
# Captions and compose (§15)
# --------------------------------------------------------------------------
def test_caption_stacks_a_strip_above_and_covers_nothing() -> None:
    """§15: the text goes into a strip above the image, never onto it. That
    the strip costs no canvas area is compose()'s job and tested there."""
    img: demo.BGRImage = bgr(100, 300, value=77)
    out: demo.BGRImage = demo.caption(img, LINES)
    assert out.shape[1] == img.shape[1] and out.shape[0] > img.shape[0]
    assert np.array_equal(out[-100:], img)


def test_compose_does_not_grow_the_canvas_for_the_caption() -> None:
    """§15: compose() shrinks the frame by the caption height, so the strip
    costs no canvas area - growing the canvas once cost 6 FPS. With the grid
    taller than frame plus caption, the grid must set the height and appear
    unscaled, pixel for pixel."""
    frame: demo.BGRImage = bgr(360, 640, value=90)
    grid: demo.BGRImage = np.random.default_rng(0).integers(
        0, 256, (900, 1500, 3), dtype=np.uint8
    )
    canvas: demo.BGRImage = demo.compose(frame, grid, 30.0, "info")
    strip_h: int = demo.funding_strip(canvas.shape[1]).shape[0]
    assert canvas.shape[0] == grid.shape[0] + strip_h
    assert np.array_equal(canvas[: grid.shape[0], -grid.shape[1] :], grid)


def test_compose_puts_the_funding_strip_at_the_bottom() -> None:
    """§17: the notice is part of every canvas."""
    canvas: demo.BGRImage = demo.compose(bgr(360, 640), bgr(400, 800), 30.0, "info")
    strip: demo.BGRImage = demo.funding_strip(canvas.shape[1])
    assert np.array_equal(canvas[-strip.shape[0] :], strip)
