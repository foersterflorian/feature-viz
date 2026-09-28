"""Measure the demonstrator's frame rate. Prints numbers; asserts nothing.

Frame rate depends on the machine, so it is measured here rather than tested
(`DECISIONS.md` §18). This mirrors the body of `main()` and reuses
`FeatureTap`, `GridRenderer` and `compose`, exactly as the harness behind
§14 did. The JPEG encode stays in the loop because it is part of every real
frame (10.4 ms in §14); only the HTTP hand-over is left out.

The profile comes from `build_config()`, so the usual environment variables
apply (`FORCE_CPU=1`, `WEIGHTS=...`). The source defaults to the sample clip;
a video is used deliberately, since a webcam would measure its own 30 Hz.

    python tools/benchmark.py                   # GPU profile if available
    FORCE_CPU=1 python tools/benchmark.py       # reduced profile
    python tools/benchmark.py --runs 3          # repeat to see the spread
    python tools/benchmark.py --no-funding-strip   # the §14.2 comparison

Paste the output into §14 together with the machine line it prints.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from feature_viz import demonstrator as demo

ROOT: Path = Path(__file__).resolve().parent.parent
SAMPLE: Path = ROOT / "assets" / "sample.mp4"


def machine() -> str:
    gpu: str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    return (
        f"{gpu} | {platform.processor() or platform.machine()} ({os.cpu_count()} threads) | "
        f"torch {torch.__version__} | ultralytics {ultralytics.__version__} | "
        f"OpenCV {cv2.__version__}"
    )


def run(cfg: demo.Config, warmup: int) -> tuple[list[float], tuple[int, int]]:
    """One pass over the source. Returns per-frame loop times after warm-up
    and the canvas size."""
    model: YOLO = YOLO(cfg.weights)
    model.to(cfg.device)
    tap: demo.FeatureTap = demo.FeatureTap(model, cfg.targets)
    renderer: demo.GridRenderer = demo.GridRenderer(cfg, tap)
    info: str = f"{cfg.weights} | {cfg.device.upper()} | {cfg.imgsz}px | layers {cfg.targets}"
    stream: Iterator = cast(
        "Iterator",
        model.predict(
            source=cfg.source, imgsz=cfg.imgsz, stream=True, verbose=False, device=cfg.device
        ),
    )

    times: list[float] = []
    grid: demo.BGRImage | None = None
    canvas: demo.BGRImage | None = None
    n: int = 0
    t_prev: float = time.perf_counter()
    try:
        for result in stream:
            if not renderer.channels:
                renderer.calibrate()
            if n % cfg.vis_every == 0 or grid is None:
                grid = renderer.render()
            canvas = demo.compose(result.plot(), grid, 30.0, info)
            cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality])
            now: float = time.perf_counter()
            if n >= warmup:
                times.append(now - t_prev)
            t_prev = now
            n += 1
    finally:
        tap.close()
    size: tuple[int, int] = (0, 0) if canvas is None else (canvas.shape[1], canvas.shape[0])
    return times, size


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0]
    )
    parser.add_argument("--runs", type=int, default=1, help="passes over the source")
    parser.add_argument("--warmup", type=int, default=60, help="frames discarded per pass")
    parser.add_argument(
        "--no-funding-strip",
        action="store_true",
        help="measure without the §17 strip, for comparison only",
    )
    args: argparse.Namespace = parser.parse_args()

    cfg: demo.Config = demo.build_config()
    if not os.getenv("SOURCE"):
        cfg.source = str(SAMPLE)
    if not Path(cfg.source).exists() and not cfg.source.isdigit():
        print(f"[error] source not found: {cfg.source}", file=sys.stderr)
        return 1
    if args.no_funding_strip:
        # Comparison only: a zero-height strip leaves compose() untouched.
        demo.funding_strip = lambda width: np.zeros((0, width, 3), dtype=np.uint8)  # type: ignore[assignment]

    print(f"[info] {machine()}")
    print(f"[info] {cfg.weights} | {cfg.device} | {cfg.imgsz}px | layers {cfg.targets}")
    print(f"[info] source {cfg.source}, {args.warmup} warm-up frames per run")
    for i in range(args.runs):
        times, (w, h) = run(cfg, args.warmup)
        if not times:
            print(f"[error] source shorter than the warm-up ({args.warmup})", file=sys.stderr)
            return 1
        fps: list[float] = sorted(1.0 / t for t in times)
        p5: float = fps[int(0.05 * len(fps))]
        print(
            f"run {i + 1}: canvas {w}x{h}, {len(times)} frames | "
            f"FPS mean {1.0 / statistics.mean(times):.1f} / "
            f"median {statistics.median(fps):.1f} / p5 {p5:.1f} | "
            f"loop {1000.0 * statistics.mean(times):.1f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
