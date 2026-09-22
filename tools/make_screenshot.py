"""Render the still images used in talks, programmes and handover material.

Writes `docs/demo-full.png` (the whole canvas) and `docs/demo-crop.png` (the
detection frame plus the first column of feature-map panels, at roughly 2:1 -
the full canvas is 3.1:1 and becomes an unreadable strip in a printed
programme).

This mirrors the body of `main()` and reuses `FeatureTap`, `GridRenderer` and
`compose` rather than reimplementing them, exactly as the measurement harness
in `DECISIONS.md` §14 does. **The JPEG encode stays in the loop** even though
the output is a PNG: it costs 10.4 ms of the frame budget (§14), so leaving it
out would put an FPS number on the canvas that no real run ever reaches. The
picture is meant to be quotable.

The frame is grabbed late in the clip so that the EMA of the normalisation
scale has settled and the smoothed FPS is an average rather than a start-up
transient.

    python tools/make_screenshot.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterator, cast

import cv2

from feature_viz.demonstrator import (
    BGRImage,
    Config,
    FeatureTap,
    GridRenderer,
    build_config,
    compose,
)
from ultralytics import YOLO

ROOT: Path = Path(__file__).resolve().parent.parent
DOCS: Path = ROOT / "docs"
GRAB: int = 260  # of 300; late enough for the EMA and the FPS average to settle


def main() -> int:
    cfg: Config = build_config()
    cfg.source = str(ROOT / "assets" / "sample.mp4")
    if not Path(cfg.source).exists():
        print(f"[error] {cfg.source} missing; run tools/make_sample.py", file=sys.stderr)
        return 1
    if cfg.device != "cuda":
        # The reduced profile shows three layers instead of six (§5). Usable
        # for a smoke test, not for a picture that represents the project.
        print("[warn] CPU profile active - the result shows the reduced layout")

    model: YOLO = YOLO(cfg.weights)
    model.to(cfg.device)
    tap: FeatureTap = FeatureTap(model, cfg.targets)
    renderer: GridRenderer = GridRenderer(cfg, tap)
    info: str = f"{cfg.weights} | {cfg.device.upper()} | {cfg.imgsz}px | layers {cfg.targets}"

    stream: Iterator = cast(
        "Iterator",
        model.predict(
            source=cfg.source, imgsz=cfg.imgsz, stream=True, verbose=False, device=cfg.device
        ),
    )

    t_prev: float = time.perf_counter()
    fps: float = 0.0
    grid: BGRImage | None = None
    n: int = 0

    try:
        for result in stream:
            if not renderer.channels:
                renderer.calibrate()
            if n % cfg.vis_every == 0 or grid is None:
                grid = renderer.render()
            n += 1

            now: float = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            canvas: BGRImage = compose(result.plot(), grid, fps, info)
            # Kept for its cost, not its output - see the module docstring.
            cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality])

            if n < GRAB:
                continue

            DOCS.mkdir(exist_ok=True)
            cv2.imwrite(str(DOCS / "demo-full.png"), canvas)

            # One panel column beyond the detection frame. Derived from the
            # grid width rather than from the tile geometry, so it survives a
            # change of `tile` or `max_channels`; it lands a few pixels inside
            # the black gap between columns, which is invisible.
            frame_w: int = canvas.shape[1] - grid.shape[1]
            cut: int = frame_w + grid.shape[1] // cfg.panel_cols
            cv2.imwrite(str(DOCS / "demo-crop.png"), canvas[:, :cut])

            print(
                f"[info] frame {n}: canvas {canvas.shape[1]}x{canvas.shape[0]}, "
                f"{fps:.1f} FPS, {len(result.boxes)} detections"
            )
            print(f"[info] wrote {DOCS / 'demo-full.png'} and {DOCS / 'demo-crop.png'}")
            return 0
    finally:
        tap.close()

    print(f"[error] clip ended after {n} frames, before frame {GRAB}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
