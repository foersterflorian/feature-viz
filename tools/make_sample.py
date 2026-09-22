"""Build `assets/sample.mp4` from a still image.

The demonstrator needs a *video* source, not a picture. `model.predict` on a
single image yields exactly one frame, and the FPS counter that `compose`
draws onto the canvas is smoothed from a starting value of 0.0 - after one
frame it shows a number that means nothing. A short clip also keeps the EMA of
the normalisation scale (`DECISIONS.md` §4) in motion, which is what the live
case does.

The motion is a slow pan across the still plus a slight brightness drift, so
that consecutive frames genuinely differ. This mirrors how the clip for the
performance measurements in §14 was built.

**Requires ffmpeg on PATH.** The OpenCV wheels ship no usable H.264 encoder -
`avc1` resolves to a hardware encoder that is not present on a normal desktop,
and the remaining `mp4v` produces a 19 MB clip for ten seconds, which is not
something to put in a repository. Encoding is therefore handed to ffmpeg over
a pipe. This is a dependency of *this tool*, not of the demonstrator: the
resulting file is an ordinary H.264 MP4 that OpenCV reads without help.

Rebuild:

    python tools/make_sample.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

BGRImage = NDArray[np.uint8]

ROOT: Path = Path(__file__).resolve().parent.parent
SOURCE: Path = ROOT / "assets" / "street-scene.jpg"
DEST: Path = ROOT / "assets" / "sample.mp4"

WIDTH: int = 1280
HEIGHT: int = 720
FPS: int = 30
SECONDS: int = 10

# The crop window, in source pixels. Chosen so that the cable car stays fully
# inside the frame over the whole pan; a subject that drifts out of view would
# change which channels dominate the fixed channel ranking (§4).
CROP_H: int = 2200
CROP_W: int = CROP_H * WIDTH // HEIGHT
CROP_Y: int = 500
PAN_X: tuple[int, int] = (0, 480)

# Peak deviation of the brightness drift. Large enough that the percentile
# estimate has something to track, small enough not to look like a fade.
DRIFT: float = 0.04

# Conservative on purpose. The early target layers respond to exactly the
# high-frequency detail that lossy compression discards first, so a clip that
# looks fine to the eye can still flatten the tiles the demonstrator exists to
# show. CRF 18 is visually lossless for this material and the pan compresses
# well regardless.
CRF: str = "18"


def ffmpeg_command() -> list[str]:
    """Raw BGR frames in on stdin, H.264 MP4 out."""
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", CRF,
        # yuv420p and the even dimensions it needs: without it the file will
        # not play in browsers or on the projector laptop at the venue.
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(DEST),
    ]


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("[error] ffmpeg not found on PATH; see this module's docstring", file=sys.stderr)
        return 1

    still: BGRImage | None = cast("BGRImage | None", cv2.imread(str(SOURCE), cv2.IMREAD_COLOR))
    if still is None:
        print(f"[error] cannot read {SOURCE}", file=sys.stderr)
        return 1

    h, w = still.shape[:2]
    if CROP_Y + CROP_H > h or PAN_X[1] + CROP_W > w:
        print(f"[error] crop window does not fit in {w}x{h}", file=sys.stderr)
        return 1

    frames: int = FPS * SECONDS
    proc: subprocess.Popen[bytes] = subprocess.Popen(ffmpeg_command(), stdin=subprocess.PIPE)
    assert proc.stdin is not None  # stdin=PIPE guarantees it; mypy does not know

    try:
        for n in range(frames):
            t: float = n / (frames - 1)  # 0.0 .. 1.0
            x: int = int(PAN_X[0] + t * (PAN_X[1] - PAN_X[0]))
            crop: BGRImage = still[CROP_Y : CROP_Y + CROP_H, x : x + CROP_W]
            frame: BGRImage = cast(
                "BGRImage", cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            )

            # A full sine period, so the clip loops without a brightness step.
            gain: float = 1.0 + DRIFT * float(np.sin(2.0 * np.pi * t))
            frame = cast("BGRImage", cv2.convertScaleAbs(frame, alpha=gain, beta=0.0))

            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        code: int = proc.wait()

    if code != 0:
        print(f"[error] ffmpeg exited with {code}", file=sys.stderr)
        return 1

    size_kb: int = DEST.stat().st_size // 1024
    print(f"[info] wrote {DEST} - {frames} frames, {WIDTH}x{HEIGHT}, {FPS} fps, {size_kb} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
