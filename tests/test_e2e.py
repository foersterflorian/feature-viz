"""The demonstrator as it is actually started (DECISIONS.md §3, §8, §11, §17).

Needs the real yolo26n.pt; skipped without it. The run itself goes through a
subprocess, exactly as `feature-viz` is launched, with the reduced profile so
that the result does not depend on the GPU of the machine.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import numpy as np
import pytest
from ultralytics import YOLO

from feature_viz import demonstrator as demo
from tests.support import EXPECTED_TARGETS, ROOT, WEIGHTS

pytestmark = pytest.mark.weights

SAMPLE: Path = ROOT / "assets" / "sample.mp4"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def first_frame(url: str, timeout: float) -> bytes:
    """Reads one JPEG from the MJPEG stream."""
    with urlopen(url, timeout=timeout) as resp:
        assert resp.readline() == b"--FRAME\r\n"
        resp.readline()  # Content-Type
        length: int = int(resp.readline().split(b":")[1])
        resp.readline()
        return bytes(resp.read(length))


@pytest.fixture
def running(tmp_path: Path) -> Iterator[tuple[subprocess.Popen[str], int, Path]]:
    """The demonstrator on the sample clip, started in an empty directory."""
    port: int = free_port()
    env: dict[str, str] = {
        **os.environ,
        "FORCE_CPU": "1",
        "SOURCE": str(SAMPLE),
        "PORT": str(port),
        "WEIGHTS": str(WEIGHTS),
        "DISPLAY_MODE": "mjpeg",
    }
    proc: subprocess.Popen[str] = subprocess.Popen(
        [sys.executable, "-m", "feature_viz.demonstrator"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        yield proc, port, tmp_path
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def wait_for_health(port: int, proc: subprocess.Popen[str], timeout: float = 60.0) -> None:
    deadline: float = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"demonstrator exited early:\n{proc.stdout.read() if proc.stdout else ''}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1):
                return
        except (URLError, ConnectionError):
            time.sleep(0.2)
    pytest.fail("demonstrator did not come up")


def test_real_checkpoint_has_the_documented_targets(gpu_cfg: demo.Config) -> None:
    """§3: verified against the YAML build elsewhere; this is the checkpoint
    that is actually shown."""
    tap: demo.FeatureTap = demo.FeatureTap(YOLO(str(WEIGHTS)), gpu_cfg.targets)
    try:
        assert {i: tap.label(i) for i in tap.targets} == EXPECTED_TARGETS
    finally:
        tap.close()


def test_sample_clip_streams_and_ends_cleanly(
    running: tuple[subprocess.Popen[str], int, Path],
) -> None:
    """Start to finish: the stream delivers a decodable canvas with the
    funding strip at the bottom, and the process stops by itself at the end
    of the clip with exit code 0."""
    proc, port, cwd = running
    wait_for_health(port, proc)

    jpeg: bytes = first_frame(f"http://127.0.0.1:{port}/stream.mjpg", timeout=30)
    canvas = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert canvas is not None and canvas.ndim == 3
    # §17: the right end of the strip is white ground (JPEG, hence not 255).
    strip_h: int = demo.funding_strip(canvas.shape[1]).shape[0]
    assert canvas[-strip_h:, -20:].mean() > 245

    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 0, out
    assert "[info] stopped" in out

    # §11: with an absolute WEIGHTS nothing is downloaded into the working
    # directory - which in a container would be a vanishing overlay layer.
    assert list(cwd.iterdir()) == []
