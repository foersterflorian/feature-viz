"""The demonstrator as it is actually started (DECISIONS.md §3, §8, §10, §11, §17).

Runs go through a subprocess, exactly as `feature-viz` is launched, with the
reduced profile so that the result does not depend on the GPU of the machine.
The tests marked `weights` need the real yolo26n.pt, the one marked `camera` a
webcam as well; each is skipped when its requirement is missing.
"""

from __future__ import annotations

import os
import signal
import re
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
    """The demonstrator on the sample clip, started in an empty directory.

    PORT=0 lets the OS choose, and the port is taken from the startup line -
    which is therefore checked too: it must name the port actually bound.
    """
    env: dict[str, str] = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",  # the startup line must arrive while it runs
        "FORCE_CPU": "1",
        "SOURCE": str(SAMPLE),
        "PORT": "0",
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
        yield proc, stream_port(proc), tmp_path
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def stream_port(proc: subprocess.Popen[str]) -> int:
    """Reads stdout up to the `[info] stream:` line and returns its port."""
    assert proc.stdout is not None
    seen: list[str] = []
    for line in proc.stdout:
        seen.append(line)
        match: re.Match[str] | None = re.match(r"\[info\] stream: http://localhost:(\d+)/", line)
        if match:
            return int(match.group(1))
    pytest.fail("no stream line before exit:\n" + "".join(seen))


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


@pytest.mark.weights
def test_real_checkpoint_has_the_documented_targets(gpu_cfg: demo.Config) -> None:
    """§3: verified against the YAML build elsewhere; this is the checkpoint
    that is actually shown."""
    tap: demo.FeatureTap = demo.FeatureTap(YOLO(str(WEIGHTS)), gpu_cfg.targets)
    try:
        assert {i: tap.label(i) for i in tap.targets} == EXPECTED_TARGETS
    finally:
        tap.close()


@pytest.mark.weights
def test_sample_clip_streams_and_stops_cleanly(
    running: tuple[subprocess.Popen[str], int, Path],
) -> None:
    """Start to finish: the stream delivers a decodable canvas with the
    funding strip at the bottom, and SIGTERM ends the run with exit code 0.
    (The clip loops, so it never ends by itself - see the next test.)"""
    proc, port, cwd = running
    wait_for_health(port, proc)

    jpeg: bytes = first_frame(f"http://127.0.0.1:{port}/stream.mjpg", timeout=30)
    canvas = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert canvas is not None and canvas.ndim == 3
    # §17: the right end of the strip is white ground (JPEG, hence not 255).
    strip_h: int = demo.funding_strip(canvas.shape[1]).shape[0]
    assert canvas[-strip_h:, -20:].mean() > 245

    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0, out
    assert "[info] stopped" in out

    # §11: with an absolute WEIGHTS nothing is downloaded into the working
    # directory - which in a container would be a vanishing overlay layer.
    assert list(cwd.iterdir()) == []


@pytest.mark.weights
def test_video_file_loops(tmp_path: Path) -> None:
    """§8: a file starts over instead of ending the run. A 10-frame clip is
    through in well under a second, so still running and still streaming
    after several seconds means it looped."""
    clip: Path = tmp_path / "short.mp4"
    writer: cv2.VideoWriter = cv2.VideoWriter(
        str(clip), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (160, 128)
    )
    for i in range(10):
        writer.write(np.full((128, 160, 3), 20 * i, dtype=np.uint8))
    writer.release()

    env: dict[str, str] = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "FORCE_CPU": "1",
        "SOURCE": str(clip),
        "PORT": "0",
        "WEIGHTS": str(WEIGHTS),
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
        port: int = stream_port(proc)
        wait_for_health(port, proc)
        time.sleep(3.0)
        assert proc.poll() is None, "stopped at the end of the clip"
        first_frame(f"http://127.0.0.1:{port}/stream.mjpg", timeout=10)
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == 0, out


def test_unusable_source_stops_before_anything_starts(tmp_path: Path) -> None:
    """§8: a missing camera ends the run with one readable line and exit 1 -
    before the model loads (hence no `weights` mark) and before the server
    prints a URL that would never show a picture."""
    env: dict[str, str] = {**os.environ, "SOURCE": "99", "PORT": str(free_port())}
    proc: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-m", "feature_viz.demonstrator"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 1
    assert "SOURCE=assets/sample.mp4" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "[info] stream:" not in proc.stdout



@pytest.mark.weights
@pytest.mark.camera
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
def test_stopping_a_camera_run_exits_cleanly(tmp_path: Path, sig: signal.Signals) -> None:
    """§10: Ctrl+C or SIGTERM during a webcam run used to abort the C++ runtime
    ("terminate called without an active exception", exit 134), because
    ultralytics' reader thread was still inside VideoCapture.read()."""
    env: dict[str, str] = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "FORCE_CPU": "1",
        "SOURCE": "0",
        "PORT": "0",
        "WEIGHTS": str(WEIGHTS),
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
        port: int = stream_port(proc)
        wait_for_health(port, proc)
        first_frame(f"http://127.0.0.1:{port}/stream.mjpg", timeout=30)  # camera delivers
        proc.send_signal(sig)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == 0, out
    assert "terminate called" not in out
    assert "[info] stopped" in out
