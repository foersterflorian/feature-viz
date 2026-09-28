"""The demonstrator as it is actually started (DECISIONS.md §3, §8, §11, §17).

Runs go through a subprocess, exactly as `feature-viz` is launched, with the
reduced profile so that the result does not depend on the GPU of the machine.
The tests marked `weights` need the real yolo26n.pt, the one marked `camera` a
webcam as well; each is skipped when its requirement is missing.
"""

from __future__ import annotations

import os
import re
import signal
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
STREAM_LINE: re.Pattern[str] = re.compile(r"\[info\] stream: http://localhost:(\d+)/")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def frames(url: str, n: int, timeout: float) -> list[bytes]:
    """Reads `n` consecutive JPEGs from one MJPEG connection. The server sends
    a frame only when a new one has been published, so every frame after the
    first proves the pipeline is still producing."""
    out: list[bytes] = []
    with urlopen(url, timeout=timeout) as resp:
        for _ in range(n):
            assert resp.readline() == b"--FRAME\r\n"
            resp.readline()  # Content-Type
            length: int = int(resp.readline().split(b":")[1])
            resp.readline()
            out.append(bytes(resp.read(length)))
            resp.readline()  # CRLF after the payload
    return out


def launch(cwd: Path, **env: str) -> subprocess.Popen[str]:
    """`feature-viz` as a subprocess on the reduced profile, with the real
    checkpoint and PORT=0. Unbuffered, so that the startup line arrives while
    the process runs."""
    return subprocess.Popen(
        [sys.executable, "-m", "feature_viz.demonstrator"],
        cwd=cwd,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "FORCE_CPU": "1",
            "PORT": "0",
            "WEIGHTS": str(WEIGHTS),
            **env,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


@pytest.fixture
def stopper() -> Iterator[list[subprocess.Popen[str]]]:
    """Collects launched processes and kills whatever a failing test left
    running."""
    procs: list[subprocess.Popen[str]] = []
    yield procs
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def stream_port(proc: subprocess.Popen[str]) -> int:
    """Reads stdout up to the `[info] stream:` line and returns its port, so
    the line is checked too: it must name the port actually bound."""
    assert proc.stdout is not None
    seen: list[str] = []
    for line in proc.stdout:
        seen.append(line)
        match: re.Match[str] | None = STREAM_LINE.match(line)
        if match:
            return int(match.group(1))
    pytest.fail("no stream line before exit:\n" + "".join(seen))


def wait_for_health(port: int, proc: subprocess.Popen[str], timeout: float = 60.0) -> None:
    deadline: float = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out: str = proc.stdout.read() if proc.stdout else ""
            pytest.fail(f"demonstrator exited early:\n{out}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1):
                return
        except (URLError, ConnectionError):
            time.sleep(0.2)
    pytest.fail("demonstrator did not come up")


def start(cwd: Path, stopper: list[subprocess.Popen[str]], **env: str) -> tuple[
    subprocess.Popen[str], str
]:
    """Launches, waits until the server answers, returns the stream URL."""
    proc: subprocess.Popen[str] = launch(cwd, **env)
    stopper.append(proc)
    port: int = stream_port(proc)
    wait_for_health(port, proc)
    return proc, f"http://127.0.0.1:{port}/stream.mjpg"


def stop(proc: subprocess.Popen[str], sig: signal.Signals = signal.SIGTERM) -> str:
    proc.send_signal(sig)
    out, _ = proc.communicate(timeout=30)
    return out


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
    tmp_path: Path, stopper: list[subprocess.Popen[str]]
) -> None:
    """Start to finish: the stream delivers a decodable canvas with the
    funding strip at the bottom, and SIGTERM ends the run with exit code 0.
    (The clip loops, so it never ends by itself - see the next test.)"""
    proc, url = start(tmp_path, stopper, SOURCE=str(SAMPLE))

    jpeg: bytes = frames(url, 1, timeout=30)[0]
    canvas = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert canvas is not None and canvas.ndim == 3
    # §17: the right end of the strip is white ground (JPEG, hence not 255).
    strip_h: int = demo.funding_strip(canvas.shape[1]).shape[0]
    assert canvas[-strip_h:, -20:].mean() > 245

    out: str = stop(proc)
    assert proc.returncode == 0, out
    assert "[info] stopped" in out

    # §11: with an absolute WEIGHTS nothing is downloaded into the working
    # directory - which in a container would be a vanishing overlay layer.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.weights
def test_video_file_loops(tmp_path: Path, stopper: list[subprocess.Popen[str]]) -> None:
    """§8: a file starts over instead of ending the run. A 10-frame clip is
    through in well under a second; still producing new frames several
    seconds later means it looped."""
    clip: Path = tmp_path / "short.mp4"
    writer: cv2.VideoWriter = cv2.VideoWriter(
        str(clip), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (160, 128)
    )
    for i in range(10):
        writer.write(np.full((128, 160, 3), 20 * i, dtype=np.uint8))
    writer.release()

    proc, url = start(tmp_path, stopper, SOURCE=str(clip))
    time.sleep(3.0)
    assert proc.poll() is None, "stopped at the end of the clip"
    frames(url, 3, timeout=10)
    out: str = stop(proc)
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
def test_stopping_a_camera_run_exits_cleanly(
    tmp_path: Path, stopper: list[subprocess.Popen[str]], sig: signal.Signals
) -> None:
    """§8: Ctrl+C or SIGTERM during a webcam run used to abort the C++ runtime
    ("terminate called without an active exception", exit 134), because
    ultralytics' reader thread was still inside VideoCapture.read()."""
    proc, url = start(tmp_path, stopper, SOURCE="0")
    frames(url, 1, timeout=30)  # the camera delivers
    out: str = stop(proc, sig)
    assert proc.returncode == 0, out
    assert "terminate called" not in out
    assert "[info] stopped" in out
