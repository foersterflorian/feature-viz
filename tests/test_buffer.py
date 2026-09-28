"""FrameBuffer: the hand-over between inference loop and HTTP clients
(DECISIONS.md §7)."""

from __future__ import annotations

import threading
import time

from feature_viz import demonstrator as demo


def test_wait_times_out_with_none_before_the_first_frame() -> None:
    t0: float = time.perf_counter()
    jpeg, seq = demo.FrameBuffer().wait(-1, timeout=0.05)
    assert jpeg is None and seq == 0
    assert time.perf_counter() - t0 < 1.0


def test_wait_returns_a_published_frame() -> None:
    buf: demo.FrameBuffer = demo.FrameBuffer()
    buf.publish(b"a")
    assert buf.wait(-1, timeout=0.05) == (b"a", 1)


def test_slow_reader_skips_to_the_newest_frame() -> None:
    """§7: a slow client skips frames instead of throttling the loop. This
    path was never exercised by a real client in §14."""
    buf: demo.FrameBuffer = demo.FrameBuffer()
    buf.publish(b"1")
    _, seq = buf.wait(-1, timeout=0.05)
    for frame in (b"2", b"3", b"4"):
        buf.publish(frame)
    assert buf.wait(seq, timeout=0.05) == (b"4", 4)


def test_waiting_reader_is_woken_by_publish() -> None:
    buf: demo.FrameBuffer = demo.FrameBuffer()
    result: list[tuple[bytes | None, int]] = []
    reader: threading.Thread = threading.Thread(
        target=lambda: result.append(buf.wait(0, timeout=2.0)), daemon=True
    )
    reader.start()
    time.sleep(0.05)
    buf.publish(b"x")
    reader.join(timeout=2.0)
    assert result == [(b"x", 1)]


def test_age_counts_from_the_last_publish() -> None:
    """Feeds /healthz (§7): None until the first frame, then small and
    growing, reset by every publish."""
    buf: demo.FrameBuffer = demo.FrameBuffer()
    assert buf.age() is None
    buf.publish(b"a")
    time.sleep(0.05)
    first: float | None = buf.age()
    assert first is not None and 0.04 < first < 1.0
    buf.publish(b"b")
    second: float | None = buf.age()
    assert second is not None and second < first
