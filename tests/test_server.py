"""HTTP endpoints (DECISIONS.md §7, §16, §17).

The page footer is a legal requirement, not decoration: AGPL §13 obliges
the source offer to anyone using the demonstrator over the network, and the
funding terms oblige the acknowledgement. Both are asserted here.
"""

from __future__ import annotations

import html
import http.client
import threading
import time
from collections.abc import Iterator
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from feature_viz import demonstrator as demo
from tests.support import serve


@pytest.fixture(scope="module")
def server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[str, demo.FrameBuffer]]:
    """One live server on a free port for the whole module: shutdown() waits
    out serve_forever's 0.5 s poll interval, which per test would dominate the
    run time."""
    with serve(tmp_path_factory.mktemp("data"), INFO) as (url, buffer, _):
        yield url, buffer


# Markup in the info line must arrive as text: it includes the WEIGHTS path.
INFO: str = "yolo26n.pt | <b>&"


def get(url: str) -> tuple[int, str, bytes]:
    try:
        with urlopen(url, timeout=5) as resp:
            return resp.status, resp.headers.get_content_type(), resp.read()
    except HTTPError as err:
        return err.code, "", b""


@pytest.mark.parametrize("path", ["/", "/index.html"])
def test_page_carries_the_agpl_source_offer(
    server: tuple[str, demo.FrameBuffer], path: str
) -> None:
    """§16: AGPL §13 source offer and licence on the interactive interface."""
    status, ctype, body = get(server[0] + path)
    page: str = body.decode("utf-8")
    assert (status, ctype) == (200, "text/html")
    assert f'href="{demo.SOURCE_URL}"' in page
    assert "GNU AGPL v3" in page
    assert html.escape(INFO) in page and INFO not in page


def footer(page: str) -> str:
    return page.split("<footer>", 1)[1].split("</footer>", 1)[0]


def test_page_carries_the_funding_notice(server: tuple[str, demo.FrameBuffer]) -> None:
    """§17: text, project link and both logos, in the footer - the canvas
    no longer carries them in the browser."""
    notice: str = footer(get(server[0] + "/")[2].decode("utf-8"))
    assert html.escape(demo.FUNDING_TEXT[demo.DEFAULT_LANGUAGE]) in notice
    assert f'href="{demo.FUNDING_URL}"' in notice
    for name, _ in demo.FUNDING_LOGOS:
        assert f'src="/funding/{name}"' in notice


def test_footer_sets_licence_and_pipeline_apart_from_the_funding_notice(
    server: tuple[str, demo.FrameBuffer],
) -> None:
    """§17: one footer, two sections - the funding notice, and beside it the
    AGPL source offer and the pipeline line, which left the title."""
    page: str = get(server[0] + "/")[2].decode("utf-8")
    funding, about = footer(page).split('<section class="about">', 1)
    assert "GNU AGPL v3" in about and f'href="{demo.SOURCE_URL}"' in about
    assert html.escape(INFO) in about
    assert "GNU AGPL" not in funding and html.escape(INFO) not in funding
    assert html.escape(INFO) not in page.split("<footer>", 1)[0]


@pytest.mark.parametrize(
    ("name", "ctype"),
    [(n, "image/jpeg" if n.endswith(".jpg") else "image/png") for n, _ in demo.FUNDING_LOGOS],
)
def test_logos_are_served(
    server: tuple[str, demo.FrameBuffer], name: str, ctype: str
) -> None:
    status, served_type, body = get(f"{server[0]}/funding/{name}")
    assert (status, served_type) == (200, ctype)
    assert body == demo.funding_asset(name)


@pytest.mark.parametrize(
    "path",
    [
        "/funding/../demonstrator.py",
        "/funding/%2e%2e/demonstrator.py",
        "/funding/",
        "/funding/missing.png",
        "/nothing-here",
    ],
)
def test_everything_else_is_404(server: tuple[str, demo.FrameBuffer], path: str) -> None:
    """Only the names in FUNDING_LOGOS reach the file system. http.client
    sends the path exactly as given, without normalising '..'."""
    base: str = server[0].removeprefix("http://")
    host, port = base.split(":")
    conn: http.client.HTTPConnection = http.client.HTTPConnection(host, int(port), timeout=5)
    try:
        conn.request("GET", path)
        assert conn.getresponse().status == 404
    finally:
        conn.close()


@pytest.fixture
def fresh_buffer(monkeypatch: pytest.MonkeyPatch) -> demo.FrameBuffer:
    """An empty buffer behind the shared server, for tests that depend on
    what has (not) been published."""
    buffer: demo.FrameBuffer = demo.FrameBuffer()
    monkeypatch.setattr(demo.StreamHandler, "buffer", buffer)
    return buffer


def test_healthz_is_503_before_the_first_frame(
    server: tuple[str, demo.FrameBuffer], fresh_buffer: demo.FrameBuffer
) -> None:
    """§7: while the model loads there is nothing to show yet."""
    assert get(server[0] + "/healthz")[0] == 503


def test_healthz_is_ok_while_frames_keep_coming(
    server: tuple[str, demo.FrameBuffer], fresh_buffer: demo.FrameBuffer
) -> None:
    fresh_buffer.publish(b"frame")
    assert get(server[0] + "/healthz")[::2] == (200, b"ok")


def test_healthz_is_503_when_frames_stop(
    server: tuple[str, demo.FrameBuffer],
    fresh_buffer: demo.FrameBuffer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§7: a hung camera leaves process and server running with the last
    frame frozen - exactly what "process alive" could not catch."""
    fresh_buffer.publish(b"frame")
    monkeypatch.setattr(demo, "HEALTH_MAX_AGE", 0.0)
    time.sleep(0.01)
    assert get(server[0] + "/healthz")[0] == 503


def test_stream_without_frame_source_is_503(
    server: tuple[str, demo.FrameBuffer], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(demo.StreamHandler, "buffer", None)
    assert get(server[0] + "/stream.mjpg")[0] == 503


def test_closed_tab_does_not_disturb_the_server(
    server: tuple[str, demo.FrameBuffer], capfd: pytest.CaptureFixture[str]
) -> None:
    """A viewer closing the tab mid-stream is the normal case (§7): the
    handler must drop that connection quietly and keep serving others.
    "Quietly" is the part to check - socketserver survives a failing handler
    anyway, but prints its traceback to the console on every closed tab."""
    url, buffer = server
    buffer.publish(b"first")
    with urlopen(url + "/stream.mjpg", timeout=5) as resp:
        resp.readline()
    for i in range(20):  # writes into the closed socket
        buffer.publish(bytes([i]) * 70_000)
        time.sleep(0.01)
    time.sleep(0.2)  # let the handler thread hit the broken pipe and exit
    assert "Traceback" not in capfd.readouterr().err
    assert get(url + "/healthz")[::2] == (200, b"ok")
    buffer.publish(b"after")
    with urlopen(url + "/stream.mjpg", timeout=5) as resp:
        for _ in range(4):
            resp.readline()
        assert resp.read(5) == b"after"


def test_page_pulls_frames_instead_of_streaming(server: tuple[str, demo.FrameBuffer]) -> None:
    """§7: Firefox freezes on the MJPEG push stream at full frame rate. The
    page must pull single frames; MJPEG is only the no-JavaScript fallback."""
    page: str = get(server[0] + "/")[2].decode("utf-8")
    assert 'fetch("/frame.jpg?after="' in page
    assert page.count("/stream.mjpg") == 1
    assert '<noscript><img src="/stream.mjpg"' in page


def test_frame_returns_the_newest_frame_with_its_sequence(
    server: tuple[str, demo.FrameBuffer], fresh_buffer: demo.FrameBuffer
) -> None:
    """§7: the page sends the sequence back as `after` on its next request."""
    fresh_buffer.publish(b"old")
    fresh_buffer.publish(b"new")
    with urlopen(server[0] + "/frame.jpg", timeout=5) as resp:
        assert resp.headers.get_content_type() == "image/jpeg"
        assert resp.headers["X-Frame-Seq"] == "2"
        assert resp.read() == b"new"


def test_frame_after_waits_for_a_newer_frame(
    server: tuple[str, demo.FrameBuffer], fresh_buffer: demo.FrameBuffer
) -> None:
    """§7: a client that already has the newest frame is held until the
    next one exists, so polling in a loop does not spin on the same frame."""
    fresh_buffer.publish(b"seen")
    later: threading.Timer = threading.Timer(0.3, fresh_buffer.publish, args=(b"next",))
    later.start()
    t0: float = time.monotonic()
    try:
        with urlopen(server[0] + "/frame.jpg?after=1", timeout=5) as resp:
            assert resp.read() == b"next"
            assert resp.headers["X-Frame-Seq"] == "2"
    finally:
        later.join()
    assert time.monotonic() - t0 >= 0.25


def test_frame_is_503_before_the_first_frame(
    server: tuple[str, demo.FrameBuffer], fresh_buffer: demo.FrameBuffer
) -> None:
    """§7: the page retries until the model has loaded."""
    assert get(server[0] + "/frame.jpg")[0] == 503


def test_frame_rejects_a_malformed_after(
    server: tuple[str, demo.FrameBuffer], fresh_buffer: demo.FrameBuffer
) -> None:
    fresh_buffer.publish(b"frame")
    assert get(server[0] + "/frame.jpg?after=x")[0] == 400


def test_frames_share_one_connection(
    server: tuple[str, demo.FrameBuffer], fresh_buffer: demo.FrameBuffer
) -> None:
    """§7: one request per frame; under HTTP/1.0 each would open a new TCP
    connection, about 40 per second per viewer."""
    fresh_buffer.publish(b"frame")
    host, port = server[0].removeprefix("http://").split(":")
    conn: http.client.HTTPConnection = http.client.HTTPConnection(host, int(port), timeout=5)
    try:
        conn.request("GET", "/frame.jpg")
        first: http.client.HTTPResponse = conn.getresponse()
        first.read()
        assert not first.will_close
        sock: object = conn.sock
        conn.request("GET", "/frame.jpg")
        conn.getresponse().read()
        assert conn.sock is sock
    finally:
        conn.close()


def test_stream_delivers_the_published_frame_as_multipart(
    server: tuple[str, demo.FrameBuffer],
) -> None:
    url, buffer = server
    frame: bytes = b"\xff\xd8 not really a jpeg \xff\xd9"
    buffer.publish(frame)
    with urlopen(url + "/stream.mjpg", timeout=5) as resp:
        assert resp.headers["Content-Type"] == "multipart/x-mixed-replace; boundary=FRAME"
        assert resp.readline() == b"--FRAME\r\n"
        assert resp.readline() == b"Content-Type: image/jpeg\r\n"
        assert resp.readline() == f"Content-Length: {len(frame)}\r\n".encode()
        assert resp.readline() == b"\r\n"
        assert resp.read(len(frame)) == frame
