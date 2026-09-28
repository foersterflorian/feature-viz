"""HTTP endpoints (DECISIONS.md §7, §16, §17).

The page footer is a legal requirement, not decoration: AGPL §13 obliges
the source offer to anyone using the demonstrator over the network, and the
funding terms oblige the acknowledgement. Both are asserted here.
"""

from __future__ import annotations

import html
import http.client
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from feature_viz import demonstrator as demo


@pytest.fixture(scope="module")
def server() -> Iterator[tuple[str, demo.FrameBuffer]]:
    """One live server on a free port for the whole module: shutdown() waits
    out serve_forever's 0.5 s poll interval, which per test would dominate the
    run time. The class-level state start_server() writes into StreamHandler
    is restored afterwards."""
    saved: tuple[demo.FrameBuffer | None, str] = (
        demo.StreamHandler.buffer,
        demo.StreamHandler.info,
    )
    cfg: demo.Config = demo.Config(device="cpu", imgsz=416, targets=[4], tile=56, vis_every=1)
    cfg.port = 0
    buffer: demo.FrameBuffer = demo.FrameBuffer()
    srv: ThreadingHTTPServer = demo.start_server(cfg, buffer, INFO)
    host, port = srv.server_address[:2]
    try:
        yield f"http://{host!s}:{port}", buffer
    finally:
        srv.shutdown()
        srv.server_close()
        demo.StreamHandler.buffer, demo.StreamHandler.info = saved


# Markup in the info line must arrive as text: it includes the WEIGHTS path.
INFO: str = "yolo26n.pt | <b>&"


def get(url: str) -> tuple[int, str, bytes]:
    try:
        with urlopen(url, timeout=5) as resp:
            return resp.status, resp.headers.get_content_type(), resp.read()
    except HTTPError as err:
        return err.code, "", b""


def test_page_carries_the_agpl_source_offer(server: tuple[str, demo.FrameBuffer]) -> None:
    """§16: AGPL §13 source offer and licence on the interactive interface."""
    status, ctype, body = get(server[0] + "/")
    page: str = body.decode("utf-8")
    assert (status, ctype) == (200, "text/html")
    assert f'href="{demo.SOURCE_URL}"' in page
    assert "GNU AGPL v3" in page
    assert html.escape(INFO) in page and INFO not in page


def test_page_carries_the_funding_notice(server: tuple[str, demo.FrameBuffer]) -> None:
    """§17: text, project link and both logos."""
    page: str = get(server[0] + "/")[2].decode("utf-8")
    assert html.escape(demo.FUNDING_TEXT) in page
    assert f'href="{demo.FUNDING_URL}"' in page
    for name, _ in demo.FUNDING_LOGOS:
        assert f'src="/funding/{name}"' in page


@pytest.mark.parametrize(
    ("name", "ctype"),
    [(n, "image/jpeg" if n.endswith(".jpg") else "image/png") for n, _ in demo.FUNDING_LOGOS],
)
def test_logos_are_served(server: tuple[str, demo.FrameBuffer], name: str, ctype: str) -> None:
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


def test_healthz(server: tuple[str, demo.FrameBuffer]) -> None:
    assert get(server[0] + "/healthz")[::2] == (200, b"ok")


def test_stream_without_frame_source_is_503(
    server: tuple[str, demo.FrameBuffer], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(demo.StreamHandler, "buffer", None)
    assert get(server[0] + "/stream.mjpg")[0] == 503


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
