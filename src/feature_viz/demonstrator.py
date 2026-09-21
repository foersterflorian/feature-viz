"""
Demonstrator: YOLO26 detection with feature maps rendered alongside it.

Replaces the previous set of single-purpose scripts (yolo26_featuremaps.py,
featuremap_render.py, yolo26_realtime_demo.py) with one code path.

Core decisions:
  * One code path for GPU and CPU. The difference lies solely in a
    parameter set (the profile), never in diverging logic.
  * Output defaults to an MJPEG stream in the browser rather than
    cv2.imshow. That removes every dependency on X11/Wayland - relevant
    for running this in a container later on.
  * The source is switchable via environment variable, with a video file
    as fallback. The demonstrator therefore also starts when no camera
    is attached.

On the type annotations:
  * `from __future__ import annotations` turns every annotation into a
    string. `int | None` and `list[int]` are therefore writable on older
    interpreters too, and imports needed only for type checking can live
    under TYPE_CHECKING.
  * Image buffers are `NDArray[np.uint8]` throughout. The aliases
    `GrayImage` and `BGRImage` are the same type at runtime - NumPy
    cannot express the channel count in the type - but they state the
    expectation at the signature.
  * `YoloLayer` is a Protocol for the attributes Ultralytics attaches to
    the modules at runtime (.i, .f, .type). A type checker does not know
    them on `nn.Module`.
  * Checked with: mypy demonstrator.py --ignore-missing-imports
    (ultralytics ships no stubs).

Usage:
    python demonstrator.py                  # auto-detect, browser
    SOURCE=0 python demonstrator.py         # force webcam
    FORCE_CPU=1 python demonstrator.py      # exercise the CPU path on a GPU box
    DISPLAY_MODE=window python demonstrator.py
    DUMP_STRUCTURE=1 python demonstrator.py # print the layer list

Dependencies:
    pip install ultralytics opencv-python numpy
    Python >= 3.10 (for `X | Y` in TypeAlias assignments)
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol, TypeAlias, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn
from ultralytics import YOLO

if TYPE_CHECKING:
    from types import FrameType

    from torch.utils.hooks import RemovableHandle
    from ultralytics.engine.results import Results


# ==========================================================================
# Type aliases
# ==========================================================================
# Both are the same type at runtime. NumPy cannot express the channel
# count in the type, so the names only document the expectation:
# GrayImage is (H, W), BGRImage is (H, W, 3).
GrayImage: TypeAlias = NDArray[np.uint8]
BGRImage: TypeAlias = NDArray[np.uint8]

# The cv2 stubs return ndarray[Any, dtype[integer | floating]], which no
# longer unifies with NDArray[np.uint8] since NumPy 2 made ndarray generic
# over the shape. The buffers are uint8 at runtime, so the call sites cast
# back to these aliases rather than widening them to Any.

# Signature of a PyTorch forward hook. The inputs are deliberately `Any`:
# for modules such as Concat the argument is a list of tensors, not a
# single tensor.
HookFn: TypeAlias = Callable[[nn.Module, "tuple[Any, ...]", Any], None]

# What ultralytics accepts as `source` - here: camera index or path.
SourceSpec: TypeAlias = int | str


class YoloLayer(Protocol):
    """The attributes Ultralytics attaches to every module at runtime.

    Statically, `nn.Sequential[i]` yields only an `nn.Module`; a type
    checker does not know .i, .f and .type there. This Protocol makes
    them visible without changing anything at runtime.
    """

    i: int  # index in the module chain
    f: int | list[int]  # origin of the input ("from")
    type: str  # e.g. "ultralytics.nn.modules.block.C3k2"

    def register_forward_hook(self, hook: HookFn) -> RemovableHandle: ...


# ==========================================================================
# Configuration
# ==========================================================================
@dataclass
class Config:
    """One parameter set, two profiles. No separate code paths."""

    device: str
    imgsz: int
    targets: list[int]  # layer indices; check against the structure dump
    tile: int  # tile edge length in pixels
    vis_every: int  # visualise only every n-th frame
    max_channels: int = 64
    panel_cols: int = 3

    weights: str = "yolo26n.pt"
    source: str = "0"

    # Normalisation
    alpha: float = 0.08  # EMA weight; 1.0 = no smoothing
    quantiles: tuple[float, float] = (0.01, 0.99)
    update_every: int = 5  # frames between percentile recomputations
    sample: int = 50_000  # sample size for the percentiles
    gamma: float = 0.65  # < 1 lifts the midtones (SiLU is right-skewed)

    # Output
    display_mode: str = "mjpeg"  # "mjpeg" or "window"
    port: int = 8080
    jpeg_quality: int = 80

    # Derived, not a constructor argument: see __post_init__.
    gamma_lut: NDArray[np.uint8] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.gamma_lut = (np.linspace(0.0, 1.0, 256) ** self.gamma * 255).astype(np.uint8)


def build_config() -> Config:
    force_cpu: bool = os.getenv("FORCE_CPU", "0") == "1"
    use_gpu: bool = torch.cuda.is_available() and not force_cpu

    cfg: Config
    if use_gpu:
        # Presentation profile: full resolution, six layers, every frame.
        cfg = Config(
            device="cuda", imgsz=640, targets=[2, 4, 9, 10, 16, 22], tile=72, vis_every=1
        )
    else:
        # Degraded profile for functional testing and debugging.
        # Deliberately the same logic, only smaller: smaller input, three
        # layers, visualisation only every third frame. Detection still
        # runs on every frame.
        cfg = Config(
            device="cpu", imgsz=416, targets=[4, 9, 16], tile=56, vis_every=3, panel_cols=3
        )

    cfg.weights = os.getenv("WEIGHTS", cfg.weights)
    cfg.display_mode = os.getenv("DISPLAY_MODE", cfg.display_mode)
    cfg.port = int(os.getenv("PORT", str(cfg.port)))
    cfg.source = os.getenv("SOURCE", "") or _default_source()
    return cfg


def _default_source() -> str:
    """Without an explicit setting: prefer the sample video, else webcam.

    This makes startup independent of attached hardware - whoever sees the
    demonstrator for the first time gets a picture in any case.
    """
    sample: Path = Path(__file__).parent / "assets" / "sample.mp4"
    return str(sample) if sample.exists() else "0"


# ==========================================================================
# Extraction
# ==========================================================================
class FeatureTap:
    """Taps the output tensors of selected layers via forward hooks.

    Deliberately without .cpu() in the hook: that would synchronise the
    GPU once per layer and frame. Reduction to the data volume actually
    needed happens in the renderer instead.
    """

    def __init__(self, model: YOLO, targets: Sequence[int]) -> None:
        self.layers = cast(nn.Sequential, model.model.model)  # type: ignore
        self.activations: dict[int, Tensor] = {}
        self.targets: list[int] = self._validate(targets)
        self.handles: list[RemovableHandle] = [
            self.layer(i).register_forward_hook(self._hook(i)) for i in self.targets
        ]

    def layer(self, idx: int) -> YoloLayer:
        """The module with the Ultralytics extra attributes (.i, .f, .type)."""
        return cast("YoloLayer", self.layers[idx])

    def _validate(self, targets: Sequence[int]) -> list[int]:
        n: int = len(self.layers)
        valid: list[int] = [i for i in targets if 0 <= i < n]
        dropped: list[int] = [i for i in targets if i not in valid]
        if dropped:
            print(
                f"[warn] layer indices outside the model: {dropped} (model has {n} layers)",
                file=sys.stderr,
            )
        return valid

    def _hook(self, idx: int) -> HookFn:
        """Closure instead of a class: the only state is `idx`."""

        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, (list, tuple)):
                output = output[0]
            self.activations[idx] = cast(Tensor, output).detach()

        return hook

    def label(self, idx: int) -> str:
        return self.layer(idx).type.split(".")[-1]

    def dump_structure(self) -> None:
        print(f"{'idx':>4}  {'from':>12}  type")
        print("-" * 60)
        for module in self.layers:
            m: YoloLayer = cast("YoloLayer", module)
            mark: str = " <--" if m.i in self.targets else ""
            print(f"{m.i:>4}  {str(m.f):>12}  {m.type}{mark}")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


# ==========================================================================
# Normalisation
# ==========================================================================
class Scale:
    """Temporally smoothed scale per layer, estimated from a sample.

    Shared limits across all tiles of one layer - that keeps it visible
    which channels actually fire. Separate between layers, because their
    activation levels really do differ by orders of magnitude.

    `ready` instead of `lo is None`: that keeps lo/hi float throughout and
    lets `get()` promise a `tuple[float, float]` without any Optional
    handling.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg: Config = cfg
        self.lo: float = 0.0
        self.hi: float = 1.0
        self.ready: bool = False
        self.counter: int = 0

    def get(self, tensor: Tensor) -> tuple[float, float]:
        if not self.ready or self.counter % self.cfg.update_every == 0:
            # .float() is required: on CUDA torch.quantile does not support
            # float16, so half=True would otherwise fail here.
            flat: Tensor = tensor.flatten().float()
            if flat.numel() > self.cfg.sample:
                flat = flat[:: flat.numel() // self.cfg.sample]

            q_lo, q_hi = self.cfg.quantiles
            lo: float = torch.quantile(flat, q_lo).item()
            hi: float = torch.quantile(flat, q_hi).item()
            if hi - lo < 1e-8:
                hi = lo + 1e-8

            if not self.ready:
                self.lo, self.hi = lo, hi
                self.ready = True
            else:
                a: float = self.cfg.alpha
                self.lo += a * (lo - self.lo)
                self.hi += a * (hi - self.hi)

        self.counter += 1
        return self.lo, self.hi


# ==========================================================================
# Rendering
# ==========================================================================
class GridRenderer:
    """Feature maps -> tile grid. The expensive steps run on the model's
    compute device; only the finished tile stack is transferred."""

    def __init__(self, cfg: Config, tap: FeatureTap) -> None:
        self.cfg: Config = cfg
        self.tap: FeatureTap = tap
        self.scales: dict[int, Scale] = {i: Scale(cfg) for i in tap.targets}
        # Channel indices per layer; set once, fixed afterwards.
        self.channels: dict[int, Tensor] = {}

    def calibrate(self) -> None:
        """Channel selection by activity, once on the first frame.

        Fixed rather than per frame: otherwise the tiles keep swapping
        places and the grid becomes unreadable.
        """
        for i in self.tap.targets:
            t: Tensor = self.tap.activations[i]
            energy: Tensor = t[0].float().pow(2).mean(dim=(1, 2)).sqrt()
            order: Tensor = torch.argsort(energy, descending=True)
            n: int = min(self.cfg.max_channels, int(t.shape[1]))
            self.channels[i] = order[:n]

    def panel(self, idx: int) -> BGRImage:
        cfg: Config = self.cfg
        tensor: Tensor = self.tap.activations[idx]
        fmap: Tensor = tensor[0, self.channels[idx]]  # (K, H, W)

        lo, hi = self.scales[idx].get(fmap)

        # Resize first, normalise second - the interpolation then runs at
        # tile size instead of at full resolution.
        # NEAREST keeps the coarse grid structure of deep layers visible.
        small: Tensor = F.interpolate(
            fmap.unsqueeze(1).float(), size=(cfg.tile, cfg.tile), mode="nearest"
        ).squeeze(1)
        norm: Tensor = ((small - lo) / (hi - lo)).clamp_(0.0, 1.0)
        # The only GPU -> CPU transfer, already reduced to tile size.
        tiles: NDArray[np.uint8] = (norm * 255.0).to(torch.uint8).cpu().numpy()

        k: int = int(tiles.shape[0])
        t: int = cfg.tile
        cols: int = int(np.ceil(np.sqrt(k)))
        rows: int = int(np.ceil(k / cols))
        if k < rows * cols:
            pad: NDArray[np.uint8] = np.zeros((rows * cols - k, t, t), dtype=np.uint8)
            tiles = np.concatenate([tiles, pad], axis=0)

        # The grid in a single reshape instead of hundreds of single calls
        gray: GrayImage = (
            tiles.reshape(rows, cols, t, t).transpose(0, 2, 1, 3).reshape(rows * t, cols * t)
        )

        gray = cast(GrayImage, cv2.LUT(gray, cfg.gamma_lut))  # gamma as LUT, not pow
        gray[::t, :] = 50  # separator lines, purely cosmetic
        gray[:, ::t] = 50

        panel: BGRImage = cast(BGRImage, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        c, h, w = (int(tensor.shape[1]), int(tensor.shape[2]), int(tensor.shape[3]))
        cv2.putText(
            panel,
            f"L{idx} {self.tap.label(idx)}  {c}x{h}x{w}",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (60, 200, 255),
            1,
            cv2.LINE_AA,
        )
        return panel

    def render(self) -> BGRImage:
        panels: list[BGRImage] = [
            self.panel(i) for i in self.tap.targets if i in self.tap.activations
        ]
        return self._mosaic(panels)

    def _mosaic(self, panels: list[BGRImage], gap: int = 8) -> BGRImage:
        h: int = max(p.shape[0] for p in panels)
        w: int = max(p.shape[1] for p in panels)
        cells: list[BGRImage] = [
            cast(
                BGRImage,
                cv2.copyMakeBorder(
                    p, 0, h - p.shape[0], 0, w - p.shape[1], cv2.BORDER_CONSTANT, value=0
                ),
            )
            for p in panels
        ]

        cols: int = self.cfg.panel_cols
        rows: list[BGRImage] = []
        for start in range(0, len(cells), cols):
            chunk: list[BGRImage] = cells[start : start + cols]
            while len(chunk) < cols:
                chunk.append(np.zeros((h, w, 3), dtype=np.uint8))
            vsep: BGRImage = np.zeros((h, gap, 3), dtype=np.uint8)
            row: list[BGRImage] = []
            for cell in chunk:
                row += [cell, vsep]
            rows.append(np.hstack(row[:-1]))

        hsep: BGRImage = np.zeros((gap, rows[0].shape[1], 3), dtype=np.uint8)
        stacked: list[BGRImage] = []
        for r in rows:
            stacked += [r, hsep]
        return np.vstack(stacked[:-1])


def compose(frame: BGRImage, grid: BGRImage, fps: float, info: str) -> BGRImage:
    """Detection image and feature-map grid side by side."""
    target_h: int = max(frame.shape[0], grid.shape[0])

    def fit(img: BGRImage) -> BGRImage:
        s: float = target_h / img.shape[0]
        return cast(BGRImage, cv2.resize(img, (int(img.shape[1] * s), target_h)))

    frame = fit(frame)
    cv2.putText(
        frame,
        f"{fps:5.1f} FPS",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, info, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA
    )
    return np.hstack([frame, fit(grid)])


# ==========================================================================
# MJPEG output
# ==========================================================================
PAGE: Final[str] = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>YOLO26 feature-map demonstrator</title>
<style>
  body {{ background:#111; color:#ddd; font-family:sans-serif;
         margin:0; padding:16px; }}
  h1 {{ font-size:16px; font-weight:normal; margin:0 0 12px; }}
  img {{ max-width:100%; height:auto; display:block; }}
</style></head>
<body><h1>YOLO26 &mdash; detection and feature maps &nbsp;|&nbsp; {info}</h1>
<img src="/stream.mjpg" alt="Stream"></body></html>
"""


class FrameBuffer:
    """Holds exactly the most recently encoded frame. Slow clients skip
    frames instead of slowing the demonstrator down.

    The data and the lock protecting it live in the same object - the rule
    "never touch _jpeg without _cond" can only be enforced that way.
    """

    def __init__(self) -> None:
        self._cond: threading.Condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq: int = 0

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def wait(self, last_seq: int, timeout: float = 5.0) -> tuple[bytes | None, int]:
        """Blocks until a new frame is available. Returns (None, seq) for
        as long as none has ever been published."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout)
            return self._jpeg, self._seq


class StreamHandler(BaseHTTPRequestHandler):
    """HTTP endpoints. The class attributes are set by start_server -
    BaseHTTPRequestHandler allows no constructor of our own, since a new
    instance is created per connection. ClassVar makes it explicit that
    they belong to the class, not to the instance.
    """

    buffer: ClassVar[FrameBuffer | None] = None
    info: ClassVar[str] = ""

    def log_message(self, format: str, *args: Any) -> None:
        pass  # no request log on the console

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body: bytes = PAGE.format(info=self.info).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/healthz":
            # For the container health check later on
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        elif self.path == "/stream.mjpg":
            self._stream()
        else:
            self.send_error(404)

    def _stream(self) -> None:
        if self.buffer is None:
            self.send_error(503, "no frame source")
            return

        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()

        seq: int = -1
        jpeg: bytes | None
        try:
            while True:
                jpeg, seq = self.buffer.wait(seq)
                if jpeg is None:
                    continue
                self.wfile.write(b"--FRAME\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser tab closed, the normal case


def start_server(cfg: Config, buffer: FrameBuffer, info: str) -> ThreadingHTTPServer:
    StreamHandler.buffer = buffer
    StreamHandler.info = info
    server: ThreadingHTTPServer = ThreadingHTTPServer(("0.0.0.0", cfg.port), StreamHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ==========================================================================
# Main loop
# ==========================================================================
_stop: Final[threading.Event] = threading.Event()


def _on_signal(signum: int, frame: FrameType | None) -> None:
    _stop.set()


def main() -> None:
    cfg: Config = build_config()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    model: YOLO = YOLO(cfg.weights)
    model.to(cfg.device)

    tap: FeatureTap = FeatureTap(model, cfg.targets)
    if os.getenv("DUMP_STRUCTURE", "0") == "1":
        tap.dump_structure()

    renderer: GridRenderer = GridRenderer(cfg, tap)

    info: str = f"{cfg.weights} | {cfg.device.upper()} | {cfg.imgsz}px | layers {cfg.targets}"
    print(f"[info] {info}")
    print(f"[info] source: {cfg.source}")
    if cfg.device == "cpu":
        print("[info] CPU profile active - reduced rendering, meant as a functional test.")

    buffer: FrameBuffer | None = None
    if cfg.display_mode == "mjpeg":
        buffer = FrameBuffer()
        start_server(cfg, buffer, info)
        print(f"[info] stream: http://localhost:{cfg.port}/")

    source: SourceSpec = int(cfg.source) if cfg.source.isdigit() else cfg.source
    # With stream=True the call always yields an iterator of Results; the
    # signature also admits the list and Tensor forms, which it cannot return
    # here. Results is a TYPE_CHECKING import, hence the string target.
    stream: Iterator[Results] = cast(
        "Iterator[Results]",
        model.predict(
            source=source, imgsz=cfg.imgsz, stream=True, verbose=False, device=cfg.device
        ),
    )

    t_prev: float = time.perf_counter()
    fps: float = 0.0
    grid: BGRImage | None = None
    n: int = 0

    try:
        for result in stream:
            if _stop.is_set():
                break

            if not renderer.channels:
                renderer.calibrate()

            # Detection runs on every frame, the visualisation possibly
            # less often. The previous grid stays on screen.
            if n % cfg.vis_every == 0 or grid is None:
                grid = renderer.render()
            n += 1

            now: float = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            canvas: BGRImage = compose(result.plot(), grid, fps, info)

            if buffer is not None:
                ok, encoded = cv2.imencode(
                    ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality]
                )
                if ok:
                    buffer.publish(encoded.tobytes())
            else:
                cv2.imshow("YOLO26 demonstrator", canvas)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
    finally:
        tap.close()
        if cfg.display_mode == "window":
            cv2.destroyAllWindows()
        print("[info] stopped")


if __name__ == "__main__":
    main()
