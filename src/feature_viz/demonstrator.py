# feature-viz - live detection with feature-map visualisation
# Copyright (C) 2026  Florian Förster
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
  * The webcam is the default source; SOURCE selects another camera or a
    video file. A source that cannot be opened stops the demonstrator
    before anything starts, with a message naming the sample clip.

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
  * Checked with: mypy src/feature_viz/demonstrator.py
    --ignore-missing-imports (ultralytics ships no stubs).

Usage (console script from pyproject.toml; `pdm run feature-viz` outside an
activated venv, `python -m feature_viz.demonstrator` as the long form):
    feature-viz                  # auto-detect, browser
    SOURCE=assets/sample.mp4 feature-viz   # sample clip, no camera needed
    FORCE_CPU=1 feature-viz      # exercise the CPU path on a GPU box
    DISPLAY_MODE=window feature-viz
    DUMP_STRUCTURE=1 feature-viz # print the layer list

Dependencies:
    pip install ultralytics opencv-python numpy
    Python >= 3.10 (for `X | Y` in TypeAlias assignments)
"""

from __future__ import annotations

import functools
import html
import mimetypes
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
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
    cfg.source = os.getenv("SOURCE", "") or cfg.source
    return cfg


def source_error(source: str) -> str | None:
    """Why `source` cannot be used, or None if it can.

    Checked before the model loads and the server starts. Otherwise a missing
    camera surfaces as an ultralytics traceback *after* the stream URL has
    been printed - a page that never shows a picture, and no hint why.
    There is deliberately no fallback to the sample clip: at a talk, a camera
    that silently failed must not pass for a live picture (DECISIONS.md §8).
    """
    if "://" in source:
        return None  # stream URL; only ultralytics can tell
    if source.isdigit():
        # Opens the device once, briefly (about 40 ms), before ultralytics
        # opens it for real. OpenCV's own backend warnings are silenced for
        # the probe: a dozen lines of V4L2/FFMPEG noise would bury the one
        # line below that says what to do.
        level: int = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        try:
            cap: cv2.VideoCapture = cv2.VideoCapture(int(source))
            opened: bool = cap.isOpened()
            cap.release()
        finally:
            cv2.utils.logging.setLogLevel(level)
        if opened:
            return None
        return (
            f"cannot open camera {source}. Without a camera, use the sample clip:\n"
            f"    SOURCE=assets/sample.mp4 feature-viz"
        )
    if not Path(source).is_file():
        return f"source not found: {source} (relative to {Path.cwd()})"
    return None


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
# A caption line is (text, font scale, BGR colour).
CaptionLine: TypeAlias = "tuple[str, float, tuple[int, int, int]]"

FONT: Final[int] = cv2.FONT_HERSHEY_SIMPLEX


def caption_height(lines: Sequence[CaptionLine]) -> int:
    """Height of the strip `caption` would add. Needed up front so that a
    caller can shrink the image by exactly that much and keep the canvas
    the size it would have been."""
    return sum(int(round(26 * scale)) + 4 for (_, scale, _) in lines) + 4


def caption(img: BGRImage, lines: Sequence[CaptionLine]) -> BGRImage:
    """Stack a black strip carrying `lines` on top of `img`.

    The text used to be drawn onto the image itself, where it was not
    readable: feature-map tiles are bright and high-frequency, so no
    colour holds up against them, and the finished canvas is downscaled
    to roughly half size on a 1920-wide projector. A strip costs a few
    rows of pixels and covers no data at all.
    """
    heights: list[int] = [int(round(26 * scale)) + 4 for (_, scale, _) in lines]
    bar: BGRImage = np.zeros((caption_height(lines), img.shape[1], 3), dtype=np.uint8)

    y: int = 4
    for (text, scale, colour), h in zip(lines, heights):
        y += h
        cv2.putText(bar, text, (7, y - 4), FONT, scale, colour, 1, cv2.LINE_AA)
    return cast(BGRImage, np.vstack([bar, img]))


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
        label: str = f"L{idx} {self.tap.label(idx)}  {c}x{h}x{w}"
        return caption(panel, [(label, 0.7, (255, 255, 255))])

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


# ==========================================================================
# Funding notice
# ==========================================================================
# The demonstrator comes out of the K-M-I project, and its funding terms ask
# for this acknowledgement wherever results are shown. It is therefore part
# of the canvas itself, not only of the web page: a talk may run in window
# mode or show /stream.mjpg full-screen, and the stills in docs/ are cut from
# the canvas. See DECISIONS.md §17.
#
# Verbatim from the project's funding notice, typographic quotes included:
# the OpenCV >= 5 that pyproject.toml requires renders them in putText.
FUNDING_TEXT: Final[str] = (
    "The K-M-I research and development project is funded as part of the "
    "“Future of Work: Regional Competence Centers for Labor Research – "
    "Artificial Intelligence” funding initiative within the “Innovations for "
    "Tomorrow's Production, Services, and Work” program of the German Federal "
    "Ministry of Research, Technology and Space (BMFTR) and is supervised by "
    "the Project Management Agency Karlsruhe (PTKA)."
)
FUNDING_URL: Final[str] = "https://kmi-netzwerk.org/kmi-projekt/"

# (file in feature_viz/funding/, alt text). The files are the funder's
# originals, unmodified: logo guidelines forbid cropping the protected margin
# or recolouring, so they are only ever scaled.
FUNDING_LOGOS: Final[tuple[tuple[str, str], ...]] = (
    (
        "BMFTR_de_Web_RGB_gef_durch.jpg",
        "Funded by the German Federal Ministry of Research, Technology and Space",
    ),
    (
        "Logo_Kompetenzzentren_Arbeitsforschung.png",
        "Regional Competence Centres of Work Research (ReKodA)",
    ),
)

FUNDING_STRIP_H: Final[int] = 160
FUNDING_TEXT_SCALE: Final[float] = 0.75


def funding_asset(name: str) -> bytes:
    """Raw bytes of a logo, read from the installed package rather than the
    repository, so that a wheel or container install finds them too."""
    return (resources.files("feature_viz") / "funding" / name).read_bytes()


def _wrap(text: str, scale: float, max_w: int) -> list[str]:
    lines: list[str] = []
    line: str = ""
    for word in text.split():
        trial: str = f"{line} {word}".strip()
        if line and cv2.getTextSize(trial, FONT, scale, 1)[0][0] > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line]


@functools.cache
def funding_strip(width: int) -> BGRImage:
    """White strip with both logos and the funding text, `width` pixels wide.

    Built once per width and cached: the canvas width is fixed for a given
    source, so every frame after the first reuses the same array. White,
    because the BMFTR logo must sit on a white or very light ground.
    """
    logos: list[BGRImage] = []
    for name, _ in FUNDING_LOGOS:
        raw: NDArray[np.uint8] = np.frombuffer(funding_asset(name), dtype=np.uint8)
        img: BGRImage = cast(BGRImage, cv2.imdecode(raw, cv2.IMREAD_COLOR))
        s: float = FUNDING_STRIP_H / img.shape[0]
        logos.append(cast(BGRImage, cv2.resize(img, (int(img.shape[1] * s), FUNDING_STRIP_H),
                                               interpolation=cv2.INTER_AREA)))

    pad: int = 16
    x0: int = sum(logo.shape[1] for logo in logos) + pad
    lines: list[str] = _wrap(FUNDING_TEXT, FUNDING_TEXT_SCALE, width - x0 - pad)
    line_h: int = int(round(26 * FUNDING_TEXT_SCALE)) + 8
    h: int = max(FUNDING_STRIP_H, len(lines) * line_h + 2 * pad)

    strip: BGRImage = np.full((h, width, 3), 255, dtype=np.uint8)
    x: int = 0
    for logo in logos:
        # A canvas narrower than the logos is not a real case (the detection
        # frame alone is wider), but it must not crash the demonstrator.
        w: int = min(logo.shape[1], width - x)
        strip[:FUNDING_STRIP_H, x : x + w] = logo[:, :w]
        x += w
    y: int = (h - len(lines) * line_h) // 2
    for line in lines:
        y += line_h
        cv2.putText(strip, line, (x0, y - 8), FONT, FUNDING_TEXT_SCALE, (40, 40, 40), 1,
                    cv2.LINE_AA)
    return strip


def compose(frame: BGRImage, grid: BGRImage, fps: float, info: str) -> BGRImage:
    """Detection image and feature-map grid side by side, funding strip
    underneath."""

    def fit(img: BGRImage, height: int) -> BGRImage:
        s: float = height / img.shape[0]
        return cast(BGRImage, cv2.resize(img, (int(img.shape[1] * s), height)))

    lines: list[CaptionLine] = [
        (f"{fps:5.1f} FPS", 1.1, (120, 255, 120)),
        (info, 0.75, (210, 210, 210)),
    ]
    # The frame is fitted to the target height *minus* its caption strip, so
    # the strip costs no canvas area. Captioning after the fit also keeps the
    # text at a fixed pixel size, independent of the camera's resolution.
    bar: int = caption_height(lines)
    target_h: int = max(frame.shape[0] + bar, grid.shape[0])
    body: BGRImage = cast(
        BGRImage,
        np.hstack([caption(fit(frame, target_h - bar), lines), fit(grid, target_h)]),
    )
    return cast(BGRImage, np.vstack([body, funding_strip(body.shape[1])]))


# ==========================================================================
# MJPEG output
# ==========================================================================
SOURCE_URL: Final[str] = "https://github.com/foersterflorian/feature-viz"

# The notice below is not decoration. AGPL-3.0 §13 requires that users
# interacting with the program over a network be offered the Corresponding
# Source, and §5(d) wants the legal notice on the interactive interface. If
# this instance ever runs modified code, SOURCE_URL has to point at *that*
# version, not at the upstream repository.
PAGE: Final[str] = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>YOLO26 feature-map demonstrator</title>
<style>
  body {{ background:#111; color:#ddd; font-family:sans-serif;
         margin:0; padding:16px; }}
  h1 {{ font-size:16px; font-weight:normal; margin:0 0 12px; }}
  img {{ max-width:100%; height:auto; display:block; }}
  footer {{ font-size:12px; color:#888; margin-top:12px; }}
  footer a {{ color:#9bf; }}
  .funding {{ font-size:12px; color:#aaa; margin-top:16px; max-width:900px; }}
  .funding a {{ color:#9bf; }}
  /* The BMFTR logo must sit on a white ground; the page is dark. */
  .logos {{ display:inline-flex; gap:16px; background:#fff; padding:8px;
           margin-top:8px; }}
  .logos img {{ height:110px; width:auto; }}
</style></head>
<body><h1>YOLO26 &mdash; detection and feature maps &nbsp;|&nbsp; {info}</h1>
<img src="/stream.mjpg" alt="Stream">
<footer>feature-viz, Copyright &copy; 2026 Florian F&ouml;rster &mdash;
licensed under the
<a href="https://www.gnu.org/licenses/agpl-3.0.html">GNU AGPL v3</a> or later.
Source: <a href="{source}">{source}</a>.
Uses Ultralytics YOLO, also AGPL-3.0.</footer>
<section class="funding"><p>{funding_text}
More on the project: <a href="{funding_url}">{funding_url}</a></p>
<div class="logos">{funding_logos}</div></section>
</body></html>
"""


_FUNDING_FILES: Final[frozenset[str]] = frozenset(name for name, _ in FUNDING_LOGOS)


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
            body: bytes = PAGE.format(
                info=self.info,
                source=SOURCE_URL,
                funding_text=html.escape(FUNDING_TEXT),
                funding_url=FUNDING_URL,
                funding_logos="".join(
                    f'<img src="/funding/{name}" alt="{html.escape(alt)}">'
                    for name, alt in FUNDING_LOGOS
                ),
            ).encode("utf-8")
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
        elif self.path.removeprefix("/funding/") in _FUNDING_FILES:
            # Checked against a fixed list, so no path reaches the file
            # system that the code did not name itself.
            name: str = self.path.removeprefix("/funding/")
            data: bytes = funding_asset(name)
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "image/*")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(data)
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


def results(model: YOLO, cfg: Config) -> Iterator[Results]:
    """Detection results for `cfg.source`, frame by frame. A video file
    starts over when it ends; a camera or stream URL ends the run when it
    stops delivering, because then it has failed (DECISIONS.md §8).

    Each pass is a fresh `predict()` call - ultralytics has no loop option.
    Model, hooks, channel selection and normalisation state carry over, so
    the restart is invisible in the grid.
    """
    source: SourceSpec = int(cfg.source) if cfg.source.isdigit() else cfg.source
    loop: bool = isinstance(source, str) and Path(source).is_file()
    while True:
        # With stream=True the call always yields an iterator of Results; the
        # signature also admits the list and Tensor forms, which it cannot
        # return here. Results is a TYPE_CHECKING import, hence the string.
        stream: Iterator[Results] = cast(
            "Iterator[Results]",
            model.predict(
                source=source, imgsz=cfg.imgsz, stream=True, verbose=False, device=cfg.device
            ),
        )
        frames: int = 0
        for result in stream:
            frames += 1
            yield result
        # A file that yields nothing would otherwise spin here forever.
        if not loop or frames == 0:
            return


def close_source(model: YOLO) -> None:
    """Stop ultralytics' frame reader before the interpreter exits.

    For a camera or a stream URL, ultralytics reads frames in a daemon thread
    (`LoadStreams`) and only closes it when the source runs dry - which a
    camera never does. Leaving the loop by Ctrl+C or SIGTERM left that thread
    inside `VideoCapture.read()` while the interpreter shut down, and the C++
    runtime aborted: "terminate called without an active exception", exit
    134 (DECISIONS.md §10).

    Calls ultralytics' own `close()`; nothing is patched. `predictor.dataset`
    is not a documented interface, hence the getattr chain: if a future
    release moves it, this degrades to the old behaviour rather than failing.
    File sources have no `close()` and need none. Closing the predict
    generator instead was tried and does not help: the reader thread is
    independent of it.
    """
    dataset: Any = getattr(getattr(model, "predictor", None), "dataset", None)
    close: Any = getattr(dataset, "close", None)
    if callable(close):
        close()


def main() -> None:
    cfg: Config = build_config()

    error: str | None = source_error(cfg.source)
    if error is not None:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1)

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
        server: ThreadingHTTPServer = start_server(cfg, buffer, info)
        # The bound port, not cfg.port: with PORT=0 the OS picks one.
        print(f"[info] stream: http://localhost:{server.server_address[1]}/")

    stream: Iterator[Results] = results(model, cfg)

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
        close_source(model)
        if cfg.display_mode == "window":
            cv2.destroyAllWindows()
        print("[info] stopped")


if __name__ == "__main__":
    main()
