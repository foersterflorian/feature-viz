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
  * Output defaults to the browser rather than cv2.imshow. That removes
    every dependency on X11/Wayland - relevant for running this in a
    container later on. The page pulls one JPEG at a time, so every
    browser gets only as many frames as it can draw (DECISIONS.md §7).
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
import json
import mimetypes
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol, TypeAlias, cast
from urllib.parse import parse_qs, urlsplit

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

    # Camera mode, requested when SOURCE is a camera index (DECISIONS.md §8).
    # Opened bare, UVC cameras come up as YUYV 640x480; MJPG gives 720p at
    # the full 30 FPS over USB 2.0. 1080p shows no gain on a projector.
    camera_fourcc: str = "MJPG"
    camera_size: tuple[int, int] = (1280, 720)
    camera_fps: int = 30

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
    # The funding strip along the bottom of the canvas (DECISIONS.md §17).
    # Only the window has no page around it; in the browser the page's footer
    # carries the notice. Set from display_mode in build_config().
    funding_on_canvas: bool = False
    # Where the staff's language and theme are kept (DECISIONS.md §22, §23).
    # Relative to the working directory; the container points it at its
    # volume.
    data_dir: str = "data"

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
    cfg.funding_on_canvas = cfg.display_mode == "window"
    cfg.port = int(os.getenv("PORT", str(cfg.port)))
    cfg.source = os.getenv("SOURCE", "") or cfg.source
    cfg.data_dir = os.getenv("DATA_DIR", "") or cfg.data_dir
    return cfg


def source_error(source: str) -> str | None:
    """Why `source` cannot be used, or None if it can.

    Checked before the model loads and the server starts. Otherwise a missing
    camera surfaces as an ultralytics traceback *after* the stream URL has
    been printed - a page that never shows a picture, and no hint why.
    There is deliberately no fallback to the sample clip: at a talk, a camera
    that silently failed must not pass for a live picture (DECISIONS.md §8).
    """
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
    # Only a plain path that does not exist is refused. Stream URLs, globs
    # (`frames/*.jpg`), directories and `screen` are ultralytics sources too;
    # only ultralytics can judge them, and rejecting them here would take
    # away inputs it supports.
    if (
        "://" in source
        or any(ch in source for ch in "*?[")
        or source.lower().startswith("screen")
        or Path(source).exists()
    ):
        return None
    return f"source not found: {source} (relative to {Path.cwd()})"


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
# Screen language and theme
# ==========================================================================
# The page speaks German or English and comes dark or light, as the staff
# choose with Ctrl+Alt+L and Ctrl+Alt+H; the choice survives a restart
# (DECISIONS.md §22, §23). The canvas stays as it is: its few words are
# technical, and it is image content, dark in both themes.
class Language(StrEnum):
    DE = "de"
    EN = "en"


class Theme(StrEnum):
    DARK = "dark"
    LIGHT = "light"


# English and dark until the staff choose otherwise, as in teach-by-showing.
DEFAULT_LANGUAGE: Final[Language] = Language.EN
DEFAULT_THEME: Final[Theme] = Theme.DARK
LANGUAGE_FILE: Final[str] = "language.json"
THEME_FILE: Final[str] = "theme.json"


@dataclass(frozen=True, slots=True)
class Texts:
    """Every word the page says, in one language. One instance per language,
    so that a text missing in one of them is an error at import, not a gap
    on the screen. Plain text: the page escapes it. `licence` holds the
    placeholder {licence} for the link."""

    page_title: str
    headline: str
    funding_more: str
    licence: str
    source: str
    # Under the title, for a visitor the staff have not reached yet: what is
    # where, and how to read a tile (DECISIONS.md §22). It names the corners
    # of the GPU profile's grid, the one shown on the stand.
    legend: str
    explanation: str
    # The confirmation the staff see after a switch, each in its own words.
    language_name: str
    dark: str
    light: str


ENGLISH: Final[Texts] = Texts(
    page_title="YOLO26 feature-map demonstrator",
    headline="YOLO26 — detection and feature maps",
    funding_more="More on the project:",
    licence=(
        "feature-viz, Copyright © 2026 Florian Förster — licensed under the "
        "{licence} or later. Uses Ultralytics YOLO, also AGPL-3.0."
    ),
    source="Source:",
    legend=(
        "Left: what the network recognises. Right: what individual layers of the "
        "network ‘see’ in the process."
    ),
    explanation=(
        "Each tile shows a trained pattern detector: the lighter a spot is, the more "
        "strongly the detector has triggered there. In the top left, the detectors "
        "respond to simple edges and contrasts; in the bottom right, to larger shapes "
        "– from which the network ultimately constructs a ‘person’ or ‘car’."
    ),
    language_name="English",
    dark="Dark",
    light="Light",
)
GERMAN: Final[Texts] = Texts(
    page_title="YOLO26-Demonstrator für Feature-Maps",
    headline="YOLO26 – Objekterkennung und Feature-Maps",
    funding_more="Mehr zum Projekt:",
    licence=(
        "feature-viz, Copyright © 2026 Florian Förster – lizenziert unter der "
        "{licence} oder später. Verwendet Ultralytics YOLO, ebenfalls AGPL-3.0."
    ),
    source="Quellcode:",
    legend=(
        "Links: Was das Netz erkennt. Rechts: Was einzelne Schichten des Netzes dabei "
        "„sehen“."
    ),
    explanation=(
        "Jede Kachel zeigt einen trainierten Musterdetektor: Je heller eine Stelle ist, "
        "desto stärker hat der Detektor an dieser Stelle angesprochen. Oben links "
        "reagieren die Detektoren auf einfache Kanten und Kontraste; unten rechts auf "
        "größere Formen – aus denen das Netz schließlich eine „Person“ oder ein „Auto“ "
        "zusammensetzt."
    ),
    language_name="Deutsch",
    dark="Dunkel",
    light="Hell",
)
TEXTS: Final[dict[Language, Texts]] = {Language.EN: ENGLISH, Language.DE: GERMAN}


def load_choice[E: StrEnum](path: Path, key: str, kind: type[E], default: E) -> E:
    """The choice as last stored; the default when nothing, or nothing
    readable, is stored, so that a broken file never stops the demonstrator."""
    if not path.exists():
        return default
    try:
        return kind(json.loads(path.read_text(encoding="utf-8"))[key])
    except (OSError, ValueError, KeyError, TypeError):
        print(f"[warn] unreadable {path}; using {default}", file=sys.stderr)
        return default


def save_choice(path: Path, key: str, value: StrEnum) -> None:
    """Written whole and then renamed, so that a crash never leaves half a
    file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial: Path = path.with_suffix(".partial")
    partial.write_text(json.dumps({key: str(value)}), encoding="utf-8")
    partial.replace(path)


class Choices:
    """The staff's language and theme, read from `data_dir` at start and
    written back on every change. Knows nothing about HTTP.

    A file that cannot be written leaves the choice in effect until the
    next restart: losing it then is better than refusing the switch now.
    """

    def __init__(self, data_dir: Path) -> None:
        self._dir: Path = data_dir
        self._lock: threading.Lock = threading.Lock()
        self._language: Language = load_choice(
            data_dir / LANGUAGE_FILE, "language", Language, DEFAULT_LANGUAGE
        )
        self._theme: Theme = load_choice(data_dir / THEME_FILE, "theme", Theme, DEFAULT_THEME)

    def current(self) -> tuple[Language, Theme]:
        with self._lock:
            return self._language, self._theme

    def set_language(self, language: Language) -> None:
        with self._lock:
            self._language = language
            self._save(LANGUAGE_FILE, "language", language)

    def set_theme(self, theme: Theme) -> None:
        with self._lock:
            self._theme = theme
            self._save(THEME_FILE, "theme", theme)

    def _save(self, name: str, key: str, value: StrEnum) -> None:
        try:
            save_choice(self._dir / name, key, value)
        except OSError as err:
            print(f"[warn] cannot keep {key}={value}: {err}", file=sys.stderr)


# ==========================================================================
# Funding notice
# ==========================================================================
# The demonstrator comes out of the KMI project, and its funding terms ask
# for this acknowledgement wherever results are shown: in the browser in the
# page's footer, in the window and on the stills in a strip along the bottom
# of the canvas. See DECISIONS.md §17.
#
# The German text verbatim from the funder's material, German quotes
# included; the English one our translation of it, in British English and
# naming the project "KMI" as the German does. Both character for character
# the same as teach-by-showing's funding.py, so that the two demonstrators on
# the stand read as a pair (DECISIONS.md §17). The page shows the one of its
# language (§22); the canvas strip always the English one, since the canvas
# is language-neutral. Typographic quotes included: the OpenCV >= 5 that
# pyproject.toml requires renders them in putText.
FUNDING_TEXT: Final[dict[Language, str]] = {
    Language.DE: (
        "Das Forschungs- und Entwicklungsprojekt KMI wird im Rahmen der Fördermaßnahme "
        "„Zukunft der Arbeit: Regionale Kompetenzzentren der Arbeitsforschung – "
        "Künstliche Intelligenz“ im Programm „Innovationen für die Produktion, "
        "Dienstleistung und Arbeit von morgen“ des Bundesministeriums für Forschung, "
        "Technologie und Raumfahrt (BMFTR) gefördert und vom Projektträger Karlsruhe "
        "(PTKA) betreut."
    ),
    Language.EN: (
        "The KMI research and development project is funded as part of the "
        "“Future of Work: Regional Competence Centres for Labour Research – "
        "Artificial Intelligence” funding initiative within the “Innovations for "
        "Tomorrow's Production, Services, and Work” programme of the German Federal "
        "Ministry of Research, Technology and Space (BMFTR) and is supervised by "
        "the Project Management Agency Karlsruhe (PTKA)."
    ),
}
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
    lines: list[str] = _wrap(FUNDING_TEXT[Language.EN], FUNDING_TEXT_SCALE, width - x0 - pad)
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


def compose(
    frame: BGRImage, grid: BGRImage, fps: float, info: str, *, funding_on_canvas: bool
) -> BGRImage:
    """Detection image and feature-map grid side by side, and the funding
    strip underneath where no page carries the notice (§17). Keyword-only
    and without a default, so that no caller drops the notice by accident."""

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
    if not funding_on_canvas:
        return body
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
<html lang="{language}" data-theme="{theme}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{page_title}</title>
<style>
  /* Every colour of the page a token, in a dark and a light set
     (DECISIONS.md §23). The canvas is image content and stays dark; the
     footer stays white for the BMFTR logo (§17). */
  :root {{ --bg:#111; --fg:#e6e6e6; --muted:#8a8a8a; --footer-edge:transparent; }}
  :root[data-theme="light"] {{ --bg:#f4f4f2; --fg:#1a1a1a; --muted:#555;
                               --footer-edge:#c8c8c4; }}
  /* Type as in teach-by-showing beside it on the stand: the same face, the
     same sizes for its headline and its hint (§23). What a visitor is
     meant to read is at least 2vh, there judged from standing distance. */
  :root {{ --text-headline:3.2vh; --text-hint:2.4vh; }}
  /* One screen, no scrolling (DECISIONS.md §17): title on top, the canvas
     shrunk to the space left, the footer along the bottom. Sizes in vh so
     that the kiosk's 4K monitor and a laptop show the same proportions;
     max() keeps the small print readable in a short window. */
  html, body {{ height:100%; margin:0; }}
  body {{ background:var(--bg); color:var(--fg); font-family:system-ui, sans-serif;
         display:flex; flex-direction:column; }}
  h1 {{ font-size:max(20px, var(--text-headline)); font-weight:600; margin:0;
       padding:1.6vh 2.5vw 0; }}
  main {{ flex:1; min-height:0; display:flex; align-items:center;
         justify-content:center; padding:0 2.5vw 1.2vh; }}
  /* Both limits on a replaced element keep its aspect ratio. */
  main canvas, main img {{ max-width:100%; max-height:100%; display:block; }}
  /* White, because the BMFTR logo must sit on a white ground (§17). */
  footer {{ display:flex; align-items:center; gap:2vw; padding:0.8vh 2.5vw;
           background:#fff; color:#333; font-size:max(12px, 1.15vh);
           line-height:1.35; border-top:1px solid var(--footer-edge); }}
  footer p {{ margin:0; }}
  footer a {{ color:#1d4ed8; }}
  .funding {{ flex:1; display:flex; align-items:center; gap:2vw; }}
  .funding img {{ height:max(80px, 7vh); width:auto; }}
  /* Licence and pipeline, set off from the funding notice: a rule, a
     narrower column, smaller and greyer type. */
  /* 30vw: at the stand's size the footer is then as high as
     teach-by-showing's, 137 px, and the funding text keeps its lines. */
  .about {{ max-width:30vw; padding-left:2vw; border-left:1px solid #ccc;
           color:#555; font-size:max(11px, 1vh); }}
  .about p + p {{ margin-top:0.5vh; }}
  /* What is where and how to read a tile (§22): below the title, in the
     space the width-limited canvas leaves free on the stand's monitor. */
  .legend {{ padding:0.4vh 2.5vw 1.2vh; font-size:max(14px, var(--text-hint));
            line-height:1.4; max-width:120ch; color:var(--muted); }}
  .legend p {{ margin:0; }}
  /* Three lines held for the explanation, which the German fills at the
     stand's size and the English does not: the canvas then keeps its
     place when the staff switch the language, as in teach-by-showing. */
  .legend p + p {{ margin-top:0.4vh; min-height:calc(3 * 1.4em); }}
  /* The staff's confirmation after a switch, on a box of the ground's
     colour so that it reads over the canvas too. */
  #notice {{ position:fixed; top:8vh; left:50%; transform:translateX(-50%);
            margin:0; padding:0.3vh 1.2vw; border-radius:0.6vh;
            background:var(--bg); color:var(--fg); font-size:max(20px, 4vh);
            font-weight:600; }}
</style></head>
<body><h1>{headline}</h1>
<div class="legend"><p>{legend}</p>
<p>{explanation}</p></div>
<p id="notice" hidden data-language="{language_name}" data-dark="{dark}"
   data-light="{light}"></p>
<main>
<canvas id="stream" role="img" aria-label="Stream"></canvas>
<noscript><img src="/stream.mjpg" alt="Stream"></noscript>
</main>
<script>
// Pull, not push (DECISIONS.md section 7): the next frame is requested only
// once the previous one is drawn, so a browser that decodes slowly gets fewer
// frames instead of a growing backlog. `after` makes the server hold the
// request until a newer frame exists, so a fast browser does not spin.
(async () => {{
  const canvas = document.getElementById("stream");
  const ctx = canvas.getContext("2d");
  const pause = (ms) => new Promise((done) => setTimeout(done, ms));
  let seq = -1;
  for (;;) {{
    try {{
      const resp = await fetch("/frame.jpg?after=" + seq, {{ cache: "no-store" }});
      if (!resp.ok) {{
        await pause(500);  // no frame yet: the model is still loading
        continue;
      }}
      seq = Number(resp.headers.get("X-Frame-Seq"));
      const frame = await createImageBitmap(await resp.blob());
      if (canvas.width !== frame.width || canvas.height !== frame.height) {{
        canvas.width = frame.width;
        canvas.height = frame.height;
      }}
      ctx.drawImage(frame, 0, 0);
      frame.close();
    }} catch (err) {{
      await pause(500);  // server restarting or network gone: keep trying
    }}
  }}
}})();

// The staff's keys (DECISIONS.md section 22, 23): Ctrl+Alt+L switches the
// language, Ctrl+Alt+H the theme, once per press. The server keeps the
// choice. A new language reloads the page, whose words the server fills in;
// a new theme only sets the attribute, so the picture keeps running.
(() => {{
  const root = document.documentElement;
  const notice = document.getElementById("notice");
  let timer = 0;
  const confirm = (word) => {{
    notice.textContent = word;
    notice.hidden = false;
    clearTimeout(timer);
    timer = setTimeout(() => {{ notice.hidden = true; }}, 3000);
  }};
  // Survives the reload, so that the new language is confirmed in itself.
  const MARK = "feature-viz-language-switched";
  try {{
    if (sessionStorage.getItem(MARK)) {{
      sessionStorage.removeItem(MARK);
      confirm(notice.dataset.language);
    }}
  }} catch (err) {{}}  // storage blocked: the reload is confirmation enough
  const choose = async (key, value) => {{
    try {{
      return (await fetch("/choice?" + key + "=" + value, {{ method: "POST" }})).ok;
    }} catch (err) {{
      return false;
    }}
  }};
  window.addEventListener("keydown", async (e) => {{
    if (!e.ctrlKey || !e.altKey || e.repeat) return;
    const key = e.key.toLowerCase();
    if (key === "l") {{
      e.preventDefault();
      if (await choose("language", root.lang === "de" ? "en" : "de")) {{
        try {{ sessionStorage.setItem(MARK, "1"); }} catch (err) {{}}
        location.reload();
      }}
    }} else if (key === "h") {{
      e.preventDefault();
      const theme = root.dataset.theme === "light" ? "dark" : "light";
      if (await choose("theme", theme)) {{
        root.dataset.theme = theme;
        confirm(notice.dataset[theme]);
      }}
    }}
  }});
}})();
</script>
<footer>
<section class="funding">{funding_logos}
<p>{funding_text} {funding_more}
<a href="{funding_url}">{funding_url}</a></p></section>
<section class="about">
<p>{licence}</p>
<p>{source_label} <a href="{source}">{source}</a></p>
<p>{info}</p>
</section>
</footer>
</body></html>
"""


LICENCE_LINK: Final[str] = (
    '<a href="https://www.gnu.org/licenses/agpl-3.0.html">GNU AGPL v3</a>'
)


def page(language: Language, theme: Theme, info: str) -> str:
    """The page in the staff's language and theme. Every text is escaped
    here; `info` carries the WEIGHTS path from the environment."""
    texts: Texts = TEXTS[language]
    return PAGE.format(
        language=language,
        theme=theme,
        page_title=html.escape(texts.page_title),
        headline=html.escape(texts.headline),
        legend=html.escape(texts.legend),
        explanation=html.escape(texts.explanation),
        language_name=html.escape(texts.language_name),
        dark=html.escape(texts.dark),
        light=html.escape(texts.light),
        info=html.escape(info),
        source=SOURCE_URL,
        source_label=html.escape(texts.source),
        licence=html.escape(texts.licence).format(licence=LICENCE_LINK),
        funding_text=html.escape(FUNDING_TEXT[language]),
        funding_more=html.escape(texts.funding_more),
        funding_url=FUNDING_URL,
        funding_logos="".join(
            f'<img src="/funding/{name}" alt="{html.escape(alt)}">'
            for name, alt in FUNDING_LOGOS
        ),
    )


_FUNDING_FILES: Final[frozenset[str]] = frozenset(name for name, _ in FUNDING_LOGOS)


# /healthz reports unhealthy once the newest frame is older than this. Far
# above any frame interval (the CPU profile still publishes every frame,
# about 30 ms apart), far below the time it takes someone to notice a frozen
# picture at a talk.
HEALTH_MAX_AGE: Final[float] = 5.0


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
        self._published: float | None = None  # time.monotonic() of the last publish

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._published = time.monotonic()
            self._cond.notify_all()

    def age(self) -> float | None:
        """Seconds since the last publish; None before the first frame."""
        with self._cond:
            return None if self._published is None else time.monotonic() - self._published

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
    choices: ClassVar[Choices | None] = None
    # The page fetches every frame separately (/frame.jpg). Under the
    # default HTTP/1.0 each of those would open a new TCP connection.
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass  # no request log on the console

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            language, theme = (
                self.choices.current()
                if self.choices is not None
                else (DEFAULT_LANGUAGE, DEFAULT_THEME)
            )
            body: bytes = page(language, theme, self.info).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # A reload after a switch must not come from a cache (§22).
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/healthz":
            self._health()
        elif urlsplit(self.path).path == "/frame.jpg":
            self._frame()
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

    def do_POST(self) -> None:
        """`/choice?language=de|en` or `/choice?theme=dark|light`: the staff's
        switch (DECISIONS.md §22, §23). Exactly one known key with a valid
        value, and no body - which keeps this below §12's threshold for a
        web framework."""
        url = urlsplit(self.path)
        if url.path != "/choice":
            self.send_error(404)
            return
        if self.choices is None:
            self.send_error(503, "no choices kept")
            return
        if int(self.headers.get("Content-Length") or 0) != 0:
            # Unread, a body would be taken for the next request on this
            # keep-alive connection.
            self.close_connection = True
            self.send_error(400, "no body expected")
            return
        params: dict[str, list[str]] = parse_qs(url.query, keep_blank_values=True)
        try:
            ((key, values),) = params.items()
            (value,) = values
            if key == "language":
                self.choices.set_language(Language(value))
            elif key == "theme":
                self.choices.set_theme(Theme(value))
            else:
                raise ValueError(key)
        except ValueError:
            self.send_error(400, "expected one of language=de|en, theme=dark|light")
            return
        self.send_response(204)
        self.end_headers()

    def _health(self) -> None:
        """200 while frames keep coming, 503 otherwise. "Process alive" is not
        enough: a hung camera leaves the process and the server running with
        the last frame frozen on screen (DECISIONS.md §7)."""
        age: float | None = self.buffer.age() if self.buffer is not None else None
        healthy: bool = age is not None and age <= HEALTH_MAX_AGE
        body: bytes = (
            b"ok"
            if healthy
            else b"no frame yet"
            if age is None
            else f"stale: last frame {age:.1f} s ago".encode()
        )
        self.send_response(200 if healthy else 503)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _frame(self) -> None:
        """The newest frame as a single JPEG; what the page uses. With
        `after=<seq>` the request is held until a frame newer than `seq`
        exists (at most FrameBuffer.wait's timeout), so a client that polls
        in a loop never receives the same frame twice in a row while frames
        keep coming. A server restart resets the counter; wait() compares for
        inequality, so the stale `seq` of a reloaded server still returns
        at once."""
        if self.buffer is None:
            self.send_error(503, "no frame source")
            return
        try:
            after: int = int(parse_qs(urlsplit(self.path).query).get("after", ["-1"])[0])
        except ValueError:
            self.send_error(400, "after must be an integer")
            return
        jpeg: bytes | None
        seq: int
        jpeg, seq = self.buffer.wait(after)
        if jpeg is None:
            self.send_error(503, "no frame yet")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Seq", str(seq))
        self.end_headers()
        self.wfile.write(jpeg)

    def _stream(self) -> None:
        """MJPEG for players (VLC, ffplay) and for the page without
        JavaScript. Not what the page uses: Firefox cannot shed a push stream
        it decodes too slowly and freezes (DECISIONS.md §7)."""
        if self.buffer is None:
            self.send_error(503, "no frame source")
            return

        self.send_response(200)
        # The stream has no length and ends only when the client leaves.
        self.send_header("Connection", "close")
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
    StreamHandler.choices = Choices(Path(cfg.data_dir))
    server: ThreadingHTTPServer = ThreadingHTTPServer(("0.0.0.0", cfg.port), StreamHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ==========================================================================
# Frame rate
# ==========================================================================
class FrameRate:
    """Smoothed frame rate for the canvas (DECISIONS.md §20).

    The EMA runs over the interval between frames and is inverted once,
    instead of averaging the per-frame rate 1/dt. For a stream source,
    ultralytics polls for the next frame with a fixed sleep, so the intervals
    alternate between about 16 and 50 ms; with the talk camera read that way,
    the mean of 1/dt read 38 FPS for a real 28.
    """

    def __init__(self) -> None:
        self.last: float | None = None
        self.interval: float | None = None

    def tick(self, now: float) -> float:
        """Record a frame at `now` (seconds) and return the smoothed rate.
        0.0 until a second frame gives a first interval, which then seeds
        the average instead of easing in from zero."""
        if self.last is not None:
            dt: float = max(now - self.last, 1e-6)
            self.interval = dt if self.interval is None else 0.9 * self.interval + 0.1 * dt
        self.last = now
        return 0.0 if self.interval is None else 1.0 / self.interval


# ==========================================================================
# Main loop
# ==========================================================================
_stop: Final[threading.Event] = threading.Event()


def _on_signal(signum: int, frame: FrameType | None) -> None:
    _stop.set()


def open_camera(index: int, cfg: Config) -> cv2.VideoCapture:
    """Open camera `index` in the mode `cfg` asks for.

    ultralytics opens a camera with a bare `cv2.VideoCapture`, which OpenCV
    sets to 640x480 - on UVC cameras that is YUYV, and a mode set beforehand
    with v4l2-ctl does not survive the open (DECISIONS.md §8). FOURCC goes
    first: V4L2 only offers the larger sizes at full rate once MJPG is
    selected. A camera that lacks the mode delivers what it has; the caller
    reports the mode actually in effect.
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*cfg.camera_fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera_size[1])
    cap.set(cv2.CAP_PROP_FPS, cfg.camera_fps)
    # CAP_PROP_BUFFERSIZE stays at the driver default. With a single buffer
    # the driver cannot capture while the loop works, so every other frame
    # is lost: 16.7 FPS instead of 28.7 (DECISIONS.md §8). The loop is faster
    # than the camera, so the default queue does not fill up either.
    return cap


def camera_mode(cap: cv2.VideoCapture) -> str:
    """The mode in effect, e.g. "MJPG 1280x720 @ 30"."""
    code: int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc: str = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
    w: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return f"{fourcc} {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f}"


def results(model: YOLO, cfg: Config) -> Generator[Results, None, None]:
    """Detection results for `cfg.source`, frame by frame. A video file
    starts over when it ends; a camera or stream URL ends the run when it
    stops delivering, because then it has failed (DECISIONS.md §8).

    A camera is read here, not by ultralytics, so that its mode can be set
    (`open_camera`); each frame goes to `predict()` as an array. Everything
    else is ultralytics' source handling. For a file, each pass is a fresh
    `predict()` call - ultralytics has no loop option. Model, hooks, channel
    selection and normalisation state carry over, so the restart is
    invisible in the grid.
    """
    source: SourceSpec = int(cfg.source) if cfg.source.isdigit() else cfg.source
    if isinstance(source, int):
        cap: cv2.VideoCapture = open_camera(source, cfg)
        print(f"[info] camera: {camera_mode(cap)}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                # Without stream=True, predict() returns one Results per
                # input image - here exactly one.
                yield cast(
                    "list[Results]",
                    model.predict(
                        source=frame, imgsz=cfg.imgsz, verbose=False, device=cfg.device
                    ),
                )[0]
        finally:
            # Also reached when main() closes the generator on Ctrl+C.
            cap.release()
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

    For a stream URL, ultralytics reads frames in a daemon thread
    (`LoadStreams`) and only closes it when the source runs dry. A camera
    was read that way too until results() took it over, and a camera never
    runs dry: leaving the loop by Ctrl+C or SIGTERM left that thread inside
    `VideoCapture.read()` while the interpreter shut down, and the C++
    runtime aborted: "terminate called without an active exception", exit
    134 (DECISIONS.md §8).

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

    stream: Generator[Results, None, None] = results(model, cfg)

    rate: FrameRate = FrameRate()
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

            fps: float = rate.tick(time.perf_counter())
            canvas: BGRImage = compose(
                result.plot(), grid, fps, info, funding_on_canvas=cfg.funding_on_canvas
            )

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
        # Releases a camera opened by results(); close_source() stops
        # ultralytics' reader for a stream URL.
        stream.close()
        tap.close()
        close_source(model)
        if cfg.display_mode == "window":
            cv2.destroyAllWindows()
        print("[info] stopped")


if __name__ == "__main__":
    main()
