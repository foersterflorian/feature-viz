# Design Decisions

**Project:** YOLO feature-map visualisation demonstrator (TUC-KMI)
**Status:** porting from the YOLOv7-based original to Ultralytics YOLO26
**Last updated:** 2026-09-21

This file records *why* the code looks the way it does. The code says what it
does; this says what was considered and rejected, so that changing something
here is a deliberate act rather than an accident.

Read this before modifying `demonstrator.py`. If you change one of these
decisions, update the corresponding section rather than deleting it — the
rejected alternatives are the useful part.

---

## 0. What this demonstrator shows

A live camera (or video) feed with detection boxes and labels on the left, and
the internal feature maps of several selected network layers on the right,
rendered as grids of greyscale tiles. The point is to make the abstraction
gradient through the network visible: early layers look like edge filters at
high spatial resolution, deep layers are coarse and selective.

The original version was built on YOLOv7 and ran at ~30 FPS from a webcam.

---

## 1. Model: Ultralytics YOLO26, not YOLOv7

**Context.** The original used YOLOv7 (WongKinYiu lineage). The current state of
the YOLO family lives in the Ultralytics ecosystem. YOLO26 was released in
January 2026: NMS-free end-to-end inference, DFL removed, edge-optimised, five
scales (n/s/m/l/x), AGPL-3.0.

**Decision.** Port to Ultralytics YOLO26.

**Why.** This is a port, not an upgrade — different repository, different module
library, different layer structure. Doing it once now is cheaper than doing it
later from an even more distant starting point. YOLO11 is the conservative
alternative and shares the same block vocabulary (C3k2, C2PSA), so the code
here works on both by changing `WEIGHTS`.

**Consequences.**

- The extraction mechanism changed (see §2).
- The layer *identities* changed. YOLOv7's E-ELAN, SPPCSPC and RepConv have no
  one-to-one counterpart. YOLO26's backbone/neck blocks are C3k2, with SPPF
  (residual in YOLO26) and C2PSA attention.
- **Anything the old demonstrator showed about anchors, objectness maps or NMS
  is gone.** YOLO26 is anchor-free and its default inference path is NMS-free
  (an optional one-to-many path with NMS exists). If the talk track relied on
  explaining NMS, it needs rewriting. The two-path design is itself a
  reasonable thing to explain instead.

**Licensing.** AGPL-3.0 is more restrictive than YOLOv7's GPL-3.0, and it
matters if the demonstrator is ever exposed as a network service. If that
becomes a problem, RF-DETR (Apache 2.0) is the alternative, but its layer
structure differs substantially and the visualisation would need rework.

**Revisit when.** YOLO27 was announced for around September 2026, adding
monocular and stereo depth estimation. Check whether it has shipped and whether
its block structure still matches the layer indices used here.

---

## 2. Extraction: forward hooks, not a patched forward pass

**Context.** In YOLOv7 the usual approach was to patch `forward_once` in the
repository to intercept intermediate tensors.

**Decision.** Use `register_forward_hook` on selected modules of
`model.model.model`. No repository modification.

**Why.** The Ultralytics model is still an `nn.Sequential` whose modules carry
`.i` (index), `.f` (from) and `.type`, so index-based layer selection survives
the port. Hooks keep the `ultralytics` package a stock pip dependency, which
matters for reproducibility — a patched vendored copy is exactly the kind of
thing that stops being rebuildable after two years.

**Alternative rejected.** `model.predict(..., visualize=True)` writes per-stage
feature maps to disk automatically. Useful for *validating* a layer selection
quickly, but it renders the first 32 channels as a fixed tile image and gives
no control over normalisation, so it cannot drive the live view.

**Note.** The hook deliberately does *not* call `.cpu()`. Doing so synchronises
the GPU once per layer per frame and transfers full-resolution tensors
(megabytes). Reduction to what is actually displayed happens in the renderer,
on the GPU. This is the single largest throughput decision in the file.

---

## 3. Layer selection

**Decision.** A small set of indices in `Config.targets`, chosen to span the
network: stem, several backbone stages, SPPF, C2PSA, and neck outputs.

**Why these.** The didactic arc is resolution and selectivity changing with
depth. C2PSA (attention) is worth including specifically because it has no
YOLOv7 counterpart and its activations are spatially far more selective than
plain convolution stages — it reads well to an audience.

**Important.** *The indices are not portable across model sizes or
generations.* Verify them before a demonstration:

```bash
DUMP_STRUCTURE=1 feature-viz
```

This prints the module list with the current targets marked.

**Verified 2026-09-21** against `yolo26n.pt` (24 modules, indices 0–23). The
GPU targets `[2, 4, 9, 10, 16, 22]` resolve to C3k2, C3k2, SPPF, C2PSA, C3k2
and C3k2; the CPU subset `[4, 9, 16]` to C3k2, SPPF and C3k2. Index 23 is
`Detect` and is deliberately not a target. This holds for **this weights file
only** — re-run the dump after changing `WEIGHTS`, since a different scale or
generation renumbers the chain.

---

## 4. Normalisation

This is where most of the visual quality lives, and where the original
implementation had the most room.

**Decision, in four parts:**

1. **One scale per layer, shared across all tiles of that layer** — not one
   scale per tile.
2. **Separate scales between layers.**
3. **EMA smoothing of the scale across frames** (`alpha`, default 0.08).
4. **Percentile clipping** (1st/99th) rather than exact min/max, estimated on a
   subsample, recomputed every `update_every` frames.

**Why.**

1. Per-tile normalisation makes every channel look equally "full" and destroys
   the information that a given channel is barely firing on this input. For a
   demonstrator that is actively misleading. A shared scale shows at a glance
   which filters respond.
2. Activation magnitudes differ by orders of magnitude between early and deep
   layers. A global scale would black out most of the display.
3. Per-frame normalisation is the main cause of brightness flicker in video:
   the scale jumps whenever the scene changes. The EMA removes it. Set
   `alpha=1.0` to recover the original per-frame behaviour for still images.
4. A single outlier activation drags the whole grid dark under exact min/max.
   Percentiles on a subsample also avoid sorting ~10⁶ values six times per
   frame, which was the most expensive single operation in the naive version.

**Gamma.** SiLU activations are strongly right-skewed, so a linear map leaves
much of the structure near black. Gamma ≈ 0.65 lifts the midtones. It is
applied as a 256-entry LUT on `uint8`, not as a floating-point power over
millions of pixels.

**Channel selection.** Channels are ranked by RMS activation and the top
`max_channels` are shown. "The first N channels" is arbitrary — channel order
carries no meaning. The ranking is computed **once, on the first frame, and
then frozen**; recomputing per frame makes tiles swap positions constantly and
the grid becomes unreadable.

**Tile scaling.** `INTER_NEAREST`, deliberately. Smooth interpolation hides the
coarse grid of deep layers, which is precisely the thing the demonstrator is
supposed to show.

---

## 5. One code path for GPU and CPU

**Context.** Demonstrations run on NVIDIA hardware (RTX 4090). Some machines at
the chair have no dedicated GPU, and the demonstrator must still start there
for functional testing and development.

**Decision.** A single code path. GPU and CPU differ only in a parameter set
(`build_config()`): input size, number of target layers, tile size, and how
often the visualisation is refreshed. Extraction, normalisation and rendering
are identical.

**Why.** A second implementation is exactly the failure mode this project is
trying to avoid. Whoever built the second path knows why it differs; once that
person leaves, a successor faces two implementations, one of which is broken,
and no way to tell which.

**`FORCE_CPU=1`** forces the reduced profile on a GPU machine. A fallback that
is only ever exercised on someone else's hardware is usually broken by the time
it is needed.

**On CPU the detection still runs every frame**; only the visualisation is
refreshed less often and the previous grid stays on screen. Dropping detection
frames looks broken; a slightly stale feature grid does not.

---

## 6. ONNX / OpenVINO export — rejected

**Context.** Considered as a way to get usable frame rates on CPU-only
machines.

**Decision.** Rejected.

**Why.** Feature-map extraction *is* possible under ONNX Runtime — forward
hooks are a PyTorch construct, but the intermediate tensors can be marked as
additional graph outputs and come back from a single `session.run()`. Node
names carry the Ultralytics module hierarchy (layer 9 appears as
`/model.9/...`), so the mapping is tractable.

It was rejected because:

- Demonstrations only ever happen on GPU machines. The CPU path is a functional
  test, not a presentation mode, so the performance gain buys nothing.
- Marked tensors block Conv/BN/activation fusion.
- The returned feature maps are full-resolution, all-channel NumPy arrays. The
  GPU-side reduction that makes the live path fast is unavailable, so CPU pays
  the full reduction cost anyway unless `Gather`/`Resize` nodes are surgically
  inserted into the graph.
- Above all: it is a second code path. See §5.

**Revisit when.** Demonstrations are required on machines without CUDA. In that
case prefer OpenVINO over ONNX Runtime on x86 — usually noticeably faster.

---

## 7. Output: MJPEG over HTTP, not an OpenCV window

**Decision.** Default output is an MJPEG stream on port 8080.
`DISPLAY_MODE=window` restores `cv2.imshow` for local development.

**Why.** `cv2.imshow` requires a desktop session. Under X11 that means
`DISPLAY`, a mounted `/tmp/.X11-unix` and an `xhost` grant; under Wayland it
goes through XWayland because the OpenCV pip wheels have no native Wayland
backend. In a container this is the single most fragile part of the setup.

Streaming removes the problem entirely: no desktop access, no socket mounts,
and the output can be put on a projector, a tablet or a second machine without
sitting at the demo box. JPEG encoding costs a few milliseconds per frame,
which is affordable inside a 33 ms budget.

**Buffering.** The server holds only the most recently encoded frame. A slow
client skips frames instead of throttling the inference loop.

**`/healthz`** exists for the container health check. It should eventually
report "last frame newer than N seconds" rather than "process alive" — a hung
camera leaves the process running with no frames coming out, and that is the
case worth catching.

---

## 8. Source selection and the sample video

**Decision.** `SOURCE` selects the input. With no explicit value the code
prefers `assets/sample.mp4` and only falls back to the webcam.

**Why.** It makes the webcam an optional extra rather than a startup
requirement. Anyone can clone the repository on any machine and see the
demonstrator work within minutes, with no device paths, no group permissions
and no camera. That is the difference between "it doesn't run and nobody knows
why" and "it runs, the camera just isn't wired up yet".

**TODO:** `assets/sample.mp4` does not exist yet. Ten seconds of footage with a
few recognisable objects is enough. Check licensing before committing it.

**Webcam caveat.** Many UVC cameras default to YUYV rather than MJPG and drop
to 5–10 FPS at 1080p regardless of model speed. This is the most common cause
of "the demonstrator is not smooth". Note that `model.predict(source=0)` opens
the stream internally, so camera parameters are not reachable from here. If
this becomes a problem, capture with an explicit `cv2.VideoCapture` (fourcc
MJPG, explicit FPS) and pass frames to `predict` as arrays.

---

## 9. Containerisation (planned, not yet implemented)

**Goal.** Reproducibility for the project record, and the ability to run on
other machines at the chair. A `requirements.txt` is only reproducible for as
long as PyPI still serves those versions and the wheels still match the local
CUDA — which is precisely what makes a three-year-old demonstrator unusable.

**Decisions taken so far:**

- **Pin a versioned Ultralytics image tag, not `latest`.** `latest` tracks the
  most recent main-branch build; versioned tags are bound to a release. A
  demonstrator that behaves differently after a rebuild defeats the purpose.
- **Use CDI device requests (`--device nvidia.com/gpu=all`), not `--gpus all`.**
  The legacy flag can lose GPU access when the host reloads systemd during
  routine package updates, surfacing as `Failed to initialize NVML: Unknown
  Error`. Requires Docker ≥ 28.2.0 and nvidia-container-toolkit ≥ 1.18. This
  failure mode likes to appear in the middle of a demonstration.
- **Bake the weights into the image at a fixed absolute path** and point
  `WEIGHTS` at it. See §11 — by default they land in the current working
  directory, which in a container is an overlay layer that disappears with
  `--rm`, leaving a demonstrator that cannot start without network access.
- **Set `YOLO_CONFIG_DIR`** (e.g. `/tmp/ultralytics`). If the container runs as
  a UID without a writable home, Ultralytics cannot create `settings.json` and
  warns on every start.
- **`--ipc=host`**, otherwise PyTorch can stall on the default shared-memory
  limit.
- **Camera device**, when used: prefer `/dev/v4l/by-id/...` over `/dev/video0`.
  UVC cameras expose several `videoN` nodes (the second is often a metadata
  node with no image) and the numbering can shift across reboots. Add
  `--group-add video` if not running as root.
- **Single entry point.** `docker compose up` and nothing else. A container
  nobody knows how to start is as dead as an orphaned venv.
- **Archive with `docker save`.** The resulting tarball is a complete runnable
  system, independent of whether PyPI, Docker Hub or the Ultralytics weights
  URL still resolve in five years. As a project artefact this is far more
  robust than a dependency list.

**Known limitation.** Docker does not abstract the GPU. A target machine still
needs an NVIDIA card, a current driver and the container toolkit. Without a CDI
device the same image runs on CPU via the §5 fallback, which is the intended
behaviour.

---

## 10. Unverified

Everything below was reasoned about but not measured. Verify before relying on
it.

- **Frame rate.** 30 FPS is plausible arithmetic, not a measurement. A YOLO26n
  on a 4090 is 1–2 ms per frame and the visualisation path should be similar,
  against a 33 ms budget — but the interaction of `stream=True` with the MJPEG
  thread has not been observed under load.
- **Display path cost.** `np.hstack` plus JPEG encoding is CPU memory
  bandwidth. Should be uncritical below ~1920×1080 total canvas.
- **FP16.** `half=True` is not needed at nano scale on a 4090. If enabled at
  larger scales, note that `torch.quantile` does not accept `float16` on CUDA;
  `Scale.get` already casts with `.float()` for this reason.
- **pyright has never been run.** `pyproject.toml` configures it (basic mode)
  but it is not installed; only mypy is. The two need not agree — see §13.

---

## 11. Where Ultralytics stores weights

Relevant to §9 and a recurring source of confusion.

Given a bare filename such as `yolo26n.pt`, resolution order is:

1. The path as given, relative to the current working directory.
2. `SETTINGS["weights_dir"]`.
3. Download from the `ultralytics/assets` GitHub releases — **into the current
   working directory**, not into `weights_dir`.

So `weights_dir` is consulted when *searching* but is not the download target.
Passing a URL instead of a filename does download into `weights_dir`.

Settings live in `~/.config/Ultralytics/settings.json` (inspect with
`yolo settings`); the directory can be moved with `YOLO_CONFIG_DIR`.

Practical effect: starting the script from a different directory re-downloads
the weights. Reference an absolute path via `WEIGHTS` to avoid scattered
copies.

---

## 12. If runtime control is ever needed

Currently the configuration is fixed at startup and the HTTP server is
hand-written on `http.server`. That is the right trade for one endpoint serving
bytes: no extra dependency, nothing to break in five years.

The threshold for switching to FastAPI is **the second endpoint with a request
body** — at that point manual `Content-Length` parsing, validation and error
responses stop being worth writing by hand, and the generated OpenAPI page at
`/docs` becomes a real asset for handover: a successor can see what is
adjustable without reading the code.

If that happens, the structure should be:

```
core/      Config, FeatureTap, Scale, GridRenderer   — knows nothing about HTTP
engine.py  pipeline thread, frame buffer, control surface
api.py     FastAPI — calls into the engine and nothing else
```

**The one rule that matters:** the image-processing core must not know about
HTTP. If web-framework types leak into the renderer, replacing the framework
later means rewriting the project.

**The one trap that matters:** FastAPI is async, the inference loop is
blocking. The loop must stay an ordinary `threading.Thread`. Write the MJPEG
endpoint as a **synchronous** `def` returning a `StreamingResponse` — Starlette
iterates sync generators in a threadpool, so a blocking wait inside is safe.
An `async def` there blocks the event loop and stalls the whole server.

Parameters would then split by cost: gamma/alpha/quantiles/vis_every take
effect on the next frame; targets/max_channels/tile require re-registering
hooks and resetting calibration; weights/imgsz/source/device require a pipeline
restart. Configuration changes must be staged under a lock and picked up at the
top of the loop, never mid-frame — half-old, half-new parameters combined with
the EMA produce a state nobody can explain.

---

## 13. Type checking: narrow aliases plus casts

**Context.** The project requires `mypy src/feature_viz/demonstrator.py
--ignore-missing-imports` to stay clean, but mypy was not actually installed —
the claim in the module docstring predates the current NumPy. Adding it (mypy
2.3.1, NumPy 2.x) produced five errors in code nobody had changed, all of them
stub imprecision rather than defects:

- Four from OpenCV. `cv2.LUT`, `cv2.cvtColor`, `cv2.copyMakeBorder` and
  `cv2.resize` are stubbed as returning `ndarray[Any, dtype[integer |
  floating]]`. NumPy 2 made `ndarray` generic over the shape, so that union no
  longer unifies with `NDArray[np.uint8]`.
- One from Ultralytics. `model.predict()` is annotated `Iterator[Results |
  Tensor] | list[Results] | list[Tensor]`, because the return type depends on
  the `stream` argument — which a signature cannot express.

**Decision.** Keep `GrayImage` and `BGRImage` as `NDArray[np.uint8]` and
`cast()` at the five call sites.

**Why.** The aliases are the only place the uint8 expectation is written down;
widening them to `NDArray[Any]` would delete the information the annotations
exist to carry, and it would do so everywhere in order to satisfy four lines. A
cast is local and greppable, and it names the type being asserted at the point
where the assertion is made.

**Rejected.** `# type: ignore[assignment]` — silences an error category on a
line rather than stating what the value is, and keeps silencing it after the
line changes. Writing cv2 stubs — an unbounded maintenance obligation against a
library that ships its own. Dropping the mypy requirement — the annotations are
load-bearing for readers who did not write this.

**The trap.** A cast asserts, it does not check. If a cv2 call ever returned a
non-uint8 array, mypy would now stay silent about it. The runtime dtypes were
confirmed once, by exercising `GridRenderer.render` and `compose` on synthetic
input; nothing keeps them confirmed.

**The `predict` cast target must stay a string.** `Results` is imported under
`TYPE_CHECKING` only, and `cast()` evaluates its first argument at runtime — a
bare `cast(Iterator[Results], ...)` raises `NameError` when the demonstrator
starts, and no type checker will warn about it.

**Tooling note.** `pyproject.toml` also carries a `[tool.pyright]` section in
basic mode. Two checkers are configured; only mypy is installed and run. See
§10.
