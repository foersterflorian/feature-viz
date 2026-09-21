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
matters because the demonstrator *is* a network service (§7). This decided the
project's own licence — see §16. If it ever becomes a problem, RF-DETR
(Apache 2.0) is the alternative, but its layer structure differs substantially
and the visualisation would need rework.

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
`Detect` and is deliberately not a target. The same dump on `s`, `m`, `l` and
`x` marks the same six block types, so the indices are stable **across the
YOLO26 scales** — but not across generations. Re-run the dump after changing
`WEIGHTS` to anything that is not a YOLO26 checkpoint.

### 3.1 Model scale: the demonstrator stays on `yolo26n`

**Decision.** `yolo26n.pt` remains the default, and the reason is this section,
not performance.

**Why.** `max_channels = 64` caps how many tiles a layer can contribute, so a
wider model does not show more — it shows a smaller fraction of itself.
Channel counts at the six target layers, measured 2026-09-21:

| Weights | Channels at L2, L4, L9, L10, L16, L22 | Visible in the grid |
|---|---|---|
| `yolo26n.pt` | 64, 128, 256, 256, 64, 256 | 384 of 1024 — **38 %** |
| `yolo26s.pt` | 128, 256, 512, 512, 128, 512 | 384 of 2048 — 19 % |
| `yolo26m.pt` | 256, 512, 512, 512, 256, 512 | 384 of 2560 — 15 % |
| `yolo26l.pt` | 256, 512, 512, 512, 256, 512 | 384 of 2560 — 15 % |
| `yolo26x.pt` | 384, 768, 768, 768, 384, 768 | 384 of 3840 — **10 %** |

On `yolo26n`, layers 2 and 16 have exactly 64 channels and are therefore shown
**in full**. "This is what the layer computes" is then literally true. On `x`
the audience sees the top 10 % by activation energy, and the honest phrasing
becomes "this is the selection we find most interesting". For a demonstrator
whose entire purpose is to make the layer visible, that is the expensive loss,
and it appears in no frame-rate measurement.

**Speed does not decide this.** Every scale holds above 30 FPS on the GPU
profile (§14.1), so throughput is not an argument for or against a larger
model here.

**The CPU fallback does decide it.** Both profiles load the same weights file,
and §5 depends on `FORCE_CPU=1` staying demonstrable. Reduced profile: `n`
41.6 FPS, `m` 13.8, `x` 7.0. With `x` the fallback is no longer a fallback.

**When to revisit.** Only if detections on the real presentation material are
visibly inadequate — detection *quality* was never assessed, only cost. The
step would then be `m`, never `l` or `x`: those cost the most frame rate on the
GPU, end the CPU path, and show the smallest fraction of each layer.

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
sitting at the demo box. JPEG encoding was assumed to cost "a few
milliseconds"; measured it is **10.2 ms**, the single largest item in the
frame budget (§14). It is still affordable inside 33 ms, but it is the first
thing to trade away if headroom is ever needed.

**Buffering.** The server holds only the most recently encoded frame. A slow
client skips frames instead of throttling the inference loop. The skipping
path has not been exercised: the one client measured kept up with every
published frame (§14).

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

- **Everything in §14 holds for one machine and one clip.** Camera capture,
  several simultaneous clients and browser-side decoding are still unmeasured.
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

---

## 14. Measured performance

Measured 2026-09-21. This replaces the frame-rate and display-cost estimates
that stood in §10; both were wrong about *where* the time goes, though the
headline number held.

**Machine.** Pop!_OS, **RTX 4070 Ti (12 GB)**, Ryzen 7 5700G (16 threads),
driver 580.173.02, torch 2.14.0+cu130, ultralytics 8.4.157, OpenCV 5.0.0.
Note this is **not** the RTX 4090 named as target hardware in `CLAUDE.md` —
the numbers below are a lower bound for the demonstration machine.

**Method.** A 1200-frame 1280×720 clip built from the ultralytics `bus.jpg`
asset with a slow pan and brightness drift, so that consecutive frames differ
and roughly five objects are detected throughout. 60 frames of warm-up
discarded, then 400 frames measured. The harness reuses `FeatureTap`,
`GridRenderer` and `compose` and mirrors the body of `main()`. A video source
is used deliberately: a webcam would measure the camera's 30 Hz, not the
pipeline.

| Configuration | FPS (mean / median / p5) | Loop |
|---|---|---|
| GPU profile, default | 34.0 / 34.5 / 29.7 | 29.5 ms |
| GPU profile, one MJPEG client attached | 32.5 / 33.0 / 28.9 | 30.9 ms |
| Detection only, no feature maps | 90.5 / 90.7 / 85.6 | 11.1 ms |
| CPU profile (`FORCE_CPU=1`) | 40.9 / 41.2 / 35.4 | 24.6 ms |

Repeat runs agree to about 1%. These figures are **after** the caption strips
of §15; before them the GPU profile measured 36.3 FPS on a 3806×1160 canvas.
The strips cost 2.3 FPS, which is the price of a legible label.

**The 30 FPS target is met, for the wrong reason.** Inference is not the
bottleneck. Mean milliseconds per frame in the GPU profile:

```
ultralytics (pre + inference + post)   7.8
JPEG encode                           10.4
compose (resize + hstack)              4.9
feature-map render                     4.9
result.plot()                          1.0
```

**Read that split with care.** CUDA is asynchronous, and the first
`.cpu()` in `GridRenderer.panel` is what forces the wait, so part of the GPU
time is charged to `render` rather than to `inference`. The line items are
indicative; only the loop total is measured end to end.

The display path is **15.3 ms, over half the frame budget**, and JPEG encoding
alone costs more than the network. §10 previously assumed this was uncritical
"below ~1920×1080 total canvas" — but the canvas is **3806×1160**, 4.4
megapixels, more than twice that threshold. The estimate was not conservative,
it was aimed at the wrong number.

**The visualisation costs about 16.5 ms per frame** — detection alone runs at
90 FPS. That is the price of the demonstrator, and it is paid on the CPU, not
the GPU. If more headroom is ever needed, `jpeg_quality`, the canvas size and
`vis_every` are the levers, in that order; a faster GPU buys comparatively
little.

**`stream=True` and the MJPEG thread do not fight.** One local client costs
4.5% throughput and received all 400 published frames — nothing was skipped.
The frame-skipping path in `FrameBuffer` was therefore never exercised here;
it remains untested under a slow or remote client.

**The CPU profile is not slower — it shows less.** At 41.6 FPS it beats the GPU
profile, because it runs a smaller input, three layers instead of six and
refreshes the grid every third frame. "Degraded" in §5 refers to what is on
screen, never to the frame rate. On this box CPU inference is 13.4 ms against
7.8 ms on the GPU; the rest of the difference is the smaller canvas.

### 14.1 Model scale: every scale stays above 30 FPS

Measured on the same clip and harness, GPU profile, all five weights files:

| Weights | Params | FPS (mean / median) | Loop | Isolated forward |
|---|---|---|---|---|
| `yolo26n.pt` | 2.6 M | 36.0 / 36.5 | 27.9 ms | 9.08 ms |
| `yolo26s.pt` | 10.0 M | 35.5 / 36.0 | 28.3 ms | 9.17 ms |
| `yolo26m.pt` | 21.9 M | 35.1 / 35.6 | 28.6 ms | 9.76 ms |
| `yolo26l.pt` | 26.3 M | 31.5 / 31.9 | 31.6 ms | 14.10 ms |
| `yolo26x.pt` | 59.0 M | 32.0 / 32.3 | 31.3 ms | 14.90 ms |

This sweep was measured before the caption strips of §15 and is therefore
about 2 FPS optimistic in absolute terms; the comparison between scales is
unaffected, since all five carry the same display path.

`l` and `x` were each run twice (31.8/31.2 and 32.0/32.0 FPS). They are not
distinguishable in the loop: `x` measures marginally faster than `l` despite
the slower isolated forward, and the gap is inside the run-to-run spread.

"Isolated forward" is `model.model(x)` on a fixed 1×3×640×640 tensor with
`torch.cuda.synchronize()` around the loop — no capture, no visualisation.

**Twenty-three times the parameters cost four FPS.** The reason is that at
batch 1 the forward is bound by kernel-launch latency, not by arithmetic. For
`yolo26m`, batch 1 takes 9.53 ms and batch 2 takes 10.83 ms — doubling the work
adds 1.3 ms. Only from batch 4 does it scale with the work (22.1 / 50.0 /
107.4 ms for 4 / 8 / 16), settling near 6.2 ms per image. At batch 1 the GPU
spends most of its time idle between small kernels.

This is also why the step that costs something is **m → l**, not `x`'s jump in
parameter count: `l` and `x` are the deeper configurations, so they issue more
kernels. `x` has 2.2× the parameters of `l` for 0.8 ms more. Parameter count is
the wrong axis here; the number of launched kernels is the right one.

**Consequence for the demonstrator.** A larger model is not the thing that
breaks real time here; the display path is. `n` through `m` costs about 1 FPS
in total, and even `x` still runs at 32 FPS. Model choice is therefore not a
speed question — which is the opposite of the assumption the profile in
`build_config()` was written under. What does decide it is channel coverage in
the grid and the CPU fallback: see §3.1.

The target indices are unchanged across **all five scales** — verified by dump,
all six resolve to the same block types (§3).

**Caveats.** All of this is batch 1 at 640 px on a 12 GB card; `x` fits, but
there is no headroom study. The fixed overhead is paid on the **CPU** issuing
kernels, so a machine with a faster GPU but a slower single core will not
necessarily do better. Detection *quality* was not assessed at all — this
section is about cost, and the larger models are worth their millisecond only
if they visibly detect better on the material actually shown.

---

## 15. Labels: a caption strip, not text on the tiles

**Context.** Each panel carried its layer label, and the detection frame its
FPS and pipeline line, drawn straight onto the image with `cv2.putText` in
amber and cyan at scale 0.42 and 0.5. At a talk they were unreadable. Two
reasons compound: feature-map tiles are bright and high-frequency, so no
colour holds up against them, and the finished canvas is about 3800 px wide,
which a 1920-wide projector or browser window downscales by half — halving the
effective type size with it.

**Decision.** A black strip stacked above each image, carrying white text at
scale 0.7 (panels) and 1.1 / 0.75 (frame). One helper, `caption()`, serves
both.

**Why not an outline.** Drawing a black stroke behind bright glyphs is the
usual trick and was tried first. It is readable, but it sits on top of the
tiles, so it competes with exactly the content the demonstrator exists to
show, and at half scale the stroke thickens into a smudge. A strip is
unambiguous at any scale and covers no data.

**The strip must not grow the canvas.** The first version captioned the frame
after fitting it to the target height. That made the frame taller than the
grid, so the grid was scaled up to match and the canvas grew 15% — costing
6 FPS, from 36.3 to 30.2. `compose` now fits the frame to the target height
*minus* the strip, so the strip is free in area terms. Captioning after the
fit also keeps the text at a fixed pixel size instead of one that depends on
the camera's resolution.

**Cost.** 2.3 FPS, 36.3 to 34.0 (§14), from the six extra `putText` calls, the
per-panel stack and the 4% larger canvas. Writing each panel directly into a
preallocated buffer instead of `np.vstack` was measured and saves 0.16 ms of a
29.5 ms loop — not worth the loss of clarity.

---

## 16. Licence: AGPL-3.0-or-later, not MIT

**Context.** The repository started with MIT, the author's default, and is
public on GitHub. Of the dependencies, only `ultralytics` is copyleft
(AGPL-3.0-or-later); torch, OpenCV and NumPy are permissive.

**Decision.** The whole project is AGPL-3.0-or-later.

**Why MIT was not tenable.** The program is functionless without ultralytics:
it imports `YOLO`, registers forward hooks in its module chain and reads the
attributes ultralytics attaches at runtime (§2). Distributed together that is a
combined work, and the recipient cannot be granted it under MIT. Labelling the
repository MIT would tell readers something untrue about what they receive.

**§13 is the sharp edge.** The demonstrator binds `0.0.0.0` and serves an HTML
page plus an MJPEG stream (§7), so users interact with it over a network.
AGPL §13 then requires that they be offered the Corresponding Source. That
obligation exists regardless of what the LICENSE file says; declaring MIT would
only have hidden it. `PAGE` therefore carries a footer with copyright, licence
and a source link, which also satisfies §5(d) for the interactive interface.

**Rejected alternatives.**

- *MIT for our own files plus a note that ultralytics is AGPL.* Common
  practice, but it works only where the permissive part is independently
  useful. Here it is not, and it leaves §13 unaddressed.
- *Make the code model-agnostic and ship a permissive default backend.* The
  hooks, renderer, mosaic and HTTP server genuinely do not depend on
  ultralytics, so this is feasible — but the layer indices would have to be
  re-established per backend (§3), and the point of the project is YOLO26.
  Worth revisiting only if industrial reuse becomes a concrete requirement.
- *Ultralytics Enterprise Licence.* Removes the obligations, costs money and
  procurement effort, and contradicts the intent to be open.

**Consequences.** Anyone building on this must publish their changes under the
same terms, including when they only expose it as a service. For a publicly
funded research demonstrator that is the intended outcome, but it does bar
closed reuse by an industry partner without a separate agreement.

**Open, not decided here.** Whether copyright sits with the author or with TU
Chemnitz as the employer, and whether any funder imposes licence conditions,
was not assessed. The notice currently names the author as given in
`pyproject.toml`. Both questions belong to the institution, not to this file.

**Weights** are not redistributed: `*.pt` is git-ignored and downloaded at
runtime (§11), so Ultralytics' model files carry their own terms and are not
conveyed by this repository.