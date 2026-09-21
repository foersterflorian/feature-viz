# CLAUDE.md

Project instructions for Claude Code. Keep this file short — it is prepended to
every session.

## What this is

A demonstrator for a research project: live object detection (Ultralytics
YOLO26) shown next to the feature maps of selected network layers, rendered as
greyscale tile grids. Ported from an older YOLOv7-based version.

It is shown at talks and handed to colleagues who did not write it. Clarity and
reproducibility outrank cleverness.

## Read first

`DECISIONS.md` records why the code is shaped the way it is, including the
alternatives that were rejected and why. Read it before changing anything
structural. If you change one of those decisions, update that section — do not
delete it.

`DECISIONS.md` §10 lists what has been reasoned about but never measured. Treat
those as open tasks, not as facts.

## Conventions

- **English only.** Code, comments, docstrings, commit messages, documentation.
- **Full type annotations** on every function signature, class attribute, and
  any local whose type is not obvious. `mypy src/feature_viz/demonstrator.py
  --ignore-missing-imports` must stay clean (ultralytics ships no stubs).
- **One code path.** GPU and CPU differ by a parameter set in
  `build_config()`, never by branching logic. A second implementation is the
  failure mode this project exists to avoid.
- **Standard library first.** Every dependency is a future breakage. The
  threshold for adding one is stated per case in `DECISIONS.md` §12.
- Comments explain *why*, not *what*. The annotations already say what.

## Commands

```bash
feature-viz                              # auto profile, stream on :8080
DUMP_STRUCTURE=1 feature-viz             # print the module list with targets marked
FORCE_CPU=1 feature-viz                  # exercise the reduced profile on a GPU box
DISPLAY_MODE=window feature-viz
SOURCE=0 feature-viz                     # force webcam
WEIGHTS=/abs/path/yolo26n.pt feature-viz

mypy src/feature_viz/demonstrator.py --ignore-missing-imports
```

`feature-viz` is the console script declared in `pyproject.toml`. Outside an
activated venv, prefix with `pdm run`. `python -m feature_viz.demonstrator` is
the equivalent long form.

Target hardware: Pop!_OS, RTX 4090, AMD 16-core. CUDA is available.

## Rules

- **Verify, do not assert.** This project has a history of plausible-sounding
  numbers. If you change layer indices, run `DUMP_STRUCTURE=1` and read the
  actual output. If you touch the render path, run it and check the frame rate.
  If you cannot verify something, say so and add it to `DECISIONS.md` §10.
- **Never patch the installed `ultralytics` package.** Feature extraction goes
  through forward hooks precisely so the dependency stays stock.
- **Weights:** a bare filename downloads into the current working directory,
  not into `weights_dir`. Use an absolute path via `WEIGHTS`. See
  `DECISIONS.md` §11.
- Commit before starting a substantial change, so it can be reverted cleanly.
