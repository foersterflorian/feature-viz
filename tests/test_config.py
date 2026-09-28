"""Profiles, environment and source handling (DECISIONS.md §5, §8, §11)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from feature_viz import demonstrator as demo
from tests.support import ROOT


def test_gpu_profile_when_cuda_is_available(gpu_cfg: demo.Config) -> None:
    assert gpu_cfg.device == "cuda"
    assert gpu_cfg.targets == [2, 4, 9, 10, 16, 22]


def test_cpu_profile_when_cuda_is_absent(cpu_cfg: demo.Config) -> None:
    assert cpu_cfg.device == "cpu"
    assert cpu_cfg.targets == [4, 9, 16]


def test_force_cpu_overrides_an_available_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """§5: FORCE_CPU=1 must exercise the reduced profile on a GPU box."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("FORCE_CPU", "1")
    assert demo.build_config().device == "cpu"


def test_cpu_targets_are_a_subset_of_gpu_targets(
    cpu_cfg: demo.Config, gpu_cfg: demo.Config
) -> None:
    """§5: the reduced profile shows less of the same thing, never something
    different."""
    assert set(cpu_cfg.targets) <= set(gpu_cfg.targets)


def test_profiles_share_everything_but_the_profile_parameters(
    cpu_cfg: demo.Config, gpu_cfg: demo.Config
) -> None:
    """§5: GPU and CPU differ by a parameter set, not by logic. Anything not
    in the profile must be identical between the two."""
    profile: set[str] = {"device", "imgsz", "targets", "tile", "vis_every", "panel_cols"}
    cpu: dict[str, object] = {k: v for k, v in vars(cpu_cfg).items() if k not in profile}
    gpu: dict[str, object] = {k: v for k, v in vars(gpu_cfg).items() if k not in profile}
    lut_c = cpu.pop("gamma_lut")
    lut_g = gpu.pop("gamma_lut")
    assert cpu == gpu
    assert np.array_equal(lut_c, lut_g)  # type: ignore[arg-type]


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch, cpu_cfg: demo.Config) -> None:
    monkeypatch.setenv("WEIGHTS", "/abs/yolo26s.pt")
    monkeypatch.setenv("DISPLAY_MODE", "window")
    monkeypatch.setenv("PORT", "9001")
    monkeypatch.setenv("SOURCE", "clip.mp4")
    cfg: demo.Config = demo.build_config()
    assert (cfg.weights, cfg.display_mode, cfg.port, cfg.source) == (
        "/abs/yolo26s.pt",
        "window",
        9001,
        "clip.mp4",
    )


def test_webcam_is_the_default_source(cpu_cfg: demo.Config) -> None:
    """§8: the live picture is what the demonstrator is for."""
    assert cpu_cfg.source == "0"


def test_empty_source_means_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("SOURCE", "")
    assert demo.build_config().source == "0"


def test_existing_file_is_a_valid_source() -> None:
    assert demo.source_error(str(ROOT / "assets" / "sample.mp4")) is None


def test_missing_file_is_reported() -> None:
    error: str | None = demo.source_error("no/such/clip.mp4")
    assert error is not None and "no/such/clip.mp4" in error


def test_missing_camera_points_at_the_sample_clip() -> None:
    """§8: no silent fallback, but the one line that gets a fresh clone
    running. Index 99 exists on no machine."""
    error: str | None = demo.source_error("99")
    assert error is not None
    assert "SOURCE=assets/sample.mp4" in error


@pytest.mark.parametrize(
    "source",
    ["rtsp://camera.local/stream", str(ROOT / "assets"), str(ROOT / "assets" / "*.jpg"), "screen"],
    ids=["url", "directory", "glob", "screen"],
)
def test_other_ultralytics_sources_are_passed_through(source: str) -> None:
    """URLs, directories, globs and screen capture are valid ultralytics
    sources; the up-front check must not refuse them."""
    assert demo.source_error(source) is None


def test_gamma_lut_is_a_monotonic_full_range_uint8_table(cpu_cfg: demo.Config) -> None:
    lut: np.ndarray = cpu_cfg.gamma_lut
    assert lut.dtype == np.uint8 and lut.shape == (256,)
    assert lut[0] == 0 and lut[255] == 255
    assert np.all(np.diff(lut.astype(int)) >= 0)
    # gamma < 1 lifts the midtones (§4)
    assert lut[128] > 128


# --------------------------------------------------------------------------
# Shutdown (§10)
# --------------------------------------------------------------------------
class _Dataset:
    def __init__(self) -> None:
        self.closed: bool = False

    def close(self) -> None:
        self.closed = True


def test_close_source_closes_the_frame_reader() -> None:
    """The camera case: ultralytics' LoadStreams has close(), and it must be
    called, or its reader thread aborts the interpreter on exit."""
    dataset: _Dataset = _Dataset()
    model: SimpleNamespace = SimpleNamespace(predictor=SimpleNamespace(dataset=dataset))
    demo.close_source(model)  # type: ignore[arg-type]
    assert dataset.closed


@pytest.mark.parametrize(
    "model",
    [
        SimpleNamespace(),  # predict never ran
        SimpleNamespace(predictor=None),
        SimpleNamespace(predictor=SimpleNamespace(dataset=object())),  # file source
    ],
)
def test_close_source_tolerates_anything_without_close(model: SimpleNamespace) -> None:
    demo.close_source(model)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Looping (§8)
# --------------------------------------------------------------------------
class _FakeModel:
    """Stands in for YOLO.predict: every call is one pass over the source."""

    def __init__(self, frames_per_pass: int) -> None:
        self.frames_per_pass: int = frames_per_pass
        self.passes: int = 0

    def predict(self, **kwargs: object) -> list[int]:
        self.passes += 1
        # A regression in the empty-file guard must fail, not hang the suite.
        if self.passes > 100:
            raise RuntimeError("results() keeps restarting a source that yields nothing")
        return list(range(self.frames_per_pass))


def _take(model: _FakeModel, cfg: demo.Config, n: int) -> list[object]:
    out: list[object] = []
    for r in demo.results(model, cfg):  # type: ignore[arg-type]
        out.append(r)
        if len(out) == n:
            break
    return out


def test_video_file_starts_over(cpu_cfg: demo.Config, tmp_path: Path) -> None:
    clip: Path = tmp_path / "clip.mp4"
    clip.touch()
    cpu_cfg.source = str(clip)
    model: _FakeModel = _FakeModel(frames_per_pass=3)
    assert _take(model, cpu_cfg, 8) == [0, 1, 2, 0, 1, 2, 0, 1]
    assert model.passes == 3


@pytest.mark.parametrize("source", ["0", "rtsp://camera.local/stream"])
def test_camera_and_stream_end_the_run(cpu_cfg: demo.Config, source: str) -> None:
    """A camera that stops delivering has failed; restarting it forever
    would hide that."""
    cpu_cfg.source = source
    model: _FakeModel = _FakeModel(frames_per_pass=3)
    assert _take(model, cpu_cfg, 100) == [0, 1, 2]
    assert model.passes == 1


def test_empty_video_file_does_not_spin(cpu_cfg: demo.Config, tmp_path: Path) -> None:
    clip: Path = tmp_path / "empty.mp4"
    clip.touch()
    cpu_cfg.source = str(clip)
    model: _FakeModel = _FakeModel(frames_per_pass=0)
    assert _take(model, cpu_cfg, 5) == []
    assert model.passes == 1
