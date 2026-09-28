"""Profiles and environment handling (DECISIONS.md §5, §8, §11)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from feature_viz import demonstrator as demo


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


def test_empty_source_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("SOURCE", "")
    assert demo.build_config().source == demo._default_source()


@pytest.mark.xfail(
    strict=True,
    reason="§8 promises the sample clip, but _default_source() looks in "
    "src/feature_viz/assets/ while the clip lives in the repository's assets/",
)
def test_default_source_is_the_sample_clip() -> None:
    """§8: without SOURCE, a fresh clone must show the sample, not wait for
    a webcam."""
    source: str = demo._default_source()
    assert source.endswith("sample.mp4")
    assert Path(source).is_file()


def test_gamma_lut_is_a_monotonic_full_range_uint8_table(cpu_cfg: demo.Config) -> None:
    lut: np.ndarray = cpu_cfg.gamma_lut
    assert lut.dtype == np.uint8 and lut.shape == (256,)
    assert lut[0] == 0 and lut[255] == 255
    assert np.all(np.diff(lut.astype(int)) >= 0)
    # gamma < 1 lifts the midtones (§4)
    assert lut[128] > 128
