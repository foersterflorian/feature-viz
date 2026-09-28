"""Funding notice (DECISIONS.md §17)."""

from __future__ import annotations

import re

import cv2
import numpy as np
import pytest

from feature_viz import demonstrator as demo
from tests.support import ROOT


def test_notice_in_code_matches_readme_verbatim() -> None:
    """One wording everywhere. The README links the project name, so the
    Markdown link is stripped before comparing."""
    readme: str = (ROOT / "README.md").read_text(encoding="utf-8")
    section: str = readme.split("## Funding", 1)[1].split("\n\n", 2)[1]
    plain: str = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", section).strip()
    assert plain == demo.FUNDING_TEXT


def test_readme_shows_the_packaged_logos() -> None:
    readme: str = (ROOT / "README.md").read_text(encoding="utf-8")
    for name, _ in demo.FUNDING_LOGOS:
        assert f"src/feature_viz/funding/{name}" in readme


@pytest.mark.parametrize("name", [n for n, _ in demo.FUNDING_LOGOS])
def test_logos_load_from_the_package(name: str) -> None:
    raw: np.ndarray = np.frombuffer(demo.funding_asset(name), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    assert img is not None and img.ndim == 3


@pytest.mark.parametrize("width", [3790, 2627, 1200])
def test_wrap_keeps_every_word_and_fits_the_width(width: int) -> None:
    lines: list[str] = demo._wrap(demo.FUNDING_TEXT, demo.FUNDING_TEXT_SCALE, width)
    assert " ".join(lines).split() == demo.FUNDING_TEXT.split()
    for line in lines:
        w: int = cv2.getTextSize(line, demo.FONT, demo.FUNDING_TEXT_SCALE, 1)[0][0]
        assert w <= width


@pytest.mark.parametrize("width", [3790, 2627, 1200])
def test_strip_is_white_and_at_least_logo_high(width: int) -> None:
    strip: demo.BGRImage = demo.funding_strip(width)
    assert strip.shape[1] == width and strip.shape[2] == 3
    assert strip.shape[0] >= demo.FUNDING_STRIP_H
    # The BMFTR logo must sit on white (§17): the last column is pure ground.
    assert np.all(strip[:, -1] == 255)


def test_strip_is_built_once_per_width() -> None:
    assert demo.funding_strip(2000) is demo.funding_strip(2000)


@pytest.mark.parametrize("width", [1, 50, 200])
def test_strip_survives_a_canvas_narrower_than_the_logos(width: int) -> None:
    assert demo.funding_strip(width).shape[1] == width
