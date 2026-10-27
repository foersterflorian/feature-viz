"""The page dark or light, chosen by the staff and kept on disk
(DECISIONS.md §23)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from feature_viz import demonstrator as demo

SELECTORS: dict[demo.Theme, str] = {
    demo.Theme.DARK: ":root",
    demo.Theme.LIGHT: ':root[data-theme="light"]',
}


def page() -> str:
    return demo.page(demo.Language.EN, demo.Theme.DARK, "")


def tokens(theme: demo.Theme) -> dict[str, str]:
    """The custom properties one rule of the page's stylesheet sets."""
    rule = re.search(re.escape(SELECTORS[theme]) + r" \{([^}]*)\}", page())
    assert rule is not None, theme
    return dict(
        (name.strip(), value.strip())
        for name, value in (d.split(":", 1) for d in rule.group(1).split(";") if d.strip())
    )


def luminance(colour: str) -> float:
    """WCAG relative luminance of #rgb or #rrggbb."""
    hex_: str = colour.lstrip("#")
    if len(hex_) == 3:
        hex_ = "".join(c * 2 for c in hex_)
    channels: list[float] = [int(hex_[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_both_themes_name_the_same_colours() -> None:
    assert set(tokens(demo.Theme.DARK)) == set(tokens(demo.Theme.LIGHT))
    assert {"--bg", "--fg"} <= set(tokens(demo.Theme.DARK))


@pytest.mark.parametrize("theme", list(demo.Theme))
def test_text_reads_on_its_ground(theme: demo.Theme) -> None:
    """WCAG: text at least 4.5:1 against its ground, in both sets - the
    muted legend included."""
    t: dict[str, str] = tokens(theme)
    for name in ("--fg", "--muted"):
        assert contrast(t[name], t["--bg"]) >= 4.5, name


def test_the_page_takes_its_colours_from_the_tokens() -> None:
    """Ground and text follow the theme; the footer stays white for the
    BMFTR logo (§17) in both."""
    styles: str = page().split("<style>", 1)[1].split("</style>", 1)[0]
    body = re.search(r"\n  body \{([^}]*)\}", styles)
    footer = re.search(r"\n  footer \{([^}]*)\}", styles)
    assert body is not None and footer is not None
    assert "background:var(--bg)" in body.group(1) and "color:var(--fg)" in body.group(1)
    assert "background:#fff" in footer.group(1)


@pytest.mark.parametrize("theme", list(demo.Theme))
def test_the_page_comes_in_the_stored_theme(theme: demo.Theme) -> None:
    assert f'data-theme="{theme}"' in demo.page(demo.Language.EN, theme, "")


def test_dark_is_the_default_and_the_theme_survives_a_restart(tmp_path: Path) -> None:
    assert demo.Choices(tmp_path).current()[1] is demo.Theme.DARK
    demo.Choices(tmp_path).set_theme(demo.Theme.LIGHT)
    assert demo.Choices(tmp_path).current() == (demo.Language.EN, demo.Theme.LIGHT)


def test_the_page_switches_the_theme_without_a_reload() -> None:
    """Ctrl+Alt+H posts the other theme and only sets the attribute, so that
    the picture keeps running; the confirmation names it in the page's
    language."""
    keys: str = page().split('addEventListener("keydown"', 1)[1]
    branch: str = keys.split('key === "h"', 1)[1]
    assert 'choose("theme"' in branch and "root.dataset.theme = theme" in branch
    assert "reload" not in branch
    assert 'data-dark="Dark"' in page() and 'data-light="Light"' in page()


def test_the_type_matches_teach_by_showing() -> None:
    """The same face and the same headline and hint sizes as the crane's
    screen beside it on the stand; the legend is set like its hint."""
    styles: str = page().split("<style>", 1)[1].split("</style>", 1)[0]
    assert "font-family:system-ui, sans-serif" in styles
    assert "--text-headline:3.2vh" in styles and "--text-hint:2.4vh" in styles
    h1 = re.search(r"\n  h1 \{([^}]*)\}", styles)
    legend = re.search(r"\n  \.legend \{([^}]*)\}", styles)
    assert h1 is not None and legend is not None
    assert "var(--text-headline)" in h1.group(1) and "font-weight:600" in h1.group(1)
    assert "var(--text-hint)" in legend.group(1) and "color:var(--muted)" in legend.group(1)
