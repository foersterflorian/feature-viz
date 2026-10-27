"""The page in German or English, chosen by the staff and kept on disk
(DECISIONS.md §22)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from feature_viz import demonstrator as demo
from tests.support import serve


def post(url: str, data: bytes | None = None) -> int:
    try:
        with urlopen(Request(url, data=data, method="POST"), timeout=5) as resp:
            return resp.status
    except HTTPError as err:
        return err.code


def test_german_notice_is_the_funders_wording() -> None:
    """§17, §22: the German text comes from the funder, German quotes and
    "KMI" included; the English one is ours and checked against the README
    in test_funding."""
    german: str = demo.FUNDING_TEXT[demo.Language.DE]
    assert german.startswith("Das Forschungs- und Entwicklungsprojekt KMI wird")
    assert "„Zukunft der Arbeit" in german and "Intelligenz“" in german
    assert set(demo.FUNDING_TEXT) == set(demo.Language)


@pytest.mark.parametrize("language", list(demo.Language))
def test_every_language_has_its_texts_and_the_licence_link(language: demo.Language) -> None:
    texts: demo.Texts = demo.TEXTS[language]
    assert texts.licence.count("{licence}") == 1
    assert all(getattr(texts, name) for name in demo.Texts.__slots__)  # type: ignore[attr-defined]


@pytest.mark.parametrize("language", list(demo.Language))
def test_the_page_comes_in_its_language(language: demo.Language) -> None:
    """Every fixed word of the page in one language, the funding notice
    included; the licence link and the source offer (§16) in both."""
    texts: demo.Texts = demo.TEXTS[language]
    page: str = demo.page(language, demo.Theme.DARK, "info")
    assert f'<html lang="{language}"' in page
    for text in (
        texts.page_title,
        texts.headline,
        texts.legend,
        texts.explanation,
        texts.funding_more,
        texts.source,
    ):
        assert html.escape(text) in page
    assert html.escape(demo.FUNDING_TEXT[language]) in page
    assert html.escape(texts.licence).format(licence=demo.LICENCE_LINK) in page
    assert ">GNU AGPL v3</a>" in page
    assert f'href="{demo.SOURCE_URL}"' in page
    other: demo.Language = next(lang for lang in demo.Language if lang != language)
    assert html.escape(demo.TEXTS[other].headline) not in page


def test_english_is_the_default_and_the_language_survives_a_restart(tmp_path: Path) -> None:
    assert demo.Choices(tmp_path).current()[0] is demo.Language.EN
    demo.Choices(tmp_path).set_language(demo.Language.DE)
    assert demo.Choices(tmp_path).current()[0] is demo.Language.DE


@pytest.mark.parametrize("content", ["", "{", '{"language": "fr"}', "[]", '"de"'])
def test_an_unreadable_choice_falls_back_to_its_default(tmp_path: Path, content: str) -> None:
    (tmp_path / demo.LANGUAGE_FILE).write_text(content, encoding="utf-8")
    assert demo.Choices(tmp_path).current()[0] is demo.DEFAULT_LANGUAGE


def test_a_choice_is_written_whole(tmp_path: Path) -> None:
    demo.Choices(tmp_path).set_language(demo.Language.DE)
    assert json.loads((tmp_path / demo.LANGUAGE_FILE).read_text()) == {"language": "de"}
    assert [p.name for p in tmp_path.iterdir()] == [demo.LANGUAGE_FILE]


def test_a_choice_that_cannot_be_kept_still_takes_effect(tmp_path: Path) -> None:
    """A data directory that cannot be written must not refuse the switch."""
    blocked: Path = tmp_path / "file"
    blocked.write_text("")
    choices: demo.Choices = demo.Choices(blocked / "data")
    choices.set_language(demo.Language.DE)
    assert choices.current()[0] is demo.Language.DE


def test_the_staffs_switch_is_kept_and_the_next_page_follows(tmp_path: Path) -> None:
    """The page posts the other language and reloads (§22); the server
    keeps it on disk and fills in the next page in it."""
    with serve(tmp_path) as (url, _, _):
        assert '<html lang="en"' in urlopen(url + "/", timeout=5).read().decode("utf-8")
        assert post(url + "/choice?language=de") == 204
        with urlopen(url + "/", timeout=5) as resp:
            assert resp.headers["Cache-Control"] == "no-store"
            assert '<html lang="de"' in resp.read().decode("utf-8")
    assert demo.Choices(tmp_path).current()[0] is demo.Language.DE


def test_anything_but_one_known_choice_is_refused(tmp_path: Path) -> None:
    with serve(tmp_path) as (url, _, _):
        for query in ("", "language=fr", "colour=red", "language=de&theme=light",
                      "language=de&language=en", "language"):
            assert post(url + "/choice?" + query) == 400, query
        assert post(url + "/choice?language=de", data=b"x") == 400
        assert post(url + "/elsewhere?language=de") == 404
    assert not (tmp_path / demo.LANGUAGE_FILE).exists()


def test_the_page_switches_the_language_by_reloading() -> None:
    """Ctrl+Alt+L, once per press, posts the other language and reloads."""
    page: str = demo.page(demo.Language.EN, demo.Theme.DARK, "")
    keys: str = page.split("addEventListener(\"keydown\"", 1)[1]
    branch: str = keys.split('key === "l"', 1)[1].split('key === "h"', 1)[0]
    assert "if (!e.ctrlKey || !e.altKey || e.repeat) return;" in keys
    assert 'choose("language"' in branch and "location.reload()" in branch


def test_the_legend_holds_its_lines_whatever_the_language() -> None:
    """The explanation keeps three lines' room, which the German fills at the
    stand's size and the English does not, so that the canvas stays in place
    when the staff switch the language."""
    page: str = demo.page(demo.Language.EN, demo.Theme.DARK, "")
    rule: str = page.split("  .legend p + p {", 1)[1].split("}", 1)[0]
    assert "min-height:calc(3 * 1.4em)" in rule
