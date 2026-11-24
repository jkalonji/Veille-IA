from html.parser import HTMLParser

import dashboard

TITLE = 'Altman says "AGI is near" & x" onmouseover="alert(1)'
ART = {
    "title": TITLE, "url": 'https://e.com/?a=1&b="2"', "source": "R&D <News>",
    "country": "US", "category": "Business / Industrie", "sentiment": "neutre",
    "published": "2026-11-24",
}


def _parse(html_str):
    found = []

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            found.append((tag, dict(attrs)))

    P().feed(html_str)
    return found


def test_table_row_keeps_attributes_intact():
    tags = _parse(dashboard._articles_to_html_table([ART]))
    tr = next(a for t, a in tags if t == "tr")
    assert tr["data-title"] == TITLE.lower()  # what the JS search reads back
    assert "onmouseover" not in tr
    link = next(a for t, a in tags if t == "a")
    assert link["href"] == ART["url"] and "onmouseover" not in link


def test_hot_card_escapes_title_url_and_source():
    out = dashboard._render_hot_card_html(ART, {"border": "#fff"})
    tags = _parse(out)
    assert {t for t, _ in tags} <= {"div", "a", "span"}  # <News> stayed text
    link = next(a for t, a in tags if t == "a")
    assert link["href"] == ART["url"] and "onmouseover" not in link
