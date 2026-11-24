from html.parser import HTMLParser

import main


def test_digest_escapes_titles_and_urls(monkeypatch):
    sent = []
    monkeypatch.setattr(main, "_post_telegram", lambda token, chat, text: sent.append(text))
    art = main.Article(
        title="AT&T <beta> deal", url="https://x.com/?a=1&b=2", source="R&D Weekly",
        country="US", published="2026-11-24", category="Business / Industrie",
        hot_topic=True, summary="Q&A <live>.",
    )
    main._send_domain_digest("t", "c", "ia", [art], "")
    hot = sent[-1]
    assert "AT&amp;T &lt;beta&gt; deal" in hot
    assert "R&amp;D Weekly" in hot and "Q&amp;A &lt;live&gt;." in hot
    assert 'href="https://x.com/?a=1&amp;b=2"' in hot
    tags = []

    class Tags(HTMLParser):
        def handle_starttag(self, tag, attrs):
            tags.append(tag)

    Tags().feed(hot)
    assert set(tags) <= {"b", "i", "a"}  # the title's <beta> must not become a tag
