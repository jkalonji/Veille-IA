import asyncio
import copy

import main
from tests.fake_supabase import FakeSupabase


class ArticlesDown(FakeSupabase):
    """Supabase whose `articles` reads time out."""

    def table(self, name):
        q = super().table(name)
        if name == "articles":
            def boom():
                raise TimeoutError("read timed out")
            q.execute = boom
        return q


def test_failed_window_read_leaves_stories_untouched(monkeypatch):
    db = ArticlesDown()
    db.tables["stories"] = [
        {"id": 1, "domain": "ia", "topic_key": "gpt-6", "label": "GPT-6", "status": "open",
         "article_urls": ["https://a/1", "https://b/2", "https://c/3"]},
    ]
    before = copy.deepcopy(db.tables["stories"])

    async def fake_topics(items, *args, **kwargs):
        return [["Gemini 4"] for _ in items]

    async def no_review(stories, *args, **kwargs):
        return stories

    monkeypatch.setattr(main, "extract_topics", fake_topics)
    monkeypatch.setattr(main, "review_stories", no_review)
    monkeypatch.setattr(main, "_fetch_existing_stories", lambda client, domain: {})
    # Enough fresh articles on another topic that a today-only rebuild would
    # save "Gemini 4" and close the open "GPT-6" story.
    arts = [
        main.Article(title=f"Gemini 4 news {i}", url=f"https://n/{i}", source=src, country="US", published="2026-11-24")
        for i, src in enumerate(["A", "B", "C"])
    ]
    asyncio.run(main.refresh_stories("ia", arts, db, None, "m"))

    assert db.tables["stories"] == before
    assert all(a.topics == ["Gemini 4"] for a in arts)  # topics are still tagged


def test_stored_articles_are_skipped(monkeypatch):
    db = FakeSupabase()
    db.tables["articles"] = [{"url": "https://old/1", "title": "Old", "source": "A", "domain": "ia",
                              "published": "2026-11-24", "topics": ["GPT-6"]}]
    sent_to_groq = []

    async def fake_topics(items, *args, **kwargs):
        sent_to_groq.extend(i["title"] for i in items)
        return [["GPT-6"] for _ in items]

    async def no_review(stories, *args, **kwargs):
        return stories

    monkeypatch.setattr(main, "extract_topics", fake_topics)
    monkeypatch.setattr(main, "review_stories", no_review)
    monkeypatch.setattr(main, "_fetch_existing_stories", lambda client, domain: {})
    old = main.Article(title="Old", url="https://old/1", source="A", country="US", published="2026-11-24")
    new = main.Article(title="New", url="https://new/2", source="B", country="US", published="2026-11-24")

    kept = asyncio.run(main.refresh_stories("ia", [old, new], db, None, "m"))

    assert kept == [new]
    assert sent_to_groq == ["New"]


# ---------------------------------------------------------------------------
# build_story_candidates / _apply_story_flags — the rules that decide what is hot
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402


def _day(ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=ago)).strftime("%Y-%m-%d")


def _row(url, source="A", ago=0, topics=("T",)):
    return {"url": url, "title": url, "source": source, "published": _day(ago), "topics": list(topics)}


def _keys(window, existing=frozenset()):
    return {s["topic_key"] for s in main.build_story_candidates(window, set(existing))}


def test_story_needs_three_articles():
    two = [_row("1", "A"), _row("2", "B")]
    assert _keys(two) == set()
    assert _keys(two + [_row("3", "C")]) == {"t"}


def test_story_needs_two_sources():
    assert _keys([_row(str(i), "A") for i in range(3)]) == set()
    assert _keys([_row("0", "A"), _row("1", "A"), _row("2", "B")]) == {"t"}


def test_story_too_broad_over_five_percent_of_window():
    # 400 tagged articles: 5% = 20 articles max for one topic (above the floor of 15)
    filler = [_row(f"f{i}", topics=(f"filler {i}",)) for i in range(380)]
    at_cap = [_row(f"t{i}", "AB"[i % 2]) for i in range(20)]
    assert "t" in _keys(filler + at_cap)
    over = filler[:-1] + at_cap + [_row("t20", "A")]  # still 400 articles, topic on 21
    assert "t" not in _keys(over)


def test_overlapping_topics_merge_and_keep_the_stored_key():
    # "Sam Altman" covers 4 of the 5 "Altman" articles: 4/4 >= 80% -> one story
    window = [_row(str(i), "AB"[i % 2], topics=("Altman", "Sam Altman") if i < 4 else ("Altman",))
              for i in range(5)]
    assert _keys(window) == {"altman"}
    # The DB already knows the story as "sam-altman": the merge keeps that key (and its id)
    assert _keys(window, {"sam-altman"}) == {"sam-altman"}
    # Exactly 80%: 4 shared, each topic on 5 articles -> merged
    window = [_row(str(i), "AB"[i % 2], topics=("X", "Y") if i < 4 else ("X",) if i == 4 else ("Y",))
              for i in range(6)]
    assert len(_keys(window)) == 1
    # 3 shared of 4 (75%) stays two stories
    window = [_row(str(i), "AB"[i % 2], topics=("X", "Y") if i < 3 else ("X",) if i < 6 else ("Y",))
              for i in range(7)]
    assert _keys(window) == {"x", "y"}


def _story(key, rows):
    return {"topic_key": key, "label": key.upper(), "summary": "", "articles": rows}


def _art(url):
    return main.Article(title=url, url=url, source="A", country="US", published=_day())


def test_story_flags():
    supa = _story("supa", [_row(f"s{i}") for i in range(5)])                         # 5 today
    four = _story("four", [_row(f"f{i}") for i in range(4)])                         # 4 today
    hot = _story("hot", [_row("h0"), _row("h1", ago=1), _row("h2", ago=10)])          # 2 recent
    cold = _story("cold", [_row("c0"), _row("c1", ago=10), _row("c2", ago=12)])       # 1 recent
    arts = {u: _art(u) for u in ("s0", "f0", "h0", "c0", "lone")}
    arts["lone"].hot_topic, arts["lone"].story_id = True, 99                            # stale flags
    main._apply_story_flags(list(arts.values()), [supa, four, hot, cold], {"supa": 1, "hot": 2, "cold": 3, "four": 4})

    s, h, c, lone = arts["s0"], arts["h0"], arts["c0"], arts["lone"]
    assert (s.hot_topic, s.supa_hot, s.story_id, s.hot_reason, s.mention_count) == (True, True, 1, "SUPA", 5)
    assert (h.hot_topic, h.supa_hot, h.story_id, h.mention_count) == (True, False, 2, 2)
    assert (arts["f0"].hot_topic, arts["f0"].supa_hot) == (True, False)
    assert (c.hot_topic, c.story_id, c.hot_reason) == (False, 3, "")
    assert (lone.hot_topic, lone.story_id) == (False, None)


def test_story_flags_pick_the_most_active_story():
    big = _story("big", [_row("x"), _row("b1"), _row("b2")])
    small = _story("small", [_row("x"), _row("s1", ago=1)])
    art = _art("x")
    main._apply_story_flags([art], [small, big], {"big": 1, "small": 2})
    assert (art.story_id, art.hot_reason) == (1, "BIG")
