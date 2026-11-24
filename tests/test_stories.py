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
