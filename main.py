"""
AI Radar - Outil de Veille IA Automatise
Collecte, classifie et envoie quotidiennement l'actualite IA sur Telegram.
"""

import asyncio
import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from time import mktime

import aiohttp
import feedparser
from groq import AsyncGroq
import requests
from supabase import create_client
from bluesky_scraper import fetch_all_bluesky

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Article:
    title: str
    url: str
    source: str
    country: str
    published: str  # "YYYY-MM-DD"
    description: str = ""
    domain: str = "ia"
    category: str = ""
    sentiment: str = ""
    hot_topic: bool = False
    mention_count: int = 0
    supa_hot: bool = False
    hot_source: str = ""    # pipe-separated detection signals: "trends|hn|github|db"
    hot_reason: str = ""    # label of the article's most active story, when it's hot
    summary: str = ""       # groq-generated 1-sentence summary in French
    story_id: int | None = None  # the article's most active story (it can belong to several)
    topics: list[str] = field(default_factory=list)  # groq-extracted topics, the basis of stories
    published_is_estimated: bool = False  # True if `published` is collection time, not a real source date

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_html(text: str) -> str:
    """Strip HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text).strip()


def parse_feed_date(entry) -> datetime | None:
    """Extract a timezone-aware datetime from a feedparser entry."""
    for attr in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, attr, None)
        if parsed:
            return datetime.fromtimestamp(mktime(parsed), tz=timezone.utc)
    return None


def compute_stats(articles: list[Article]) -> dict[str, int]:
    """Count articles per category."""
    stats: dict[str, int] = {}
    for a in articles:
        stats[a.category] = stats.get(a.category, 0) + 1
    return stats


_MENTION_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "as", "by", "from", "is", "are", "was", "were", "be",
    "been", "have", "has", "had", "will", "would", "could", "should",
    "that", "this", "these", "those", "its", "not", "new", "how", "why",
    "what", "when", "where", "who", "which", "more", "can", "all", "out",
    "over", "about", "into", "than", "their", "they", "there", "says",
    "said", "just", "also", "after", "amid",
}


def _compute_article_mentions(articles: list[Article]) -> None:
    """Set mention_count and supa_hot on each article in-place."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tokens: dict[str, set[str]] = {}
    for a in articles:
        words = re.findall(r"[a-zA-Z]{4,}", a.title.lower())
        tokens[a.url] = {w for w in words if w not in _MENTION_STOPWORDS}

    for i, a in enumerate(articles):
        mine = tokens[a.url]
        count = sum(
            1 for j, other in enumerate(articles)
            if i != j and other.published == today and len(mine & tokens[other.url]) >= 2
        ) if mine else 0
        a.mention_count = count
        a.supa_hot = a.hot_topic and count > 5 and a.published == today


# ---------------------------------------------------------------------------
# 0b. Semantic stories (replaces title n-gram clustering + cross-day matching)
# ---------------------------------------------------------------------------
# Groq tags each article with 1-4 canonical *topics*: a person, a specific
# event or matter, or a specific issue. Every run regroups the last
# STORY_WINDOW_DAYS of articles by shared topic, and a topic shared by enough
# articles is a *story*. An article can belong to several stories. Stories are
# re-derived from scratch on each run and keyed by (domain, topic_key), so a
# story keeps its id across runs as long as its topic is still there.

STORY_WINDOW_DAYS = 30
STORY_MIN_ARTICLES = 3
STORY_MIN_SOURCES = 2
STORY_MAX_SHARE = 0.05     # a topic in more than 5% of the window's articles is too broad...
STORY_MAX_FLOOR = 15       # ...unless the window is small (e.g. before the topics backfill)
STORY_MERGE_OVERLAP = 0.8  # two topics sharing this share of the smaller one's articles are one story
STORY_NAMING_LIMIT = 50    # most active stories sent to Groq for a summary
HOT_RECENT_DAYS = 2        # an article is hot when one of its stories has >= HOT_MIN_RECENT
HOT_MIN_RECENT = 2         # articles published in the last HOT_RECENT_DAYS days
SUPA_HOT_MIN_TODAY = 5
TOPIC_BATCH_SIZE = 20
KNOWN_TOPICS_LIMIT = 200   # existing topic names shown to Groq so it reuses them verbatim
GROQ_RETRIES = 6           # topic/story calls are batched and big — let the SDK wait out 429s

DOMAIN_TOPIC_HINTS: dict[str, dict[str, str]] = {
    "ia": {
        "event": "'California OpenAI Subpoena', 'Gemini 4 Argon Launch', 'Apple macOS Agent Restrictions'",
        "actor": "'Sam Altman', 'Mistral AI', 'Jensen Huang', 'Huawei'",
        "issue": "'AI Agent Security', 'OpenAI Safety Culture', 'AI Copyright Lawsuits', "
                 "'AI Chip Export Controls', 'AI Data Center Power Demand'",
        "bad": "'AI', 'AI Agents', 'LLMs', 'Generative AI', 'Machine Learning', 'Research', "
               "'Funding Round', 'Startups', 'Technology', 'Robotics', and research fields or "
               "techniques such as 'Model Compression', 'Interpretability', 'Benchmarks'",
    },
    "politique_evenements": {
        "event": "'Gaza Ceasefire Talks', 'Turkey Earthquake Response', 'EU Russia Sanctions Package'",
        "actor": "'Donald Trump', 'Hamas', 'Wagner Group'",
        "issue": "'Sudan Civil War', 'Sahel Security Crisis', 'European Energy Security'",
        "bad": "'Politics', 'War', 'Elections', 'Diplomacy', 'International Relations', 'World News'",
    },
}


def _topic_key(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")


_TOPIC_KIND_SUFFIX = re.compile(r"\s+\(?(actor|event|issue)\)?$", re.I)


def _known_names(known: Counter, fresh: list[str]) -> list[str]:
    """Topic names to show Groq: the established ones (used by several
    articles), then the ones coined during this run, then the window's other
    names newest first — `known` is built from a newest-first window, and a
    Counter keeps insertion order among equal counts. A same-day event covered
    across several batches is what most needs its first name reused."""
    established = [t for t, n in known.most_common(KNOWN_TOPICS_LIMIT // 2) if n >= 2]
    names = established + fresh[::-1] + [t for t in known if known[t] < 2]
    return list(dict.fromkeys(names))[:KNOWN_TOPICS_LIMIT]


def _clean_topics(raw) -> list[str]:
    """Keep 1-4 distinct, reasonably sized topic names from Groq's output."""
    if not isinstance(raw, list):
        return []
    topics, seen = [], set()
    for t in raw:
        if not isinstance(t, str):
            continue
        # Groq sometimes appends the kind it was asked for ("Huawei Actor")
        t = _TOPIC_KIND_SUFFIX.sub("", " ".join(t.split()))
        key = _topic_key(t)
        if 2 <= len(t) <= 60 and key and key not in seen:
            topics.append(t)
            seen.add(key)
    return topics[:4]


async def extract_topics(
    items: list[dict], known: Counter, client, model: str, domain: str, batch_pause: float = 3.0
) -> list[list[str]]:
    """Tag each {title, description} item with its topics, one Groq call per batch.

    `known` counts topic names already in use, built from the window newest
    first; a selection is shown to Groq (see `_known_names`) so it reuses an
    existing name instead of coining a variant ("Altman" vs "Sam Altman"),
    which is what lets articles from different days meet in the same story.
    Updated in place with the names coined here. An item whose batch failed
    gets [] — the backfill picks it up later.
    """
    hints = DOMAIN_TOPIC_HINTS.get(domain, DOMAIN_TOPIC_HINTS["ia"])
    client = client.with_options(max_retries=GROQ_RETRIES)
    results: list[list[str]] = [[] for _ in items]
    fresh: list[str] = []
    for start in range(0, len(items), TOPIC_BATCH_SIZE):
        batch = items[start:start + TOPIC_BATCH_SIZE]
        payload = [
            {"id": i, "title": it["title"], "description": clean_html(it.get("description") or "")[:200]}
            for i, it in enumerate(batch)
        ]
        known_names = _known_names(known, fresh)
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": (
                        "You tag news articles with what they are about, so that articles about the "
                        "same subject can be grouped together across days.\n"
                        "For each article return 1 to 4 topics, mixing these three kinds:\n"
                        f"- EVENT: the specific event or matter it reports (who + what), e.g. {hints['event']}.\n"
                        f"- ACTOR: the person or organisation the article is mainly about, e.g. {hints['actor']}.\n"
                        "- ISSUE: the public debate or problem it is part of, one the press follows and "
                        f"other articles on other days would share, e.g. {hints['issue']}. Never a research "
                        "field or a technique.\n"
                        "Give the EVENT, plus the ACTOR and the ISSUE when there is a clear one.\n\n"
                        "RULES:\n"
                        "- 2-6 words, English, Title Case, keep proper nouns. Write only the name, "
                        "never the kind ('Huawei', not 'Huawei Actor').\n"
                        f"- Too broad, never use: {hints['bad']}.\n"
                        "- If an article is about the same thing as one of KNOWN_TOPICS, reuse that name "
                        "EXACTLY instead of writing a variant.\n"
                        "- Only tag what the article is actually about, not what it merely mentions.\n\n"
                        "Return JSON: {\"articles\": [{\"id\": <id>, \"topics\": [\"...\"]}]}\n\n"
                        f"KNOWN_TOPICS = {json.dumps(known_names, ensure_ascii=False)}\n\n"
                        f"ARTICLES = {json.dumps(payload, ensure_ascii=False)}"
                    ),
                }],
                temperature=0.1,
                max_tokens=4000,
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            for row in data.get("articles", []):
                i = row.get("id")
                if isinstance(i, int) and 0 <= i < len(batch):
                    results[start + i] = _clean_topics(row.get("topics"))
        except Exception as e:
            logging.warning(f"Topic extraction failed for a batch of {len(batch)} ({domain}): {e}")
        for topics in results[start:start + len(batch)]:
            fresh.extend(t for t in topics if t not in known)
            known.update(topics)
        if start + TOPIC_BATCH_SIZE < len(items):
            await asyncio.sleep(batch_pause)
    tagged = sum(1 for t in results if t)
    logging.info(f"Topics [{domain}]: tagged {tagged}/{len(items)} articles")
    return results


def _published_day(a: dict) -> str:
    return (a.get("published") or "")[:10]


def _story_activity(story: dict) -> tuple[int, int]:
    """Sort key: articles in the last HOT_RECENT_DAYS days, then total size."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=HOT_RECENT_DAYS)).strftime("%Y-%m-%d")
    recent = sum(1 for a in story["articles"] if _published_day(a) >= cutoff)
    return recent, len(story["articles"])


def _merge_into(base: dict, other: dict) -> None:
    by_url = {a["url"]: a for a in base["articles"]}
    for a in other["articles"]:
        by_url.setdefault(a["url"], a)
    base["articles"] = sorted(by_url.values(), key=_published_day, reverse=True)


def build_story_candidates(window: list[dict], existing_keys: set[str] = frozenset()) -> list[dict]:
    """Group window articles (dicts with url, title, source, published, topics)
    into stories by shared topic. Returns stories most active first, each
    {topic_key, label, summary, articles (newest first)}.

    Topics that are too thin (< STORY_MIN_ARTICLES articles or a single source)
    or too broad (> STORY_MAX_SHARE of the window) don't make a story. Two
    topics covering nearly the same articles (variants Groq didn't unify) are
    merged, the surviving key being one that already exists in the DB when
    possible, so the story keeps its id.
    """
    tagged = [a for a in window if a.get("topics")]
    if not tagged:
        return []
    max_size = max(STORY_MAX_FLOOR, int(len(tagged) * STORY_MAX_SHARE))

    groups: dict[str, dict] = {}
    for a in tagged:
        for t in a["topics"]:
            key = _topic_key(t)
            if not key:
                continue
            g = groups.setdefault(key, {"topic_key": key, "names": Counter(), "articles": {}})
            g["names"][t] += 1
            g["articles"][a["url"]] = a

    stories = []
    for g in groups.values():
        arts = list(g["articles"].values())
        if not STORY_MIN_ARTICLES <= len(arts) <= max_size:
            continue
        if len({a.get("source") for a in arts}) < STORY_MIN_SOURCES:
            continue
        stories.append({
            "topic_key": g["topic_key"],
            "label": g["names"].most_common(1)[0][0],
            "summary": "",
            "articles": sorted(arts, key=_published_day, reverse=True),
        })

    # Biggest first, so a near-duplicate folds into the larger story
    stories.sort(key=lambda s: len(s["articles"]), reverse=True)
    kept: list[dict] = []
    for s in stories:
        urls = {a["url"] for a in s["articles"]}
        for k in kept:
            k_urls = {a["url"] for a in k["articles"]}
            if len(urls & k_urls) >= STORY_MERGE_OVERLAP * min(len(urls), len(k_urls)):
                if s["topic_key"] in existing_keys and k["topic_key"] not in existing_keys:
                    k["topic_key"], k["label"] = s["topic_key"], s["label"]
                _merge_into(k, s)
                break
        else:
            kept.append(s)

    kept.sort(key=_story_activity, reverse=True)
    return kept


async def review_stories(stories: list[dict], client, model: str, domain: str) -> list[dict]:
    """One Groq call over the most active stories: drop the incoherent ones
    (articles that only share a broad theme, e.g. "Energy Supply Disruptions"
    gathering a fuel tax cut and a solar carport), and write each kept story a
    short summary of its latest development, shown as "{label} — {summary}".
    The label stays the topic name. Returns the kept stories; if Groq fails,
    all of them, with no summary."""
    top = stories[:STORY_NAMING_LIMIT]
    if not top:
        return stories
    items = [
        {"id": i, "topic": s["label"], "titles": [a["title"] for a in s["articles"][:6]]}
        for i, s in enumerate(top)
    ]
    try:
        resp = await client.with_options(max_retries=GROQ_RETRIES).chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    "Each story below is a news topic with the titles of its most recent articles, "
                    "newest first.\n"
                    "For each story:\n"
                    "- `coherent`: true if the articles are about the same event, the same person or "
                    "organisation, or the same specific issue; false if they only share a broad theme "
                    "and a reader would see unrelated news.\n"
                    "- `summary`: 3-6 words, English, Title Case, naming the LATEST development "
                    "according to the titles. It is shown as \"<topic> — <summary>\", so it must not "
                    "repeat the topic.\n\n"
                    "Return JSON: {\"stories\": [{\"id\": <id>, \"coherent\": <bool>, \"summary\": \"...\"}]}\n\n"
                    f"STORIES = {json.dumps(items, ensure_ascii=False)}"
                ),
            }],
            temperature=0.1,
            max_tokens=4000,
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        logging.warning(f"Story review failed ({domain}): {e}")
        return stories
    incoherent: set[int] = set()
    for row in data.get("stories", []):
        i = row.get("id")
        if not isinstance(i, int) or not 0 <= i < len(top):
            continue
        if row.get("coherent") is False:
            incoherent.add(i)
        summary = " ".join(str(row.get("summary") or "").split())
        if summary and summary.lower() != top[i]["label"].lower():
            top[i]["summary"] = summary
    if incoherent:
        logging.info(f"Stories [{domain}]: dropped as incoherent: {[top[i]['label'] for i in sorted(incoherent)]}")
    return [s for i, s in enumerate(top) if i not in incoherent] + stories[STORY_NAMING_LIMIT:]


def _fetch_window_articles(client, domain: str, days: int = STORY_WINDOW_DAYS, extra_cols: str = "") -> list[dict]:
    """The domain's articles of the last `days` days, paginated past Supabase's
    1000-row cap. Returns [] (stories then built from this run's articles only)
    if the `topics` column isn't there yet."""
    if client is None:
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    rows: list[dict] = []
    try:
        while True:
            page = (
                client.table("articles")
                .select("url, title, source, published, topics" + extra_cols)
                .eq("domain", domain)
                .gte("published", cutoff)
                .order("url")
                .range(len(rows), len(rows) + 999)
                .execute()
            ).data or []
            if not page:
                return rows
            rows.extend(page)
    except Exception as e:
        logging.warning(f"Window fetch failed ({domain}): {e} — stories built from this run only")
        return []


def _fetch_existing_stories(client, domain: str) -> dict[str, dict]:
    """topic_key -> {id, label, summary, status} for this domain's topic-keyed stories."""
    if client is None:
        return {}
    try:
        rows = (
            client.table("stories")
            .select("id, topic_key, label, summary, status")
            .eq("domain", domain)
            .not_.is_("topic_key", "null")
            .execute()
        ).data or []
        return {r["topic_key"]: r for r in rows}
    except Exception as e:
        logging.warning(f"Existing stories fetch failed ({domain}): {e}")
        return {}


def _save_stories(client, domain: str, stories: list[dict], existing: dict[str, dict]) -> dict[str, int]:
    """Upsert this run's stories, close every other open story of the domain
    (including pre-topics stories), and return topic_key -> story id.

    A story's label is fixed once written (stable anchor in the dashboard and
    the newsletter picker); only its summary follows the latest development."""
    if client is None or not stories:
        return {}
    rows = []
    for s in stories:
        prev = existing.get(s["topic_key"], {})
        s["label"] = prev.get("label") or s["label"]
        s["summary"] = s["summary"] or prev.get("summary") or ""
        days = [_published_day(a) for a in s["articles"]]
        rows.append({
            "domain": domain,
            "topic_key": s["topic_key"],
            "label": s["label"],
            "summary": s["summary"],
            "first_seen": min(days),
            "last_seen": max(days),
            "article_count": len(s["articles"]),
            "status": "open",
            "recent_titles": "|".join(a["title"] for a in s["articles"][:5]),
            "article_urls": [a["url"] for a in s["articles"]],
        })
    try:
        saved = client.table("stories").upsert(rows, on_conflict="domain,topic_key").execute().data or []
    except Exception as e:
        logging.error(f"Stories upsert failed ({domain}): {e}")
        return {}
    ids = {r["topic_key"]: r["id"] for r in saved}

    try:
        open_rows = (
            client.table("stories").select("id").eq("domain", domain).eq("status", "open").execute()
        ).data or []
        stale = [r["id"] for r in open_rows if r["id"] not in set(ids.values())]
        for i in range(0, len(stale), 100):
            client.table("stories").update({"status": "closed"}).in_("id", stale[i:i + 100]).execute()
    except Exception as e:
        logging.warning(f"Closing stale stories failed ({domain}): {e}")
    logging.info(f"Stories [{domain}]: {len(ids)} saved")
    return ids


def _apply_story_flags(articles: list["Article"], stories: list[dict], ids: dict[str, int]) -> None:
    """Set the hot_* fields and story_id of this run's articles from the stories.

    hot_topic: one of the article's stories is active right now (>= HOT_MIN_RECENT
    articles in the last HOT_RECENT_DAYS days). hot_reason / story_id point at
    the article's most active story; mention_count is that story's recent
    article count. These per-article fields feed the classifier and Telegram;
    the dashboard reads full story membership from `stories.article_urls`."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    best: dict[str, tuple[tuple[int, int], dict]] = {}
    for s in stories:
        activity = _story_activity(s)
        for a in s["articles"]:
            if a["url"] not in best or activity > best[a["url"]][0]:
                best[a["url"]] = (activity, s)
    for art in articles:
        hit = best.get(art.url)
        if hit is None:
            art.hot_topic, art.hot_reason, art.mention_count, art.supa_hot, art.story_id = False, "", 0, False, None
            continue
        (recent, _), s = hit
        art.story_id = ids.get(s["topic_key"])
        art.hot_topic = recent >= HOT_MIN_RECENT
        art.hot_reason = s["label"] if art.hot_topic else ""
        art.mention_count = recent if art.hot_topic else 0
        today_count = sum(1 for a in s["articles"] if _published_day(a) == today)
        art.supa_hot = art.hot_topic and today_count >= SUPA_HOT_MIN_TODAY


def _save_topics(client, rows: list[dict]) -> None:
    if client is None:
        return
    for r in rows:
        try:
            client.table("articles").update({"topics": r["topics"]}).eq("url", r["url"]).execute()
        except Exception as e:
            logging.warning(f"Topics update failed for {r['url']}: {e}")


async def refresh_stories(domain: str, new_articles: list["Article"], client, groq_client, model: str) -> None:
    """Tag this run's articles with topics, rebuild the domain's stories over the
    window, save them, and flag the run's articles hot from them."""
    rows = sorted(_fetch_window_articles(client, domain), key=_published_day, reverse=True)
    window = {r["url"]: r for r in rows}
    known = Counter(t for r in window.values() for t in (r.get("topics") or []))

    # An article fetched again keeps the topics an earlier run gave it
    todo = []
    for a in new_articles:
        prev = (window.get(a.url) or {}).get("topics")
        if prev:
            a.topics = prev
        else:
            todo.append(a)
    if todo:
        extracted = await extract_topics(
            [{"title": a.title, "description": a.description} for a in todo], known, groq_client, model, domain
        )
        for a, topics in zip(todo, extracted):
            a.topics = topics
    for a in new_articles:
        window[a.url] = {"url": a.url, "title": a.title, "source": a.source, "published": a.published, "topics": a.topics}


    existing = _fetch_existing_stories(client, domain)
    stories = build_story_candidates(list(window.values()), set(existing))
    stories = await review_stories(stories, groq_client, model, domain)
    ids = _save_stories(client, domain, stories, existing)
    _apply_story_flags(new_articles, stories, ids)


BACKFILL_MAX_ARTICLES = 500  # per run, ~80k Groq tokens: leaves the daily collection its share of the 200k/day quota


async def backfill_topics(model: str | None = None) -> None:
    """Resumable: tag up to BACKFILL_MAX_ARTICLES window articles that have no
    topics yet, newest first, then rebuild the stories of the domains touched.
    Stops early when a whole chunk comes back untagged (Groq quota reached).
    Each run resumes where the previous one stopped; once everything is
    tagged, a run does nothing."""
    client = _get_supabase_client()
    if client is None:
        sys.exit("SUPABASE_URL and SUPABASE_KEY must be set.")
    groq_client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    model = model or (os.environ.get("GROQ_MODEL") or "openai/gpt-oss-120b").strip("'\"").strip()

    budget = BACKFILL_MAX_ARTICLES
    for domain in DOMAIN_META:
        if budget <= 0:
            break
        window = sorted(
            _fetch_window_articles(client, domain, extra_cols=", description"), key=_published_day, reverse=True
        )
        todo = [r for r in window if not r.get("topics")]
        logging.info(f"Backfill [{domain}]: {len(todo)}/{len(window)} articles without topics")
        if not todo:
            continue
        known = Counter(t for r in window for t in (r.get("topics") or []))
        todo = todo[:budget]
        budget -= len(todo)
        chunk = TOPIC_BATCH_SIZE * 5
        quota_hit = False
        for start in range(0, len(todo), chunk):
            rows = todo[start:start + chunk]
            extracted = await extract_topics(rows, known, groq_client, model, domain)
            for row, topics in zip(rows, extracted):
                row["topics"] = topics
            _save_topics(client, [r for r in rows if r["topics"]])
            if not any(extracted):
                logging.error(f"Backfill [{domain}]: a whole chunk failed — stopping (Groq quota?). Re-run later.")
                quota_hit = True
                break
        await refresh_stories(domain, [], client, groq_client, model)
        if quota_hit:
            break


# ---------------------------------------------------------------------------
# 1. Load sources
# ---------------------------------------------------------------------------

def load_sources(path: str = "sources.json") -> list[dict]:
    """Load enabled sources from the JSON config file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sources = [s for s in data["sources"] if s.get("enabled", True)]
    logging.info(f"Loaded {len(sources)} enabled sources")
    return sources

# ---------------------------------------------------------------------------
# 2. Fetch articles (async)
# ---------------------------------------------------------------------------

async def fetch_rss(session: aiohttp.ClientSession, source: dict) -> list[Article]:
    """Fetch and parse a standard RSS feed, keeping only last-24h entries."""
    try:
        async with session.get(source["url"], timeout=aiohttp.ClientTimeout(total=30)) as resp:
            text = await resp.text()
    except Exception as e:
        logging.error(f"[{source['name']}] HTTP error: {e}")
        return []

    feed = feedparser.parse(text)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    articles = []

    for entry in feed.entries[:15]:
        pub_date = parse_feed_date(entry)
        if pub_date and pub_date < cutoff:
            continue

        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        if not title or not link:
            continue

        articles.append(Article(
            title=title,
            url=link,
            source=source["name"],
            country=source["country"],
            published=pub_date.isoformat() if pub_date else datetime.now(timezone.utc).isoformat(),
            published_is_estimated=pub_date is None,
            description=clean_html(entry.get("summary", ""))[:200],
            domain=source.get("domain", "ia"),
        ))

    logging.info(f"[{source['name']}] {len(articles)} articles")
    return articles


async def fetch_reddit(session: aiohttp.ClientSession, source: dict) -> list[Article]:
    """Fetch Reddit RSS with a proper User-Agent."""
    headers = {"User-Agent": "AI-Radar/1.0 (news aggregator bot)"}
    try:
        async with session.get(source["url"], headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            text = await resp.text()
    except Exception as e:
        logging.error(f"[{source['name']}] HTTP error: {e}")
        return []

    feed = feedparser.parse(text)
    articles = []

    for entry in feed.entries[:15]:
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        if not title or not link:
            continue

        pub_date = parse_feed_date(entry)
        articles.append(Article(
            title=title,
            url=link,
            source=source["name"],
            country=source["country"],
            published=pub_date.isoformat() if pub_date else datetime.now(timezone.utc).isoformat(),
            published_is_estimated=pub_date is None,
            description=clean_html(entry.get("summary", ""))[:200],
            domain=source.get("domain", "ia"),
        ))

    logging.info(f"[{source['name']}] {len(articles)} articles")
    return articles


async def fetch_hackernews(session: aiohttp.ClientSession, source: dict) -> list[Article]:
    """Query the HN Algolia API for AI-related stories with minimum points."""
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(hours=24)).timestamp())
    min_points = source.get("min_points", 30)
    seen_ids: set[str] = set()
    articles = []

    for keyword in source.get("keywords", ["AI"]):
        params = {
            "query": keyword,
            "tags": "story",
            "numericFilters": f"points>{min_points},created_at_i>{cutoff_ts}",
            "hitsPerPage": 20,
        }
        try:
            async with session.get(source["url"], params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                data = await resp.json()
        except Exception as e:
            logging.error(f"[HN/{keyword}] API error: {e}")
            continue

        for hit in data.get("hits", []):
            oid = hit.get("objectID", "")
            if oid in seen_ids:
                continue
            seen_ids.add(oid)

            title = hit.get("title", "").strip()
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={oid}"
            if not title:
                continue

            created_at_i = hit.get("created_at_i")
            pub_date = (
                datetime.fromtimestamp(created_at_i, tz=timezone.utc)
                if created_at_i is not None else datetime.now(timezone.utc)
            )
            articles.append(Article(
                title=title,
                url=url,
                source=source["name"],
                country=source["country"],
                published=pub_date.isoformat(),
                published_is_estimated=created_at_i is None,
                description=(hit.get("story_text") or "")[:200],
                domain=source.get("domain", "ia"),
            ))

    logging.info(f"[{source['name']}] {len(articles)} articles")
    return articles


GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Relevance filter for GDELT-sourced articles. GDELT's DOC API matches query
# keywords against the full article body, not just the title, so a query like
# "trade war" can surface an article whose actual topic is unrelated (e.g. a
# business piece that mentions tariffs in passing). Re-checking the title
# against a curated keyword list — same "1 strong OR 2+ total" logic as the
# AI_STRONG/WEAK_KEYWORDS filter in fetch_all() — catches these before Groq.
GDELT_STRONG_KEYWORDS = {
    # Conflits / Guerres
    "war", "conflict", "offensive", "airstrike", "air strike", "ceasefire",
    "cease-fire", "invasion", "troops", "missile", "civil war", "insurgent",
    "rebel", "militant", "combat", "shelling", "bombing", "gunmen",
    # Soulevements / Manifestations
    "protest", "uprising", "demonstrators", "general strike", "riot",
    "unrest", "crackdown",
    # Catastrophes naturelles
    "earthquake", "flood", "wildfire", "hurricane", "drought", "tsunami",
    "volcano", "cyclone", "typhoon", "landslide", "quake",
    # Coups d'Etat
    "coup", "ousted", "regime change", "junta", "overthrown", "toppled",
    # Diplomatie
    "diplomatic summit", "peace talks", "bilateral meeting",
    "un security council", "peace deal", "ceasefire agreement", "envoy",
    "treaty", "summit",
    # Sanctions / Guerre economique
    "sanctions", "embargo", "asset freeze", "trade war", "tariffs",
    "export ban", "export controls",
}
# Weak: generic terms that need a companion keyword to count as a match
GDELT_WEAK_KEYWORDS = {
    "government", "president", "minister", "election", "crisis",
    "border", "opposition", "army", "forces", "military", "police",
}


def _is_relevant_gdelt_article(title: str) -> bool:
    """Check a GDELT hit's title against curated political-event keywords."""
    text = title.lower()
    strong_hits = sum(1 for kw in GDELT_STRONG_KEYWORDS if kw in text)
    weak_hits = sum(1 for kw in GDELT_WEAK_KEYWORDS if kw in text)
    return strong_hits >= 1 or (strong_hits + weak_hits) >= 2


async def fetch_gdelt_all(session: aiohttp.ClientSession, gdelt_sources: list[dict]) -> list[Article]:
    """Query the GDELT DOC 2.0 API for each configured theme, sequentially.

    GDELT's documented limit is ~1 request/5s, but in practice it intermittently
    returns an empty body well under that rate too — so sources of this type are
    fetched one after another (with a delay), and each gets one retry on failure,
    rather than being dispatched as parallel tasks like the other source types.
    """
    articles = []

    for i, source in enumerate(gdelt_sources):
        if i > 0:
            await asyncio.sleep(10)

        params = {
            "query": f"{source['query']} sourcelang:english",
            "mode": "artlist",
            "maxrecords": 15,
            "timespan": "24h",
            "format": "json",
            "sort": "DateDesc",
        }
        data = None
        for attempt in range(2):
            if attempt > 0:
                await asyncio.sleep(15)
            try:
                async with session.get(
                    GDELT_DOC_API_URL, params=params,
                    headers={"User-Agent": "AI-Radar/1.0 (news aggregator bot)"},
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    data = await resp.json(content_type=None)
                break
            except Exception as e:
                logging.warning(f"[{source['name']}] GDELT API error (attempt {attempt + 1}/2): {e}")
                data = None

        if data is None:
            logging.error(f"[{source['name']}] GDELT fetch failed, skipping")
            continue

        hits = data.get("articles", [])
        kept = 0
        for hit in hits:
            title = (hit.get("title") or "").strip()
            url = (hit.get("url") or "").strip()
            if not title or not url:
                continue

            if not _is_relevant_gdelt_article(title):
                continue

            try:
                pub_date = datetime.strptime(hit["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                estimated = False
            except (KeyError, ValueError):
                pub_date = datetime.now(timezone.utc)
                estimated = True

            articles.append(Article(
                title=title,
                url=url,
                source=source["name"],
                country=source.get("country", "🌍"),
                published=pub_date.isoformat(),
                published_is_estimated=estimated,
                domain=source.get("domain", "ia"),
            ))
            kept += 1

        logging.info(f"[{source['name']}] {kept}/{len(hits)} articles kept after relevance filter")

    return articles


async def fetch_usgs(session: aiohttp.ClientSession, source: dict) -> list[Article]:
    """Fetch the USGS 'significant earthquakes, past day' GeoJSON feed."""
    try:
        async with session.get(source["url"], timeout=aiohttp.ClientTimeout(total=15)) as resp:
            data = await resp.json(content_type=None)
    except Exception as e:
        logging.error(f"[{source['name']}] HTTP error: {e}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    articles = []

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        title = (props.get("title") or "").strip()
        url = (props.get("url") or "").strip()
        if not title or not url:
            continue

        time_ms = props.get("time")
        pub_date = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc) if time_ms else datetime.now(timezone.utc)
        if pub_date < cutoff:
            continue

        mag = props.get("mag")
        articles.append(Article(
            title=title,
            url=url,
            source=source["name"],
            country=source.get("country", "🌍"),
            published=pub_date.isoformat(),
            published_is_estimated=time_ms is None,
            description=f"Magnitude {mag}" if mag is not None else "",
            domain=source.get("domain", "ia"),
        ))

    logging.info(f"[{source['name']}] {len(articles)} articles")
    return articles


async def fetch_all(sources: list[dict]) -> list[Article]:
    """Fetch all sources in parallel, deduplicate by URL."""
    gdelt_sources = [src for src in sources if src["type"] == "gdelt"]

    async with aiohttp.ClientSession() as session:
        tasks = []
        for src in sources:
            if src["type"] == "rss":
                tasks.append(fetch_rss(session, src))
            elif src["type"] == "reddit":
                tasks.append(fetch_reddit(session, src))
            elif src["type"] == "hn_api":
                tasks.append(fetch_hackernews(session, src))
            elif src["type"] == "usgs":
                tasks.append(fetch_usgs(session, src))

        if gdelt_sources:
            tasks.append(fetch_gdelt_all(session, gdelt_sources))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    articles = []
    for result in results:
        if isinstance(result, Exception):
            logging.error(f"Fetch task failed: {result}")
        else:
            articles.extend(result)

    # Fetch Bluesky sources
    bluesky_dicts = await fetch_all_bluesky(sources)
    for d in bluesky_dicts:
        articles.append(Article(
            title=d["title"],
            url=d["url"],
            source=d["source"],
            country=d["country"],
            published=d["published"],
            published_is_estimated=d.get("published_is_estimated", False),
            description=d.get("description", ""),
            domain=d.get("domain", "ia"),
        ))

    # Deduplicate by URL
    seen: set[str] = set()
    unique = []
    for a in articles:
        if a.url not in seen:
            seen.add(a.url)
            unique.append(a)

    # Filter: keep only AI-related articles.
    # Strong keywords are AI-specific enough to qualify an article on their own.
    # Weak keywords are generic tech terms that require at least one companion match.
    AI_STRONG_KEYWORDS = {
        # Core concepts
        "artificial intelligence", "machine learning", "deep learning",
        "llm", "large language model", "neural network", "chatgpt", "gpt",
        "generative ai", "chatbot", "agi", "artificial general intelligence",
        "intelligence artificielle", "apprentissage automatique",
        # Architectures & techniques
        "transformer", "mixture of experts", "moe", "diffusion model",
        "multimodal", "vision language model", "vlm", "reasoning model",
        "context window", "sparse model", "embedding", "vector database",
        "retrieval augmented generation", "rag", "fine-tuning", "rlhf",
        "test-time compute", "inference scaling",
        # Prompt & context engineering
        "prompt engineering", "context engineering", "system prompt",
        "few-shot", "zero-shot", "chain of thought", "prompt optimization",
        "prompt injection", "jailbreak",
        # AI-assisted dev
        "vibe coding", "ai coding", "code generation", "github copilot",
        "devin", "cursor ai",
        # Model optimization
        "quantization", "knowledge distillation", "lora", "qlora", "peft",
        "model compression", "speculative decoding", "flash attention",
        "inference optimization", "efficient inference",
        # AI hardware
        "tpu", "cerebras", "graphcore", "tenstorrent", "h100", "h200", "b200",
        "blackwell", "hopper",
        # Established AI companies
        "openai", "anthropic", "deepmind", "meta ai", "hugging face",
        "stability ai", "runway", "cohere", "mistral", "xai",
        # Emerging players
        "deepseek", "qwen", "perplexity", "together ai",
        # Agents & autonomy
        "ai agent", "agentic", "model context protocol", "autonomous agent",
        # Safety, ethics & regulation
        "ai safety", "alignment", "hallucination", "ai regulation", "eu ai act",
        "responsible ai", "interpretability", "explainability", "deepfake",
        "ai governance",
        # Performance
        "open source model", "open weights", "edge ai", "on-device ai", "evals",
    }
    # Weak: generic terms that need a companion keyword to be AI-relevant
    AI_WEAK_KEYWORDS = {
        "ai", "grok", "gemini", "copilot", "cursor",
        "nvidia", "amd", "intel", "qualcomm", "tsmc", "gaudi", "arm chip",
        "chip", "semiconductor", "data center", "robot", "automation",
        "benchmark", "bias", "pruning", "distillation",
    }
    # This relevance filter only makes sense for the "ia" domain (its sources are
    # broad tech/AI feeds that need narrowing). Other domains' sources are already
    # on-topic by construction (e.g. an oil-price feed doesn't need an "is this
    # about oil" check) — they pass through untouched.
    filtered = []
    for a in unique:
        if a.domain != "ia":
            filtered.append(a)
            continue
        text = (a.title + " " + a.description).lower()
        strong_hits = sum(1 for kw in AI_STRONG_KEYWORDS if kw in text)
        weak_hits   = sum(1 for kw in AI_WEAK_KEYWORDS   if kw in text)
        # Pass if: 1 strong keyword OR 2+ keyword matches in total
        if strong_hits >= 1 or (strong_hits + weak_hits) >= 2:
            filtered.append(a)

    logging.info(f"{len(filtered)}/{len(unique)} articles kept after AI filter")

    return filtered

# ---------------------------------------------------------------------------
# 3. Classification with Groq
# ---------------------------------------------------------------------------

# Category taxonomy, classification notes and fallback category, keyed by domain.
# "ia" preserves the exact wording used before the multi-domain refactor so
# classification behavior for that domain is unchanged.
DOMAIN_TAXONOMY: dict[str, dict] = {
    "ia": {
        "categories": [
            "Innovation / Tech", "Politique / Regulation", "Business / Industrie",
            "Societe / Ethique", "Recherche Academique", "Drama / Controverses",
            "Energie / Environnement", "Semiconducteurs / Hardware",
        ],
        "notes": (
            '  - "Politique / Regulation" : geopolitique, regulation internationale, diplomatie tech, '
            "export controls chips, CHIPS Act, guerre commerciale semi-conducteurs.\n"
            '  - "Energie / Environnement" : consommation energetique de l\'IA, data centers et reseau '
            "electrique, transition energetique, energies renouvelables, nucleaire, rapports IEA/AIE, "
            "prix de l'energie.\n"
            '  - "Semiconducteurs / Hardware" : industrie des semi-conducteurs (hors geopolitique), '
            "GPU/NPU/puces IA, fonderies (TSMC, Samsung, Intel Foundry), equipementiers (ASML), "
            "nouveaux procedes de fabrication, marche des chips."
        ),
        "default_category": "Innovation / Tech",
    },
    "politique_evenements": {
        "categories": [
            "Conflits / Guerres", "Soulevements / Manifestations", "Catastrophes naturelles",
            "Changements de regime / Coups d'Etat", "Diplomatie / Sommets internationaux",
            "Sanctions / Guerre economique",
        ],
        "notes": (
            '  - "Conflits / Guerres" : conflits armes, guerres, offensives militaires, frappes '
            "aeriennes, cessez-le-feu, guerre civile - evenements militaires actifs uniquement, pas "
            "les tensions ou negociations sans action militaire (preferer Diplomatie / Sommets "
            "internationaux dans ce cas).\n"
            '  - "Soulevements / Manifestations" : manifestations, greves generales, emeutes, '
            "mouvements de contestation populaire - hors coups d'Etat organises par l'armee ou le "
            "pouvoir en place (preferer Changements de regime / Coups d'Etat dans ce cas).\n"
            '  - "Catastrophes naturelles" : seismes, inondations, incendies, ouragans, secheresses '
            "- evenements climatiques/geologiques uniquement, pas leurs consequences economiques "
            "(preferer Sanctions / Guerre economique si l'angle est economique).\n"
            '  - "Changements de regime / Coups d\'Etat" : coups d\'Etat, chutes de gouvernement, '
            "transitions de pouvoir non electorales.\n"
            '  - "Diplomatie / Sommets internationaux" : sommets, negociations, rencontres '
            "bilaterales, resolutions ONU, traites - tensions et discussions diplomatiques sans "
            "action militaire ni sanction economique.\n"
            '  - "Sanctions / Guerre economique" : sanctions internationales, embargos, guerre '
            "commerciale, gel d'actifs."
        ),
        "default_category": "Diplomatie / Sommets internationaux",
    },
    "matieres_premieres": {
        "categories": [
            "Petrole / Gaz", "Metaux / Mines", "Agriculture / Denrees",
            "Energie / Renouvelable", "Terres rares / Chaine d'approvisionnement",
        ],
        "notes": (
            '  - "Terres rares / Chaine d\'approvisionnement" : approvisionnement en terres rares et '
            "composants critiques pour l'industrie tech/IA (hors regulation, qui va dans le domaine IA "
            "si l'angle est reglementaire).\n"
            '  - "Energie / Renouvelable" : production et transition energetique (hors consommation '
            "energetique des data centers IA, qui reste dans le domaine IA)."
        ),
        "default_category": "Petrole / Gaz",
    },
    "finance": {
        "categories": [
            "Marches actions", "Taux / Banques centrales", "Crypto-actifs",
            "Fusions-acquisitions / IPO", "Dette / Obligations", "Nouveaux actifs IA",
        ],
        "notes": (
            '  - "Nouveaux actifs IA" : hors crypto-monnaies classiques - economie des tokens IA '
            "(trackers de prix de tokens, routers d'optimisation de tokens, marketplaces de "
            "compute/inference).\n"
            '  - "Crypto-actifs" : crypto-monnaies, blockchain, DeFi - hors sujets specifiquement '
            "lies aux tokens IA (voir Nouveaux actifs IA)."
        ),
        "default_category": "Marches actions",
    },
    "services": {
        "categories": [
            "Emploi / Marche du travail", "Consommation / Retail", "Indicateurs macro",
            "Immobilier", "Adoption IA (particuliers et entreprises)",
        ],
        "notes": (
            '  - "Adoption IA (particuliers et entreprises)" : taux d\'usage de l\'IA, integration en '
            "entreprise, outils grand public - angle adoption/usage, pas innovation technique "
            "(qui reste dans le domaine IA)."
        ),
        "default_category": "Indicateurs macro",
    },
}

_GROQ_PROMPT_COMMON_TAIL = """

- "sentiment": une valeur parmi ["Positif", "Negatif", "Neutre"]

- "country": le pays principalement concerne(e) par l'evenement (ex: "USA", "Chine", "France", "Japon"). N'utilise pas de region contenant plus d'un pays. Le pays que tu donneras fait reference a l'endroit ou se passe l'action, ou l'origine de l'entreprise concernee. Priorise l'endroit geographique ou se passe l'action. Si tu ne trouves rien de pertinent, utilise le label "Global". N'utilise jamais la nationalite du media qui relaie la news, car un media francais peut parler d'une news americaine par exemple."""

# Only requested for hot articles (full model) — saves tokens on non-hot majority
_GROQ_PROMPT_SUMMARY = """

- "summary": une phrase de synthese en francais (20-30 mots max) qui explique l'essentiel de l'article. Commence directement par le fait principal, sans tourner autour du pot."""

_GROQ_PROMPT_FOOTER = "\n\nNe renvoie AUCUN texte supplementaire. Uniquement l'objet JSON."


def _build_groq_prompt(domain: str, with_summary: bool) -> str:
    """Assemble the Groq classification system prompt for a given domain."""
    taxo = DOMAIN_TAXONOMY.get(domain, DOMAIN_TAXONOMY["ia"])
    categories_json = json.dumps(taxo["categories"], ensure_ascii=False)
    notes = f"\n  Notes de classification :\n{taxo['notes']}" if taxo.get("notes") else ""
    prompt = (
        "Tu es un classificateur d'actualites. Pour chaque article, renvoie UNIQUEMENT un objet "
        "JSON avec les cles suivantes :\n\n"
        f'- "category": une valeur parmi {categories_json}{notes}'
        + _GROQ_PROMPT_COMMON_TAIL
    )
    if with_summary:
        prompt += _GROQ_PROMPT_SUMMARY
    return prompt + _GROQ_PROMPT_FOOTER


VALID_SENTIMENTS = {"Positif", "Negatif", "Neutre"}


async def _classify_one(client: AsyncGroq, model_fast: str, model_full: str, article: Article) -> bool:
    """Classify a single article in-place, using its domain's taxonomy/prompt.
    Hot articles use model_full (70b): category + sentiment + country + summary.
    Non-hot articles use model_fast (8b): category + sentiment + country only.
    This cuts ~65% of 70b token usage on a typical day.
    Returns False if the Groq call failed and the article fell back to defaults."""
    taxo = DOMAIN_TAXONOMY.get(article.domain, DOMAIN_TAXONOMY["ia"])
    valid_categories = set(taxo["categories"])
    default_category = taxo["default_category"]

    user_msg = f"Titre: {article.title}\nSource: {article.source}"
    if article.description:
        user_msg += f"\nDescription: {article.description}"

    if article.hot_topic:
        model         = model_full
        system_prompt = _build_groq_prompt(article.domain, with_summary=True)
        max_tokens    = 500   # reasoning budget + category + sentiment + country + summary
    else:
        model         = model_fast
        system_prompt = _build_groq_prompt(article.domain, with_summary=False)
        max_tokens    = 300   # reasoning budget + category + sentiment + country

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
            reasoning_effort="low",  # gpt-oss models spend tokens "thinking" before the JSON;
                                     # low effort + prior tight budgets caused empty completions
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        cat    = result.get("category",  default_category)
        sent   = result.get("sentiment", "Neutre")
        article.category  = cat  if cat  in valid_categories else default_category
        # arXiv sources are always academic — override Groq's guess (ia domain only)
        if article.domain == "ia" and article.source.startswith("ArXiv"):
            article.category = "Recherche Academique"
        article.sentiment = sent if sent in VALID_SENTIMENTS  else "Neutre"
        article.country   = result.get("country", "Global") or "Global"
        if article.hot_topic:
            article.summary = (result.get("summary") or "").strip()
        return True
    except Exception as e:
        logging.warning(f"Groq error for '{article.title[:60]}': {e}")
        article.category  = default_category
        article.sentiment = "Neutre"
        article.country   = "Global"
        return False


async def classify_articles(articles: list[Article], batch_size: int = 15, batch_pause: float = 10.0) -> list[Article]:
    client     = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    model_full = (os.environ.get("GROQ_MODEL")      or "openai/gpt-oss-120b").strip("'\"").strip()
    model_fast = (os.environ.get("GROQ_MODEL_FAST") or "openai/gpt-oss-20b").strip("'\"").strip()

    hot_count  = sum(1 for a in articles if a.hot_topic)
    logging.info(f"Groq: {len(articles)} articles — {hot_count} hot ({model_full}) + {len(articles)-hot_count} non-hot ({model_fast})")

    failures = 0
    batches  = [articles[i:i + batch_size] for i in range(0, len(articles), batch_size)]
    for batch_idx, batch in enumerate(batches):
        logging.info(f"Groq: classifying batch {batch_idx + 1}/{len(batches)} ({len(batch)} articles)")
        results = await asyncio.gather(*[_classify_one(client, model_fast, model_full, a) for a in batch])
        failures += sum(1 for ok in results if not ok)
        if batch_idx < len(batches) - 1:
            logging.info(f"Groq: sleeping {batch_pause}s before next batch")
            await asyncio.sleep(batch_pause)

    # A silent 100%-fallback run (e.g. a stale/decommissioned GROQ_MODEL env var shadowing
    # the code default) previously went unnoticed for weeks — every article quietly got
    # category=default/sentiment=Neutre/country=Global, emptying the dashboard's globe and
    # category radar. Surface a hard-to-miss signal instead of one warning per article.
    if articles and failures / len(articles) > 0.2:
        msg = (
            f"Groq classification failed for {failures}/{len(articles)} articles "
            f"({failures / len(articles):.0%}) — check GROQ_API_KEY and the GROQ_MODEL/"
            f"GROQ_MODEL_FAST values (repo Actions variables override the code default "
            f"even when set to a decommissioned model)."
        )
        logging.error(msg)
        print(f"::error::{msg}")

    return articles

# ---------------------------------------------------------------------------
# 4. Supabase
# ---------------------------------------------------------------------------

def _get_supabase_client():
    """Create a Supabase client from env vars, or None if not configured."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def save_to_supabase(articles: list[Article], client=None) -> None:
    """Upsert articles into Supabase, ignoring duplicates by URL."""
    if client is None:
        client = _get_supabase_client()
    if client is None:
        logging.warning("SUPABASE_URL or SUPABASE_KEY not set — skipping Supabase save")
        return

    rows = [
        {
            "title": a.title,
            "url": a.url,
            "source": a.source,
            "country": a.country,
            "published": a.published,
            "description": a.description,
            "domain": a.domain,
            "category": a.category,
            "sentiment": a.sentiment,
            "hot_topic": a.hot_topic,
            "hot_source": a.hot_source,
            "hot_reason": a.hot_reason,
            "summary": a.summary,
            "mention_count": a.mention_count,
            "supa_hot": a.supa_hot,
            "story_id": a.story_id,
            "published_is_estimated": a.published_is_estimated,
            "topics": a.topics,
        }
        for a in articles
    ]

    # Columns that may be absent if migrations haven't been applied yet.
    # Migration required for mention_count / supa_hot:
    #   ALTER TABLE articles ADD COLUMN IF NOT EXISTS mention_count INTEGER DEFAULT 0;
    #   ALTER TABLE articles ADD COLUMN IF NOT EXISTS supa_hot BOOLEAN DEFAULT FALSE;
    # Migration required for domain (multi-domain radars):
    #   ALTER TABLE articles ADD COLUMN IF NOT EXISTS domain TEXT DEFAULT 'ia';
    # Migration required for cross-day story tracking (see CLAUDE.md):
    #   CREATE TABLE IF NOT EXISTS stories (...); ALTER TABLE articles ADD COLUMN IF NOT EXISTS story_id BIGINT REFERENCES stories(id);
    # Migration required for semantic stories (see CLAUDE.md):
    #   ALTER TABLE articles ADD COLUMN IF NOT EXISTS topics TEXT[] DEFAULT '{}';
    # Migration required for the "Publié" column fallback flag (see CLAUDE.md):
    #   ALTER TABLE articles ADD COLUMN IF NOT EXISTS published_is_estimated BOOLEAN DEFAULT FALSE;
    _OPTIONAL_COLS = (
        "hot_source", "hot_reason", "summary", "mention_count", "supa_hot", "domain",
        "story_id", "published_is_estimated", "topics",
    )

    try:
        client.table("articles").upsert(rows, on_conflict="url").execute()
        logging.info(f"Supabase: upserted {len(rows)} articles")
    except Exception as e:
        missing = [c for c in _OPTIONAL_COLS if c in str(e)]
        if missing:
            logging.warning(f"Columns missing ({missing}) — run migration. Retrying without them.")
            for row in rows:
                for col in missing:
                    row.pop(col, None)
            try:
                client.table("articles").upsert(rows, on_conflict="url").execute()
                logging.info(f"Supabase: upserted {len(rows)} articles (partial columns)")
            except Exception as e2:
                logging.error(f"Supabase upsert error: {e2}")
        else:
            logging.error(f"Supabase upsert error: {e}")


# ---------------------------------------------------------------------------
# 5. Telegram
# ---------------------------------------------------------------------------

SENTIMENT_EMOJI = {"Positif": "🟢", "Negatif": "🔴", "Neutre": "⚪"}

# Digest title + emoji per domain (used as the Telegram message header).
DOMAIN_META: dict[str, dict[str, str]] = {
    "ia":                   {"label": "Radar IA",                              "emoji": "🤖"},
    "politique_evenements": {"label": "Radar Politique / Evenements Majeurs",   "emoji": "🌍"},
    "matieres_premieres":   {"label": "Radar Matieres Premieres",              "emoji": "🛢️"},
    "finance":              {"label": "Radar Finance / Marches",               "emoji": "📈"},
    "services":             {"label": "Radar Services / Economie",             "emoji": "💼"},
}

# Category emoji lookup, scoped per domain (categories are only unique within a domain).
DOMAIN_CATEGORY_EMOJI: dict[str, dict[str, str]] = {
    "ia": {
        "Innovation / Tech":          "🚀",
        "Politique / Regulation":     "⚖️",
        "Business / Industrie":       "💼",
        "Societe / Ethique":          "🤝",
        "Recherche Academique":       "🎓",
        "Drama / Controverses":       "💥",
        "Energie / Environnement":    "⚡",
        "Semiconducteurs / Hardware": "🔬",
    },
    "politique_evenements": {
        "Conflits / Guerres":                    "⚔️",
        "Soulevements / Manifestations":          "✊",
        "Catastrophes naturelles":                "🌪️",
        "Changements de regime / Coups d'Etat":   "🏛️",
        "Diplomatie / Sommets internationaux":    "🤝",
        "Sanctions / Guerre economique":          "💣",
    },
    "matieres_premieres": {
        "Petrole / Gaz":                             "🛢️",
        "Metaux / Mines":                             "⛏️",
        "Agriculture / Denrees":                      "🌾",
        "Energie / Renouvelable":                     "⚡",
        "Terres rares / Chaine d'approvisionnement":  "💎",
    },
    "finance": {
        "Marches actions":            "📈",
        "Taux / Banques centrales":   "🏦",
        "Crypto-actifs":              "₿",
        "Fusions-acquisitions / IPO": "🤝",
        "Dette / Obligations":        "📉",
        "Nouveaux actifs IA":         "🧮",
    },
    "services": {
        "Emploi / Marche du travail":                 "👷",
        "Consommation / Retail":                      "🛒",
        "Indicateurs macro":                          "📊",
        "Immobilier":                                 "🏠",
        "Adoption IA (particuliers et entreprises)":  "🤖",
    },
}


def _post_telegram(token: str, chat_id: str, text: str) -> None:
    """Send a single Telegram message."""
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=10,
    )
    if not resp.ok:
        logging.error(f"Telegram error: {resp.status_code} {resp.text}")


def _send_domain_digest(token: str, chat_id: str, domain: str, articles: list[Article], dashboard_url: str) -> None:
    """Send one recap + hot-articles digest for a single domain's articles."""
    import time
    today = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    meta = DOMAIN_META.get(domain, DOMAIN_META["ia"])
    cat_emoji_map = DOMAIN_CATEGORY_EMOJI.get(domain, {})

    # ── Header recap ──────────────────────────────────────────────────────────
    stats = compute_stats(articles)
    hot_articles = [a for a in articles if a.hot_topic]
    header_lines = [
        f"{meta['emoji']} <b>{meta['label']} — {today}</b>",
        f"📰 {len(articles)} articles collectés · 🔥 {len(hot_articles)} hot topics",
        "",
    ]
    for cat, emoji in cat_emoji_map.items():
        count = stats.get(cat, 0)
        if count:
            header_lines.append(f"{emoji} {cat} : {count}")
    if dashboard_url:
        header_lines.append(f'\n📊 <a href="{dashboard_url}">Voir le Dashboard</a>')
    _post_telegram(token, chat_id, "\n".join(header_lines))

    if not hot_articles:
        return

    # ── Hot articles only — supa_hot first, then by date desc ─────────────────
    hot_articles.sort(key=lambda a: (not a.supa_hot, a.published), reverse=False)

    batch: list[str] = []
    batch_chars = 0

    for article in hot_articles:
        sent_emoji = SENTIMENT_EMOJI.get(article.sentiment, "⚪")
        cat_emoji  = cat_emoji_map.get(article.category, "📌")
        if article.supa_hot:
            badge = f"🌋 <b>SUPA HOT · {article.mention_count} sources</b>\n"
        else:
            badge = "🔥 "
        # Prefer the Groq-generated summary; fall back to raw description snippet
        blurb = (article.summary or article.description).strip()
        if blurb and not blurb.endswith((".", "!", "?")):
            blurb += "…"

        entry_lines = [
            f"{badge}{sent_emoji} <b>{article.title}</b>",
            f"{cat_emoji} {article.category} | {article.country} {article.source}",
        ]
        if blurb:
            entry_lines.append(f"<i>{blurb}</i>")
        entry_lines.append(f'<a href="{article.url}">🔗 Lire l\'article</a>')
        entry = "\n".join(entry_lines)

        if batch and batch_chars + len(entry) + 2 > 4000:
            _post_telegram(token, chat_id, "\n\n".join(batch))
            batch = []
            batch_chars = 0
            time.sleep(0.5)

        batch.append(entry)
        batch_chars += len(entry) + 2

    if batch:
        _post_telegram(token, chat_id, "\n\n".join(batch))

    logging.info(f"Telegram [{domain}]: sent recap + {len(hot_articles)} hot articles")


def send_telegram(articles: list[Article], dashboard_url: str = "") -> None:
    """Group articles by domain and send one digest message per domain."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    if not articles:
        _post_telegram(token, chat_id, "🤖 Radar IA : 0 nouveaux articles aujourd'hui.")
        return

    by_domain: dict[str, list[Article]] = {}
    for a in articles:
        by_domain.setdefault(a.domain, []).append(a)

    for domain, domain_articles in by_domain.items():
        _send_domain_digest(token, chat_id, domain, domain_articles, dashboard_url)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    required_vars = ["GROQ_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        logging.error(f"Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    # 1. Load sources
    sources = load_sources("sources.json")

    # 2. Fetch articles
    articles = await fetch_all(sources)
    logging.info(f"Fetched {len(articles)} unique articles")

    if not articles:
        logging.info("No articles fetched.")
        send_telegram([])
        return

    # 2b. Tag topics, rebuild each domain's stories over the window, and flag
    # hot articles from them — per domain, so a story never mixes two domains.
    by_domain: dict[str, list[Article]] = {}
    for a in articles:
        by_domain.setdefault(a.domain, []).append(a)

    groq_client_for_topics = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    model_full = (os.environ.get("GROQ_MODEL") or "openai/gpt-oss-120b").strip("'\"").strip()
    supabase_client = _get_supabase_client()
    for domain, domain_articles in by_domain.items():
        await refresh_stories(domain, domain_articles, supabase_client, groq_client_for_topics, model_full)
    logging.info(f"{sum(1 for a in articles if a.hot_topic)} articles tagged hot via stories")

    # 3. Classify with Groq
    logging.info(f"Classifying {len(articles)} articles with Groq...")
    classified = await classify_articles(articles)

    # 4. Save to Supabase
    save_to_supabase(classified, client=supabase_client)

    # 5. Send to Telegram
    logging.info("Sending articles to Telegram...")
    dashboard_url = os.environ.get("DASHBOARD_URL", "")
    send_telegram(classified, dashboard_url=dashboard_url)

    logging.info("AI Radar pipeline complete.")


if __name__ == "__main__":
    if "--backfill-topics" in sys.argv:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        asyncio.run(backfill_topics(os.environ.get("TOPIC_BACKFILL_MODEL")))
    else:
        asyncio.run(main())
