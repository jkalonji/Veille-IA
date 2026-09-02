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
from dataclasses import dataclass
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
    hot_reason: str = ""    # groq content classification: "debat"|"tech"|"societe"|"tendance"
    summary: str = ""       # groq-generated 1-sentence summary in French
    story_id: int | None = None  # cross-day story this article was matched to, if any

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
# 0b. Topic clustering (replaces keyword-based hot detection)
# ---------------------------------------------------------------------------

_BASE_NGRAM_STOPWORDS = _MENTION_STOPWORDS | {
    "using", "will", "make", "take", "open", "help", "work", "need",
    "many", "them", "know", "find", "some", "here", "even", "like",
    "time", "year", "week", "ways", "could", "would", "your", "our",
    "first", "show", "use", "say", "get", "give", "being", "most",
    "report", "says", "look", "now", "back", "move", "inside",
    "before", "between", "while", "through", "against", "without",
    "around", "next", "within", "each", "such", "does", "did",
}

# Domain-specific stopword extensions — words too generic within a given
# domain's news flow to define a meaningful cluster identifier.
_IA_EXTRA_STOPWORDS = {
    "artificial", "intelligence", "machine", "learning", "technology",
    "digital", "software", "platform", "online", "system", "systems",
    "tool", "tools", "model", "models", "neural", "network", "networks",
    "generative", "language", "large", "latest", "based", "driven",
    "powered", "enabled", "future", "global", "world", "industry",
    "company", "companies", "startup", "startups", "researchers",
    "research", "paper", "papers", "study", "team", "users", "space",
    "launches", "launch", "release", "releases", "announces", "announced",
    "introduces", "unveils", "brings", "update", "updates", "version",
}

DOMAIN_NGRAM_STOPWORDS: dict[str, set[str]] = {
    "ia": _BASE_NGRAM_STOPWORDS | _IA_EXTRA_STOPWORDS,
}

# Ngram-level blocklist — phrases too generic to define a meaningful cluster,
# scoped per domain (only "ia" has one populated for now).
DOMAIN_GENERIC_NGRAMS: dict[str, set[str]] = {
    "ia": {
        "artificial intelligence", "machine learning", "deep learning",
        "large language", "language model", "language models",
        "generative ai", "neural network", "neural networks",
        "open source", "new model", "latest model", "ai model", "ai models",
        "ai tools", "ai tool", "ai system", "ai systems", "ai research",
        "ai company", "ai startup", "ai technology", "ai applications",
        "tech news", "tech industry", "tech company",
        "research paper", "new paper", "new study", "new research",
        "ai era", "ai future", "ai development", "ai capabilities",
    },
}


def _extract_title_ngrams(title: str, domain: str = "ia") -> set[str]:
    """Extract bigrams and trigrams from a title, filtering domain-scoped stopwords."""
    stopwords = DOMAIN_NGRAM_STOPWORDS.get(domain, _BASE_NGRAM_STOPWORDS)
    # Match word tokens including hyphenated terms (gpt-4o, ai-agent)
    words = re.findall(r"[a-z][a-z0-9\-]*", title.lower())
    words = [w for w in words if len(w) >= 2 and w not in stopwords]
    ngrams: set[str] = set()
    for i in range(len(words) - 1):
        ngrams.add(f"{words[i]} {words[i + 1]}")
    for i in range(len(words) - 2):
        ngrams.add(f"{words[i]} {words[i + 1]} {words[i + 2]}")
    return ngrams


def extract_topic_clusters(articles: list["Article"], min_articles: int = 3, domain: str = "ia") -> list[dict]:
    """Cluster articles by shared bigrams/trigrams in their titles.

    Assumes all articles belong to the same `domain` (callers group by domain
    before calling this, so hot-topic clusters never mix domains).

    Returns a list of clusters sorted by score (article_count × source_count),
    each with: phrase, label, articles, article_count, source_count, score.
    Only clusters with ≥ min_articles distinct articles are returned.
    """
    if not articles:
        return []

    generic_ngrams = DOMAIN_GENERIC_NGRAMS.get(domain, set())
    art_ngrams = [(a, _extract_title_ngrams(a.title, domain)) for a in articles]

    # Count how many articles contain each ngram
    ngram_to_arts: dict[str, list] = {}
    for a, ngrams in art_ngrams:
        for ng in ngrams:
            ngram_to_arts.setdefault(ng, []).append(a)

    # Keep ngrams with ≥ min_articles distinct articles AND ≥ 2 distinct sources
    # AND not in the generic-phrases blocklist
    candidate_clusters = []
    for ng, arts in ngram_to_arts.items():
        if ng in generic_ngrams:
            continue
        if len(arts) < min_articles:
            continue
        sources = {a.source for a in arts}
        if len(sources) < 2:
            continue
        score = len(arts) * len(sources)
        candidate_clusters.append({
            "phrase": ng,
            "label": ng.title(),   # placeholder, overwritten by name_topic_clusters
            "articles": arts,
            "article_count": len(arts),
            "source_count": len(sources),
            "score": score,
        })

    if not candidate_clusters:
        return []

    # Sort by score desc, then merge highly-overlapping clusters (≥70% article overlap)
    candidate_clusters.sort(key=lambda c: (-c["score"], -len(c["phrase"])))
    merged: list[dict] = []
    for c in candidate_clusters:
        urls = {a.url for a in c["articles"]}
        is_duplicate = False
        for existing in merged:
            existing_urls = {a.url for a in existing["articles"]}
            union = urls | existing_urls
            overlap = len(urls & existing_urls) / len(union) if union else 0
            if overlap >= 0.7:
                # Keep longer phrase as the label candidate
                if len(c["phrase"]) > len(existing["phrase"]):
                    existing["phrase"] = c["phrase"]
                is_duplicate = True
                break
        if not is_duplicate:
            merged.append(c)

    merged.sort(key=lambda c: -c["score"])
    logging.info(f"Topic clusters: {len(merged)} clusters from {len(articles)} articles")
    return merged


# Per-domain config for the Groq cluster-naming prompt: words to forbid in
# labels, example labels to steer style, and a lowercase blocklist used to
# reject low-effort labels returned by Groq.
DOMAIN_CLUSTER_NAMING: dict[str, dict] = {
    "ia": {
        "forbidden": "AI, Tech, Model, Artificial Intelligence, Machine Learning, Technology, Research, Innovation, Development",
        "examples": "'OpenAI GPT-5', 'EU AI Act Vote', 'Anthropic Claude 4', 'NVIDIA Blackwell GPU', 'Sam Altman Senate Hearing'",
        "label_blocklist": {
            "ai", "tech", "model", "models", "artificial intelligence",
            "machine learning", "technology", "research", "innovation",
            "development", "news", "update", "latest", "new",
        },
    },
    "politique_evenements": {
        "forbidden": "Politics, World, News, Crisis, Conflict, Government, Country, International",
        "examples": "'Sudan Coup Attempt', 'Turkey Earthquake Response', 'EU Russia Sanctions Package', 'Venezuela Election Protests'",
        "label_blocklist": {
            "politics", "world", "news", "crisis", "conflict", "government",
            "country", "international", "update", "latest", "new",
        },
    },
    "matieres_premieres": {
        "forbidden": "Commodities, Prices, Market, Energy, Oil, Resources",
        "examples": "'OPEC Production Cut', 'Brent Crude Rally', 'Lithium Supply Shortage', 'Chile Copper Strike'",
        "label_blocklist": {
            "commodities", "prices", "market", "energy", "oil", "resources",
            "update", "latest", "new",
        },
    },
    "finance": {
        "forbidden": "Finance, Markets, Stocks, Economy, Banking",
        "examples": "'Fed Rate Decision', 'Nvidia Earnings Beat', 'Nasdaq Correction', 'ECB Rate Hold'",
        "label_blocklist": {
            "finance", "markets", "stocks", "economy", "banking",
            "update", "latest", "new",
        },
    },
    "services": {
        "forbidden": "Economy, Jobs, Services, Growth, Sector",
        "examples": "'US Jobs Report', 'Eurozone Inflation Data', 'Retail Sales Slump', 'Housing Market Cooldown'",
        "label_blocklist": {
            "economy", "jobs", "services", "growth", "sector",
            "update", "latest", "new",
        },
    },
}


async def name_topic_clusters(clusters: list[dict], client, model: str, domain: str = "ia") -> list[dict]:
    """Ask Groq to assign clean English labels to topic clusters (single batch call)."""
    if not clusters:
        return clusters

    naming = DOMAIN_CLUSTER_NAMING.get(domain, DOMAIN_CLUSTER_NAMING["ia"])
    top = clusters[:12]
    items = [
        {
            "id": i,
            "phrase": c["phrase"],
            "titles": [a.title for a in c["articles"][:3]],
        }
        for i, c in enumerate(top)
    ]
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    "You are a news editor naming topic clusters for a news dashboard.\n"
                    "For each cluster, create a SPECIFIC label (2-5 words, Title Case) that names "
                    "the EXACT subject — which entity, event, or place is involved.\n\n"
                    "RULES:\n"
                    "- Use proper nouns from the sample titles whenever possible\n"
                    f"- FORBIDDEN words (never use): {naming['forbidden']}\n"
                    "- The label must answer: WHAT specifically is happening? WHO is involved?\n"
                    f"- Examples of good labels: {naming['examples']}\n\n"
                    "Return JSON: {\"labels\": [{\"id\": <id>, \"label\": \"...\"}]}\n\n"
                    + json.dumps(items, ensure_ascii=False)
                ),
            }],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        id_to_label = {item["id"]: item["label"] for item in data.get("labels", [])}
        _LABEL_BLOCKLIST = naming["label_blocklist"]
        for i, c in enumerate(top):
            raw = id_to_label.get(i, "")
            # Reject label if it's in the blocklist or is a single generic word
            if raw and raw.lower() not in _LABEL_BLOCKLIST and len(raw) > 3:
                c["label"] = raw
            else:
                c["label"] = c["phrase"].title()
        logging.info(f"Topic clusters named: {[c['label'] for c in top]}")
    except Exception as e:
        logging.warning(f"Topic naming failed: {e} — using phrase-based labels")
        for c in top:
            c["label"] = c["phrase"].title()

    for c in clusters[12:]:
        c["label"] = c["phrase"].title()

    return clusters


# ---------------------------------------------------------------------------
# 0c. Story tracking (cross-day continuation of hot topic clusters)
# ---------------------------------------------------------------------------
# A "story" is a persistent identity a hot cluster gets matched to across
# multiple daily runs (e.g. an announcement -> its reactions -> the fallout,
# spread over several days, stays one story instead of N unrelated clusters).
# Stored in the `stories` table; `articles.story_id` points into it.

STORY_IDLE_DAYS = 4         # a story auto-closes after this many days with no new article
STORY_CANDIDATE_LIMIT = 20  # max open stories sent to Groq for matching context


def _fetch_open_stories(client, domain: str) -> list[dict]:
    """Fetch this domain's open stories, most recently active first."""
    if client is None:
        return []
    try:
        resp = (
            client.table("stories")
            .select("id, label, first_seen, last_seen, article_count, recent_titles")
            .eq("domain", domain)
            .eq("status", "open")
            .order("last_seen", desc=True)
            .limit(STORY_CANDIDATE_LIMIT)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        logging.warning(f"Stories fetch failed ({domain}): {e} — story tracking disabled for this run")
        return []


def _match_clusters_ngram(clusters: list[dict], open_stories: list[dict], domain: str) -> list[dict | None]:
    """Fallback matcher: overlap of title n-grams between today's clusters and
    each open story's recent titles. Used when Groq matching is unavailable."""
    story_ngrams = []
    for s in open_stories:
        ngrams: set[str] = set()
        for t in (s.get("recent_titles") or "").split("|"):
            ngrams |= _extract_title_ngrams(t, domain)
        story_ngrams.append((s, ngrams))

    matches: list[dict | None] = []
    for c in clusters:
        cluster_ngrams: set[str] = set()
        for a in c["articles"][:3]:
            cluster_ngrams |= _extract_title_ngrams(a.title, domain)
        best, best_overlap = None, 0
        for s, ngrams in story_ngrams:
            overlap = len(cluster_ngrams & ngrams)
            if overlap > best_overlap:
                best, best_overlap = s, overlap
        matches.append(best if best_overlap >= 2 else None)
    return matches


async def match_clusters_to_stories(
    clusters: list[dict], open_stories: list[dict], client, model: str, domain: str
) -> list[dict | None]:
    """Match today's topic clusters to existing open stories (cross-day tracking).

    Returns a list parallel to `clusters`: the matched story dict, or None if the
    cluster starts a new story. Falls back to n-gram overlap if Groq is unavailable
    or errors — same resilience pattern as `name_topic_clusters`.
    """
    if not clusters:
        return []
    if not open_stories:
        return [None] * len(clusters)

    story_items = [
        {"id": s["id"], "label": s["label"], "recent_titles": (s.get("recent_titles") or "").split("|")[:3]}
        for s in open_stories
    ]
    cluster_items = [
        {"index": i, "label": c["label"], "titles": [a.title for a in c["articles"][:3]]}
        for i, c in enumerate(clusters)
    ]
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    "You track ongoing news stories across multiple days for a news dashboard.\n"
                    "Below are OPEN_STORIES (already being tracked) and TODAY_CLUSTERS (today's new "
                    "topic clusters). For each cluster, decide if it is a continuation of one of the "
                    "open stories — same underlying event/story, even if the angle shifted (e.g. "
                    "announcement -> reactions -> consequences) — or if it is genuinely a new story.\n\n"
                    "Return JSON: {\"matches\": [{\"index\": <cluster index>, \"story_id\": <id or null>}]}\n\n"
                    f"OPEN_STORIES = {json.dumps(story_items, ensure_ascii=False)}\n\n"
                    f"TODAY_CLUSTERS = {json.dumps(cluster_items, ensure_ascii=False)}"
                ),
            }],
            temperature=0.1,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        id_to_story = {s["id"]: s for s in open_stories}
        result: list[dict | None] = [None] * len(clusters)
        for m in data.get("matches", []):
            idx, story_id = m.get("index"), m.get("story_id")
            if isinstance(idx, int) and 0 <= idx < len(clusters) and story_id in id_to_story:
                result[idx] = id_to_story[story_id]
        logging.info(
            f"Story matching [{domain}]: {sum(1 for r in result if r)}/{len(clusters)} "
            f"clusters matched to open stories"
        )
        return result
    except Exception as e:
        logging.warning(f"Story matching via Groq failed ({domain}): {e} — falling back to n-gram matching")
        return _match_clusters_ngram(clusters, open_stories, domain)


def _apply_story_matches(
    client, domain: str, clusters: list[dict], matches: list[dict | None], today: str
) -> dict[str, int]:
    """Create/update story rows in Supabase for today's clusters. Returns a
    url -> story_id map used to stamp `story_id` onto matched articles."""
    if client is None:
        return {}
    url_to_story: dict[str, int] = {}
    for cluster, matched in zip(clusters, matches):
        titles = [a.title for a in cluster["articles"][:5]]
        try:
            if matched:
                story_id = matched["id"]
                recent = "|".join((titles + (matched.get("recent_titles") or "").split("|"))[:6])
                client.table("stories").update({
                    "last_seen": today,
                    "article_count": matched.get("article_count", 0) + cluster["article_count"],
                    "recent_titles": recent,
                }).eq("id", story_id).execute()
            else:
                resp = client.table("stories").insert({
                    "domain": domain,
                    "label": cluster["label"],
                    "first_seen": today,
                    "last_seen": today,
                    "article_count": cluster["article_count"],
                    "status": "open",
                    "recent_titles": "|".join(titles),
                }).execute()
                story_id = resp.data[0]["id"]
        except Exception as e:
            logging.warning(f"Story upsert failed for cluster '{cluster['label']}' ({domain}): {e}")
            continue
        for a in cluster["articles"]:
            url_to_story[a.url] = story_id
    return url_to_story


def _close_stale_stories(client, domain: str, today: str, idle_days: int = STORY_IDLE_DAYS) -> None:
    """Auto-close open stories that haven't seen a new article in `idle_days` days."""
    if client is None:
        return
    cutoff = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=idle_days)).strftime("%Y-%m-%d")
    try:
        (
            client.table("stories")
            .update({"status": "closed"})
            .eq("domain", domain)
            .eq("status", "open")
            .lt("last_seen", cutoff)
            .execute()
        )
    except Exception as e:
        logging.warning(f"Closing stale stories failed ({domain}): {e}")


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

            articles.append(Article(
                title=title,
                url=url,
                source=source["name"],
                country=source["country"],
                published=datetime.fromtimestamp(hit.get("created_at_i", 0), tz=timezone.utc).isoformat(),
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
            except (KeyError, ValueError):
                pub_date = datetime.now(timezone.utc)

            articles.append(Article(
                title=title,
                url=url,
                source=source["name"],
                country=source.get("country", "🌍"),
                published=pub_date.isoformat(),
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


async def _classify_one(client: AsyncGroq, model_fast: str, model_full: str, article: Article) -> None:
    """Classify a single article in-place, using its domain's taxonomy/prompt.
    Hot articles use model_full (70b): category + sentiment + country + summary.
    Non-hot articles use model_fast (8b): category + sentiment + country only.
    This cuts ~65% of 70b token usage on a typical day."""
    taxo = DOMAIN_TAXONOMY.get(article.domain, DOMAIN_TAXONOMY["ia"])
    valid_categories = set(taxo["categories"])
    default_category = taxo["default_category"]

    user_msg = f"Titre: {article.title}\nSource: {article.source}"
    if article.description:
        user_msg += f"\nDescription: {article.description}"

    if article.hot_topic:
        model         = model_full
        system_prompt = _build_groq_prompt(article.domain, with_summary=True)
        max_tokens    = 180   # category + sentiment + country + summary
    else:
        model         = model_fast
        system_prompt = _build_groq_prompt(article.domain, with_summary=False)
        max_tokens    = 80    # category + sentiment + country only

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
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
    except Exception as e:
        logging.warning(f"Groq error for '{article.title[:60]}': {e}")
        article.category  = default_category
        article.sentiment = "Neutre"
        article.country   = "Global"


async def classify_articles(articles: list[Article], batch_size: int = 15, batch_pause: float = 10.0) -> list[Article]:
    client     = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    model_full = (os.environ.get("GROQ_MODEL")      or "openai/gpt-oss-120b").strip("'\"").strip()
    model_fast = (os.environ.get("GROQ_MODEL_FAST") or "openai/gpt-oss-20b").strip("'\"").strip()

    hot_count  = sum(1 for a in articles if a.hot_topic)
    logging.info(f"Groq: {len(articles)} articles — {hot_count} hot ({model_full}) + {len(articles)-hot_count} non-hot ({model_fast})")

    batches = [articles[i:i + batch_size] for i in range(0, len(articles), batch_size)]
    for batch_idx, batch in enumerate(batches):
        logging.info(f"Groq: classifying batch {batch_idx + 1}/{len(batches)} ({len(batch)} articles)")
        await asyncio.gather(*[_classify_one(client, model_fast, model_full, a) for a in batch])
        if batch_idx < len(batches) - 1:
            logging.info(f"Groq: sleeping {batch_pause}s before next batch")
            await asyncio.sleep(batch_pause)

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
    _OPTIONAL_COLS = ("hot_source", "hot_reason", "summary", "mention_count", "supa_hot", "domain", "story_id")

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

    # 2b. Extract topic clusters and mark hot articles — scoped per domain so a
    # cluster never mixes articles from two different domains.
    by_domain: dict[str, list[Article]] = {}
    for a in articles:
        by_domain.setdefault(a.domain, []).append(a)

    groq_client_for_topics = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    model_full = (os.environ.get("GROQ_MODEL") or "openai/gpt-oss-120b").strip("'\"").strip()
    supabase_client = _get_supabase_client()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Build url → cluster map and apply to articles
    url_to_cluster: dict[str, dict] = {}
    url_to_story: dict[str, int] = {}
    for domain, domain_articles in by_domain.items():
        # Try strict threshold first (≥3 articles, ≥2 sources); fall back to relaxed (≥2 articles)
        clusters = extract_topic_clusters(domain_articles, min_articles=3, domain=domain)
        if not clusters:
            logging.info(f"[{domain}] No clusters at min=3 — retrying with min=2")
            clusters = extract_topic_clusters(domain_articles, min_articles=2, domain=domain)

        if clusters:
            clusters = await name_topic_clusters(clusters, groq_client_for_topics, model_full, domain=domain)

            # Cross-day story tracking: match today's clusters against open stories
            open_stories = _fetch_open_stories(supabase_client, domain)
            matches = await match_clusters_to_stories(
                clusters, open_stories, groq_client_for_topics, model_full, domain=domain
            )
            url_to_story.update(_apply_story_matches(supabase_client, domain, clusters, matches, today_str))

        _close_stale_stories(supabase_client, domain, today_str)

        for c in clusters:
            for art in c["articles"]:
                if art.url not in url_to_cluster or c["score"] > url_to_cluster[art.url]["score"]:
                    url_to_cluster[art.url] = c

    hot_count = 0
    for art in articles:
        cluster = url_to_cluster.get(art.url)
        if cluster:
            art.hot_topic = True
            art.hot_reason = cluster["label"]
            art.mention_count = cluster["article_count"]
            art.supa_hot = cluster["article_count"] >= 5
            art.story_id = url_to_story.get(art.url)
            hot_count += 1
        else:
            art.hot_topic = False
            art.hot_reason = ""
            art.mention_count = 0
            art.supa_hot = False
            art.story_id = None
    logging.info(f"{hot_count} articles tagged hot via topic clustering")

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
    asyncio.run(main())
