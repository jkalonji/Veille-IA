"""
AI Radar - Outil de Veille IA Automatise
Collecte, classifie et stocke quotidiennement l'actualite IA.
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import mktime

import aiohttp
import feedparser
from groq import Groq
from notion_client import Client as NotionClient
import requests

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
    category: str = ""
    sentiment: str = ""

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

    for entry in feed.entries[:50]:  # cap to avoid ArXiv floods
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
            published=pub_date.strftime("%Y-%m-%d") if pub_date else datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            description=clean_html(entry.get("summary", ""))[:300],
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
            published=pub_date.strftime("%Y-%m-%d") if pub_date else datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            description=clean_html(entry.get("summary", ""))[:300],
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
                published=datetime.fromtimestamp(hit.get("created_at_i", 0), tz=timezone.utc).strftime("%Y-%m-%d"),
                description=(hit.get("story_text") or "")[:300],
            ))

    logging.info(f"[{source['name']}] {len(articles)} articles")
    return articles


async def fetch_all(sources: list[dict]) -> list[Article]:
    """Fetch all sources in parallel, deduplicate by URL."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for src in sources:
            if src["type"] == "rss":
                tasks.append(fetch_rss(session, src))
            elif src["type"] == "reddit":
                tasks.append(fetch_reddit(session, src))
            elif src["type"] == "hn_api":
                tasks.append(fetch_hackernews(session, src))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    articles = []
    for result in results:
        if isinstance(result, Exception):
            logging.error(f"Fetch task failed: {result}")
        else:
            articles.extend(result)

    # Deduplicate by URL
    seen: set[str] = set()
    unique = []
    for a in articles:
        if a.url not in seen:
            seen.add(a.url)
            unique.append(a)

    return unique

# ---------------------------------------------------------------------------
# 3. Deduplication against Notion
# ---------------------------------------------------------------------------

async def get_existing_urls(notion: NotionClient, database_id: str) -> set[str]:
    """Fetch URLs from the last 48h in Notion to avoid duplicates."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%d")
    urls: set[str] = set()
    has_more = True
    start_cursor = None

    while has_more:
        kwargs: dict = {
            "database_id": database_id,
            "filter": {
                "property": "Date",
                "date": {"on_or_after": cutoff},
            },
            "page_size": 100,
        }
        if start_cursor:
            kwargs["start_cursor"] = start_cursor

        response = await notion.databases.query(**kwargs)

        for page in response.get("results", []):
            url_prop = page.get("properties", {}).get("URL", {})
            if url_prop.get("url"):
                urls.add(url_prop["url"])

        has_more = response.get("has_more", False)
        start_cursor = response.get("next_cursor")

    return urls

# ---------------------------------------------------------------------------
# 4. Classification with Groq
# ---------------------------------------------------------------------------

GROQ_SYSTEM_PROMPT = """Tu es un classificateur d'actualites IA. Pour chaque article, renvoie UNIQUEMENT un objet JSON avec deux cles :
- "category": une valeur parmi ["Innovation / Tech", "Politique / Regulation", "Business / Industrie", "Societe / Ethique", "Recherche Academique", "Drama / Controverses", "Geopolitique"]
- "sentiment": une valeur parmi ["Positif", "Negatif", "Neutre"]

Ne renvoie AUCUN texte supplementaire, AUCUN resume, AUCUNE explication. Uniquement l'objet JSON."""

VALID_CATEGORIES = {
    "Innovation / Tech",
    "Politique / Regulation",
    "Business / Industrie",
    "Societe / Ethique",
    "Recherche Academique",
    "Drama / Controverses",
    "Geopolitique",
}

VALID_SENTIMENTS = {"Positif", "Negatif", "Neutre"}


def classify_articles(articles: list[Article]) -> list[Article]:
    """Classify each article with Groq (category + sentiment)."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    for i, article in enumerate(articles):
        user_msg = f"Titre: {article.title}\nSource: {article.source}"
        if article.description:
            user_msg += f"\nDescription: {article.description}"

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            cat = result.get("category", "Innovation / Tech")
            sent = result.get("sentiment", "Neutre")
            article.category = cat if cat in VALID_CATEGORIES else "Innovation / Tech"
            article.sentiment = sent if sent in VALID_SENTIMENTS else "Neutre"
        except Exception as e:
            logging.warning(f"Groq error for '{article.title[:60]}': {e}")
            article.category = "Innovation / Tech"
            article.sentiment = "Neutre"

        # Rate limiting: pause every 5 articles
        if (i + 1) % 5 == 0:
            time.sleep(2)

    return articles

# ---------------------------------------------------------------------------
# 5. Push to Notion
# ---------------------------------------------------------------------------

SENTIMENT_MAP = {
    "Positif": "\U0001f7e2 Positif",
    "Negatif": "\U0001f534 Negatif",
    "Neutre": "\u26aa Neutre",
}


def push_to_notion(notion: NotionClient, database_id: str, articles: list[Article]) -> int:
    """Create a Notion page for each article. Returns count of successfully pushed articles."""
    pushed = 0
    for article in articles:
        try:
            notion.pages.create(
                parent={"database_id": database_id},
                properties={
                    "Titre": {
                        "title": [{"text": {"content": article.title[:2000]}}]
                    },
                    "URL": {
                        "url": article.url
                    },
                    "Date": {
                        "date": {"start": article.published}
                    },
                    "Source": {
                        "rich_text": [{"text": {"content": article.source}}]
                    },
                    "Pays d'origine": {
                        "rich_text": [{"text": {"content": article.country}}]
                    },
                    "Categorie IA": {
                        "select": {"name": article.category}
                    },
                    "Sentiment": {
                        "select": {"name": SENTIMENT_MAP.get(article.sentiment, "\u26aa Neutre")}
                    },
                },
            )
            pushed += 1
        except Exception as e:
            logging.error(f"Notion push failed for '{article.title[:60]}': {e}")

        # Respect Notion rate limit (3 req/s)
        time.sleep(0.35)

    return pushed

# ---------------------------------------------------------------------------
# 6. Telegram notification
# ---------------------------------------------------------------------------

def send_telegram(total: int, stats: dict[str, int]) -> None:
    """Send a summary message via Telegram Bot API."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    dashboard_url = os.environ.get("NOTION_DASHBOARD_URL", "")

    if total == 0:
        text = "\U0001f916 Radar IA : 0 nouveaux articles aujourd'hui."
    else:
        lines = [
            f"\U0001f916 *Radar IA termine* : {total} nouveaux articles !",
            "",
            f"\U0001f680 Innovation / Tech : {stats.get('Innovation / Tech', 0)}",
            f"\u2696\ufe0f Politique / Regulation : {stats.get('Politique / Regulation', 0)}",
            f"\U0001f4bc Business / Industrie : {stats.get('Business / Industrie', 0)}",
            f"\U0001f91d Societe / Ethique : {stats.get('Societe / Ethique', 0)}",
            f"\U0001f393 Recherche Academique : {stats.get('Recherche Academique', 0)}",
            f"\U0001f4a5 Drama / Controverses : {stats.get('Drama / Controverses', 0)}",
            f"\U0001f30d Geopolitique : {stats.get('Geopolitique', 0)}",
            "",
            f"\U0001f4ca [Consulter le Dashboard]({dashboard_url})",
        ]
        text = "\n".join(lines)

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        if resp.ok:
            logging.info("Telegram notification sent")
        else:
            logging.error(f"Telegram error: {resp.status_code} {resp.text}")
    except Exception as e:
        logging.error(f"Telegram send failed: {e}")

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Check required env vars
    required_vars = ["GROQ_API_KEY", "NOTION_TOKEN", "NOTION_DATABASE_ID", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        logging.error(f"Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    # 1. Load sources
    sources = load_sources("sources.json")

    # 2. Fetch all articles
    raw_articles = asyncio.run(fetch_all(sources))
    logging.info(f"Fetched {len(raw_articles)} unique articles")

    # 3. Deduplicate against Notion
    notion = NotionClient(auth=os.environ["NOTION_TOKEN"])
    database_id = os.environ["NOTION_DATABASE_ID"]
    existing_urls = get_existing_urls(notion, database_id)
    new_articles = [a for a in raw_articles if a.url not in existing_urls]
    logging.info(f"{len(new_articles)} new articles after deduplication ({len(existing_urls)} existing)")

    if not new_articles:
        send_telegram(0, {})
        return

    # 4. Classify with Groq
    logging.info(f"Classifying {len(new_articles)} articles with Groq...")
    classified = classify_articles(new_articles)

    # 5. Push to Notion
    logging.info("Pushing articles to Notion...")
    pushed = push_to_notion(notion, database_id, classified)
    logging.info(f"Pushed {pushed}/{len(classified)} articles to Notion")

    # 6. Send Telegram notification
    stats = compute_stats(classified)
    send_telegram(pushed, stats)

    logging.info("AI Radar pipeline complete.")


if __name__ == "__main__":
    main()

