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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import mktime

import aiohttp
import feedparser
from groq import AsyncGroq
import requests
from supabase import create_client

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
            published=pub_date.strftime("%Y-%m-%d") if pub_date else datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            description=clean_html(entry.get("summary", ""))[:200],
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
            description=clean_html(entry.get("summary", ""))[:200],
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
                description=(hit.get("story_text") or "")[:200],
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

    # Filter: keep only AI-related articles
    ai_keywords = {
        # Core AI concepts
        "ai", "artificial intelligence", "machine learning", "deep learning",
        "llm", "large language model", "neural network", "chatgpt", "gpt",
        "generative ai", "chatbot", "autonomous", "agi", "artificial general intelligence",
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

        # Vibe coding & AI-assisted dev
        "vibe coding", "ai coding", "code generation", "copilot", "cursor",
        "devin", "github copilot",

        # Model optimization
        "quantization", "pruning", "distillation", "knowledge distillation",
        "lora", "qlora", "peft", "model compression", "speculative decoding",
        "flash attention", "inference optimization", "efficient inference",

        # Hardware: Nvidia, competitors & TSMC
        "nvidia", "amd", "intel", "qualcomm", "tsmc", "h100", "h200", "b200",
        "blackwell", "hopper", "tpu", "cerebras", "graphcore", "tenstorrent",
        "gaudi", "arm chip",

        # Major players (established)
        "openai", "anthropic", "deepmind", "gemini", "meta ai", "mistral",
        "xai", "grok", "cohere", "hugging face", "stability ai", "runway",

        # Emerging players
        "deepseek", "qwen", "perplexity", "together ai",

        # Agents & autonomy
        "ai agent", "agentic", "model context protocol", "mcp", "autonomous agent",

        # Safety, ethics & regulation
        "ai safety", "alignment", "hallucination", "ai regulation", "eu ai act",
        "responsible ai", "interpretability", "explainability", "deepfake",
        "ai governance", "bias",

        # Performance & benchmarks
        "benchmark", "evals", "open source model", "open weights", "edge ai",
        "on-device ai",

        # Infrastructure
        "data center", "semiconductor", "chip", "robot", "automation",
    }
    filtered = []
    for a in unique:
        text = (a.title + " " + a.description).lower()
        if any(kw in text for kw in ai_keywords):
            filtered.append(a)

    logging.info(f"{len(filtered)}/{len(unique)} articles kept after AI filter")
    return filtered

# ---------------------------------------------------------------------------
# 3. Classification with Groq
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


async def _classify_one(client: AsyncGroq, model: str, article: Article) -> None:
    """Classify a single article in-place."""
    user_msg = f"Titre: {article.title}\nSource: {article.source}"
    if article.description:
        user_msg += f"\nDescription: {article.description}"

    try:
        response = await client.chat.completions.create(
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


async def classify_articles(articles: list[Article], batch_size: int = 15, batch_pause: float = 10.0) -> list[Article]:
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip("'\"").strip()

    batches = [articles[i:i + batch_size] for i in range(0, len(articles), batch_size)]
    for batch_idx, batch in enumerate(batches):
        logging.info(f"Groq: classifying batch {batch_idx + 1}/{len(batches)} ({len(batch)} articles)")
        await asyncio.gather(*[_classify_one(client, model, a) for a in batch])
        if batch_idx < len(batches) - 1:
            logging.info(f"Groq: sleeping {batch_pause}s before next batch")
            await asyncio.sleep(batch_pause)

    return articles

# ---------------------------------------------------------------------------
# 4. Supabase
# ---------------------------------------------------------------------------

def save_to_supabase(articles: list[Article]) -> None:
    """Upsert articles into Supabase, ignoring duplicates by URL."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        logging.warning("SUPABASE_URL or SUPABASE_KEY not set — skipping Supabase save")
        return

    client = create_client(url, key)
    rows = [
        {
            "title": a.title,
            "url": a.url,
            "source": a.source,
            "country": a.country,
            "published": a.published,
            "description": a.description,
            "category": a.category,
            "sentiment": a.sentiment,
        }
        for a in articles
    ]

    try:
        client.table("articles").upsert(rows, on_conflict="url").execute()
        logging.info(f"Supabase: upserted {len(rows)} articles")
    except Exception as e:
        logging.error(f"Supabase upsert error: {e}")


# ---------------------------------------------------------------------------
# 5. Telegram
# ---------------------------------------------------------------------------

SENTIMENT_EMOJI = {"Positif": "🟢", "Negatif": "🔴", "Neutre": "⚪"}
CATEGORY_EMOJI = {
    "Innovation / Tech":      "🚀",
    "Politique / Regulation": "⚖️",
    "Business / Industrie":   "💼",
    "Societe / Ethique":      "🤝",
    "Recherche Academique":   "🎓",
    "Drama / Controverses":   "💥",
    "Geopolitique":           "🌍",
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


def send_telegram(articles: list[Article]) -> None:
    """Send a header summary then all articles in batches to Telegram."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    today = datetime.now(timezone.utc).strftime("%d/%m/%Y")

    if not articles:
        _post_telegram(token, chat_id, "🤖 Radar IA : 0 nouveaux articles aujourd'hui.")
        return

    # Header with stats
    stats = compute_stats(articles)
    header_lines = [f"🤖 <b>Radar IA — {today}</b>", f"📰 {len(articles)} articles collectés", ""]
    for cat, emoji in CATEGORY_EMOJI.items():
        count = stats.get(cat, 0)
        if count:
            header_lines.append(f"{emoji} {cat} : {count}")
    _post_telegram(token, chat_id, "\n".join(header_lines))

    # Articles in batches of 10
    batch: list[str] = []
    batch_chars = 0

    for article in articles:
        sent_emoji = SENTIMENT_EMOJI.get(article.sentiment, "⚪")
        cat_emoji = CATEGORY_EMOJI.get(article.category, "📌")
        desc = article.description.strip()
        if desc and not desc.endswith((".", "!", "?")):
            desc += "…"

        entry_lines = [
            f"{sent_emoji} <b>{article.title}</b>",
            f"{cat_emoji} {article.category} | {article.country} {article.source}",
        ]
        if desc:
            entry_lines.append(f"<i>{desc}</i>")
        entry_lines.append(f'<a href="{article.url}">🔗 Lire l\'article</a>')
        entry = "\n".join(entry_lines)

        # Flush batch if adding this entry would exceed Telegram's 4096-char limit
        if batch and batch_chars + len(entry) + 2 > 4000:
            _post_telegram(token, chat_id, "\n\n".join(batch))
            batch = []
            batch_chars = 0
            import time; time.sleep(0.5)  # avoid Telegram rate limit

        batch.append(entry)
        batch_chars += len(entry) + 2

    if batch:
        _post_telegram(token, chat_id, "\n\n".join(batch))

    logging.info(f"Telegram: sent {len(articles)} articles")

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

    # 3. Classify with Groq
    logging.info(f"Classifying {len(articles)} articles with Groq...")
    classified = await classify_articles(articles)

    # 4. Save to Supabase
    save_to_supabase(classified)

    # 5. Send to Telegram
    logging.info("Sending articles to Telegram...")
    send_telegram(classified)

    logging.info("AI Radar pipeline complete.")


if __name__ == "__main__":
    asyncio.run(main())
