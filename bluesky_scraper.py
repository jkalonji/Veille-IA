"""
Bluesky scraper using the atproto AsyncClient.
Fetches top posts for configured AI / robotics topics from the last 24h.

Authentication (two env vars required):
  BSKY_HANDLE       — your Bluesky handle, e.g. yourname.bsky.social
  BSKY_APP_PASSWORD — an App Password created at bsky.app/settings/app-passwords
                      (NOT your main account password)
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

try:
    from atproto import AsyncClient
    from atproto_client.exceptions import AtProtocolError
    _ATPROTO_AVAILABLE = True
except ImportError:
    _ATPROTO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post_url(handle: str, uri: str) -> str:
    """Convert AT URI (at://did:.../app.bsky.feed.post/rkey) to bsky.app URL."""
    rkey = uri.split("/")[-1]
    return f"https://bsky.app/profile/{handle}/post/{rkey}"


def _engagement_score(post) -> int:
    """Weighted virality score: likes + 3×reposts + 2×replies.
    Reposts carry more weight because they actively spread content."""
    return (
        (post.like_count    or 0)
        + (post.repost_count or 0) * 3
        + (post.reply_count  or 0) * 2
    )


def _parse_bsky_date(date_str: str) -> str:
    """Return ISO 8601 datetime string; fall back to now on parse error."""
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

async def _fetch_one_topic(client: AsyncClient, source: dict, since: str) -> list[dict]:
    """Fetch and filter posts for a single Bluesky source config."""
    query     = source["query"]
    min_score = source.get("min_score", 20)
    limit     = source.get("count", 25)
    name      = source["name"]
    country   = source.get("country", "🌐")

    try:
        resp = await client.app.bsky.feed.search_posts(
            params={
                "q":     query,
                "limit": min(limit * 3, 100),  # over-fetch, then filter by score
                "sort":  "top",
                "lang":  "en",
                "since": since,
            }
        )
    except Exception as e:
        logging.error(f"[{name}] Bluesky search error: {e}")
        return []

    articles = []
    for post in (resp.posts or []):
        if _engagement_score(post) < min_score:
            continue

        text  = (post.record.text or "").strip()
        if not text:
            continue

        # Use first line (or first 120 chars) as title — same convention as twitter_scraper
        first_line = text.split("\n")[0].strip()
        title = (first_line[:117] + "…") if len(first_line) > 120 else first_line
        if not title:
            continue

        handle   = post.author.handle
        post_url = _post_url(handle, post.uri)
        articles.append({
            "title":       title,
            "url":         post_url,
            "source":      f"Bluesky / @{handle}",
            "country":     country,
            "published":   _parse_bsky_date(post.record.created_at),
            "description": text[:200],
        })
        if len(articles) >= limit:
            break

    logging.info(f"[{name}] {len(articles)} posts kept (min_score={min_score})")
    return articles


async def fetch_all_bluesky(sources: list[dict]) -> list[dict]:
    """Fetch all sources of type 'bluesky'. Returns list of article dicts."""
    bluesky_sources = [
        s for s in sources
        if s.get("type") == "bluesky" and s.get("enabled", True)
    ]
    if not bluesky_sources:
        return []

    if not _ATPROTO_AVAILABLE:
        logging.warning("[Bluesky] atproto not installed — skipping. Run: pip install 'atproto>=0.0.55'")
        return []

    handle   = os.environ.get("BSKY_HANDLE")
    password = os.environ.get("BSKY_APP_PASSWORD")
    if not handle or not password:
        logging.warning("[Bluesky] BSKY_HANDLE or BSKY_APP_PASSWORD not set — skipping.")
        return []

    try:
        client = AsyncClient()
        await client.login(handle, password)
        logging.info(f"[Bluesky] Authenticated as @{handle}")
    except Exception as e:
        logging.error(f"[Bluesky] Login failed: {e}")
        return []

    # Only posts from the last 24h
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_articles: list[dict] = []
    seen_urls: set[str] = set()

    for i, source in enumerate(bluesky_sources):
        posts = await _fetch_one_topic(client, source, since)
        for p in posts:
            if p["url"] not in seen_urls:
                seen_urls.add(p["url"])
                all_articles.append(p)
        # Respectful rate-limiting between topic searches
        if i < len(bluesky_sources) - 1:
            await asyncio.sleep(1)

    logging.info(f"[Bluesky] {len(all_articles)} unique posts across {len(bluesky_sources)} topics")
    return all_articles
