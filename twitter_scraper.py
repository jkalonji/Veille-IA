"""
Twitter/X scraper using twikit.
Fetches top tweets for configured AI topics without requiring a paid API key.

Authentication (by priority):
  1. TWITTER_COOKIES env var — base64-encoded cookies.json (recommended for CI)
  2. cookies.json file in working directory (local runs after first login)
  3. TWITTER_USERNAME + TWITTER_EMAIL + TWITTER_PASSWORD env vars (first-time login)

Run `python twitter_scraper.py --setup` to log in and save cookies.json locally.
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import tempfile
from datetime import datetime, timezone

from twikit import Client

# Twitter date string format returned by twikit
_TWITTER_DATE_FMT = "%a %b %d %H:%M:%S %z %Y"


def _parse_date(date_str: str) -> str:
    """Convert Twitter date string to YYYY-MM-DD. Returns today on failure."""
    try:
        dt = datetime.strptime(date_str, _TWITTER_DATE_FMT)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


async def get_authenticated_client() -> Client | None:
    """Create a twikit Client, loading auth from env or credentials."""
    client = Client("en-US")

    # --- Option 1: TWITTER_COOKIES env var (base64-encoded JSON) ---
    cookies_b64 = os.getenv("TWITTER_COOKIES")
    if cookies_b64:
        try:
            cookies_json = base64.b64decode(cookies_b64).decode()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write(cookies_json)
                tmp = f.name
            client.load_cookies(tmp)
            os.unlink(tmp)
            logging.info("[Twitter] Auth via TWITTER_COOKIES env var")
            return client
        except Exception as e:
            logging.warning(f"[Twitter] Could not load TWITTER_COOKIES: {e}")

    # --- Option 2: cookies.json file ---
    if os.path.exists("cookies.json"):
        client.load_cookies("cookies.json")
        logging.info("[Twitter] Auth via cookies.json")
        return client

    # --- Option 3: username + password (writes cookies.json for future runs) ---
    username = os.getenv("TWITTER_USERNAME")
    password = os.getenv("TWITTER_PASSWORD")
    email = os.getenv("TWITTER_EMAIL")

    if not username or not password:
        logging.warning(
            "[Twitter] No auth found. Set TWITTER_COOKIES, cookies.json, "
            "or TWITTER_USERNAME/PASSWORD. Twitter sources will be skipped."
        )
        return None

    try:
        logging.info(f"[Twitter] Logging in as @{username}…")
        await client.login(
            auth_info_1=username,
            auth_info_2=email,
            password=password,
        )
        client.save_cookies("cookies.json")
        logging.info("[Twitter] Login OK — cookies saved to cookies.json")
        return client
    except Exception as e:
        logging.error(f"[Twitter] Login failed: {e}")
        return None


async def _fetch_one_topic(client: Client, source: dict) -> list[dict]:
    """Fetch top tweets for a single topic source config."""
    query = source["query"]
    min_likes = source.get("min_likes", 150)
    count = source.get("count", 20)
    name = source["name"]
    country = source.get("country", "🌐")

    try:
        tweets = await client.search_tweet(query, product="Top", count=count)
    except Exception as e:
        logging.error(f"[{name}] Twitter search error: {e}")
        return []

    articles = []
    for tweet in tweets:
        likes = tweet.favorite_count or 0
        if likes < min_likes:
            continue

        text = tweet.text or ""
        handle = tweet.user.screen_name if tweet.user else "unknown"
        url = f"https://x.com/{handle}/status/{tweet.id}"

        # First line (or first 120 chars) as title
        first_line = text.split("\n")[0].strip()
        title = (first_line[:117] + "…") if len(first_line) > 120 else first_line

        articles.append({
            "title": title,
            "url": url,
            "source": f"Twitter / @{handle}",
            "country": country,
            "published": _parse_date(tweet.created_at),
            "description": text[:200],
        })

    logging.info(f"[{name}] {len(articles)} tweets kept (min_likes={min_likes})")
    return articles


async def fetch_all_twitter(sources: list[dict]) -> list[dict]:
    """Fetch all sources of type 'twitter'. Returns list of article dicts."""
    twitter_sources = [s for s in sources if s.get("type") == "twitter" and s.get("enabled", True)]
    if not twitter_sources:
        return []

    client = await get_authenticated_client()
    if client is None:
        return []

    all_articles: list[dict] = []
    for i, source in enumerate(twitter_sources):
        articles = await _fetch_one_topic(client, source)
        all_articles.extend(articles)
        if i < len(twitter_sources) - 1:
            await asyncio.sleep(2)  # Respectful rate limiting between searches

    # Deduplicate by URL
    seen: set[str] = set()
    unique = []
    for a in all_articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)

    logging.info(f"[Twitter] {len(unique)} unique tweets fetched across {len(twitter_sources)} topics")
    return unique


# ---------------------------------------------------------------------------
# CLI: python twitter_scraper.py --setup
# ---------------------------------------------------------------------------

async def _setup():
    """Interactive first-time login: saves cookies.json and prints base64 for CI."""
    username = input("Twitter username (without @): ").strip()
    email = input("Twitter email: ").strip()
    password = input("Twitter password: ").strip()

    os.environ["TWITTER_USERNAME"] = username
    os.environ["TWITTER_EMAIL"] = email
    os.environ["TWITTER_PASSWORD"] = password

    # Force option 3 by removing cookies.json if it exists
    if os.path.exists("cookies.json"):
        os.remove("cookies.json")

    client = await get_authenticated_client()
    if client is None:
        print("Login failed. Check your credentials.")
        return

    print("\n✅ Login successful! cookies.json saved.")
    with open("cookies.json", "r") as f:
        cookies_b64 = base64.b64encode(f.read().encode()).decode()
    print("\n📋 Add this as a GitHub secret named TWITTER_COOKIES:")
    print(f"\n{cookies_b64}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true", help="Interactive login to generate cookies")
    args = parser.parse_args()

    if args.setup:
        asyncio.run(_setup())
    else:
        print("Use --setup to log in. This module is imported by main.py.")
