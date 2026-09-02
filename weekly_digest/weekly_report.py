"""
AI Radar — Weekly Digest
Fetches the top articles from the past 7 days in Supabase,
asks Groq to synthesize the main trends, and sends a structured
Telegram report.

Usage:
    python weekly_digest/weekly_report.py

Env vars required (same as main pipeline):
    SUPABASE_URL, SUPABASE_KEY
    GROQ_API_KEY
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import requests
from groq import Groq
from supabase import create_client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CATEGORY_EMOJI = {
    "Innovation / Tech":      "🚀",
    "Politique / Regulation": "⚖️",
    "Business / Industrie":   "💼",
    "Societe / Ethique":      "🤝",
    "Recherche Academique":   "🎓",
    "Drama / Controverses":   "💥",
}
# Legacy alias
_CATEGORY_ALIAS = {"Geopolitique": "Politique / Regulation"}

HOT_REASON_LABEL = {
    "debat":    "💬 Débat",
    "tech":     "⭐ Tech",
    "societe":  "📡 Société",
    "tendance": "🔮 Tendance",
}

GROQ_MODEL = (os.environ.get("GROQ_MODEL") or "openai/gpt-oss-120b").strip("'\"").strip()

# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def fetch_week_articles() -> list[dict]:
    """Return all articles from the past 7 days from Supabase."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        logging.error("SUPABASE_URL or SUPABASE_KEY not set.")
        sys.exit(1)

    client = create_client(url, key)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    resp = (
        client.table("articles")
        .select("title,url,source,country,published,category,sentiment,hot_topic,supa_hot,mention_count,hot_reason,summary,description")
        .gte("published", cutoff)
        .execute()
    )
    articles = resp.data or []
    # Apply category alias
    for a in articles:
        a["category"] = _CATEGORY_ALIAS.get(a.get("category", ""), a.get("category", ""))
    logging.info(f"Fetched {len(articles)} articles from the past 7 days")
    return articles

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_week_stats(articles: list[dict]) -> dict:
    """Compute aggregate stats for the weekly report."""
    total = len(articles)
    hot = [a for a in articles if a.get("hot_topic")]
    supa = [a for a in articles if a.get("supa_hot")]

    by_category: Counter = Counter(a.get("category", "?") for a in articles)
    by_sentiment: Counter = Counter(a.get("sentiment", "?") for a in articles)
    by_country: Counter = Counter(a.get("country", "?") for a in articles)

    top_cat = by_category.most_common(1)[0] if by_category else ("?", 0)

    # Top 10 hot articles sorted by mention_count desc then supa_hot
    top_hot = sorted(
        hot,
        key=lambda a: (a.get("supa_hot", False), a.get("mention_count", 0)),
        reverse=True,
    )[:10]

    return {
        "total": total,
        "hot_count": len(hot),
        "supa_count": len(supa),
        "by_category": by_category,
        "by_sentiment": by_sentiment,
        "by_country": by_country,
        "top_cat": top_cat,
        "top_hot": top_hot,
    }

# ---------------------------------------------------------------------------
# Groq synthesis
# ---------------------------------------------------------------------------

def generate_groq_synthesis(top_articles: list[dict]) -> str:
    """Ask Groq to write a qualitative weekly synthesis from the top hot articles."""
    if not top_articles:
        return "Aucun article hot cette semaine pour générer une synthèse."

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    articles_text = "\n".join(
        f"- {a['title']}"
        + (f" | {a.get('summary') or a.get('description', '')[:120]}" if (a.get('summary') or a.get('description')) else "")
        for a in top_articles
    )

    system = (
        "Tu es un analyste IA senior. On te donne les principaux articles IA de la semaine. "
        "Rédige une synthèse qualitative en 4-5 points, en français, sous forme de bullet points Markdown. "
        "Chaque point doit nommer un thème central, expliquer la tendance ou l'enjeu, et citer 1-2 exemples concrets tirés des articles. "
        "Sois direct, factuel, et sans intro ni conclusion. Commence immédiatement par le premier bullet."
    )
    user = f"Articles de la semaine :\n{articles_text}"

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Groq synthesis failed: {e}")
        return "Synthèse indisponible (erreur Groq)."

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def _post_telegram(token: str, chat_id: str, text: str) -> None:
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=10,
    )
    if not resp.ok:
        logging.error(f"Telegram error: {resp.status_code} {resp.text}")


def send_weekly_report(articles: list[dict], stats: dict, synthesis: str) -> None:
    token   = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    today    = datetime.now(timezone.utc)
    week_end = today.strftime("%d/%m/%Y")
    week_start = (today - timedelta(days=7)).strftime("%d/%m")

    # ── 1. Header & stats ──────────────────────────────────────────────────
    top_cat_name, top_cat_n = stats["top_cat"]
    top_cat_emoji = CATEGORY_EMOJI.get(top_cat_name, "📌")

    lines = [
        f"📊 <b>RADAR IA — Bilan de la semaine</b>",
        f"🗓 {week_start} → {week_end}",
        "",
        f"📰 <b>{stats['total']}</b> articles collectés",
        f"🔥 <b>{stats['hot_count']}</b> hot topics · 🌋 <b>{stats['supa_count']}</b> supra-hot",
        f"🏆 Catégorie dominante : {top_cat_emoji} {top_cat_name} ({top_cat_n})",
        "",
        "📊 <b>Répartition par catégorie :</b>",
    ]
    for cat, n in stats["by_category"].most_common():
        emoji = CATEGORY_EMOJI.get(cat, "📌")
        lines.append(f"  {emoji} {cat} : {n}")

    _post_telegram(token, chat_id, "\n".join(lines))

    # ── 2. Top hot articles ────────────────────────────────────────────────
    top_hot = stats["top_hot"]
    if top_hot:
        hot_lines = ["🔥 <b>TOP 10 SUJETS DE LA SEMAINE</b>", ""]
        for i, a in enumerate(top_hot, 1):
            badge = "🌋" if a.get("supa_hot") else "🔥"
            reason_label = HOT_REASON_LABEL.get(a.get("hot_reason", ""), "")
            blurb = (a.get("summary") or a.get("description") or "")[:120].strip()
            if blurb and not blurb.endswith((".", "!", "?")):
                blurb += "…"
            entry = f"{i}. {badge} <b>{a['title']}</b>"
            if reason_label:
                entry += f" <i>({reason_label})</i>"
            if blurb:
                entry += f"\n   <i>{blurb}</i>"
            entry += f"\n   <a href=\"{a['url']}\">🔗 Lire</a>"
            hot_lines.append(entry)

        _post_telegram(token, chat_id, "\n\n".join(hot_lines))

    # ── 3. Groq synthesis ─────────────────────────────────────────────────
    # Telegram HTML mode doesn't render Markdown bullets — convert them
    synthesis_html = synthesis.replace("**", "<b>").replace("* ", "• ")
    # Wrap bold close tags: naive approach (pairs)
    import re
    synthesis_html = re.sub(r"<b>(.*?)<b>", r"<b>\1</b>", synthesis_html)

    synth_msg = f"🤖 <b>SYNTHÈSE IA DE LA SEMAINE</b>\n\n{synthesis_html}"
    _post_telegram(token, chat_id, synth_msg)

    logging.info("Weekly report sent to Telegram (3 messages).")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    required = ["GROQ_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    missing  = [v for v in required if not os.environ.get(v)]
    if missing:
        logging.error(f"Missing env vars: {', '.join(missing)}")
        sys.exit(1)

    articles  = fetch_week_articles()
    if not articles:
        _post_telegram(
            os.environ["TELEGRAM_BOT_TOKEN"],
            os.environ["TELEGRAM_CHAT_ID"],
            "📊 Radar IA — bilan hebdomadaire : aucun article collecté cette semaine.",
        )
        return

    stats     = compute_week_stats(articles)
    synthesis = generate_groq_synthesis(stats["top_hot"])
    send_weekly_report(articles, stats, synthesis)


if __name__ == "__main__":
    main()
