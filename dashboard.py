"""
AI Radar - Dashboard de Veille
Usage local : streamlit run dashboard.py
Usage CI    : python dashboard.py --export [--days N] [--top N]
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from supabase import create_client

# Load .env (local only, no-op if absent)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "Innovation / Tech":      "#00b4d8",
    "Politique / Regulation": "#f4a261",
    "Business / Industrie":   "#2a9d8f",
    "Societe / Ethique":      "#e9c46a",
    "Recherche Academique":   "#a8dadc",
    "Drama / Controverses":   "#e63946",
}
CATEGORY_EMOJI = {
    "Innovation / Tech":      "🚀",
    "Politique / Regulation": "⚖️",
    "Business / Industrie":   "💼",
    "Societe / Ethique":      "🤝",
    "Recherche Academique":   "🎓",
    "Drama / Controverses":   "💥",
}

# Legacy category mapping (for articles already stored in Supabase)
_CATEGORY_ALIAS: dict[str, str] = {
    "Geopolitique": "Politique / Regulation",
}
SENTIMENT_COLORS = {
    "Positif": "#2a9d8f",
    "Negatif": "#e63946",
    "Neutre":  "#adb5bd",
}

PLOTLY_TEMPLATE = "plotly_dark"

# ---------------------------------------------------------------------------
# Geography — ISO-3 mappings for the 3D globe
# ---------------------------------------------------------------------------

# Emoji flags stored in sources.json → ISO-3
FLAG_TO_ISO3: dict[str, str] = {
    "🇺🇸": "USA", "🇫🇷": "FRA", "🇬🇧": "GBR", "🇩🇪": "DEU",
    "🇳🇱": "NLD", "🇳🇬": "NGA", "🇦🇪": "ARE", "🇸🇦": "SAU",
    "🇮🇳": "IND", "🇯🇵": "JPN", "🇨🇳": "CHN", "🇰🇷": "KOR",
    "🇿🇦": "ZAF", "🇨🇦": "CAN", "🇦🇺": "AUS", "🇷🇺": "RUS",
    "🇧🇷": "BRA", "🇮🇱": "ISR", "🇮🇹": "ITA", "🇪🇸": "ESP",
    "🇸🇬": "SGP", "🇹🇼": "TWN", "🇸🇪": "SWE", "🇨🇭": "CHE",
}

# Groq text country output (English & French) → ISO-3
TEXT_TO_ISO3: dict[str, str] = {
    # English
    "USA": "USA", "US": "USA", "United States": "USA", "America": "USA",
    "China": "CHN", "UK": "GBR", "United Kingdom": "GBR",
    "France": "FRA", "Germany": "DEU", "Japan": "JPN",
    "India": "IND", "Canada": "CAN", "Australia": "AUS",
    "Russia": "RUS", "Brazil": "BRA", "South Korea": "KOR",
    "Netherlands": "NLD", "UAE": "ARE", "Saudi Arabia": "SAU",
    "Nigeria": "NGA", "South Africa": "ZAF", "Israel": "ISR",
    "Italy": "ITA", "Spain": "ESP", "Sweden": "SWE",
    "Singapore": "SGP", "Taiwan": "TWN", "Hong Kong": "HKG",
    "Switzerland": "CHE", "Belgium": "BEL", "Denmark": "DNK",
    "Finland": "FIN", "Norway": "NOR", "Poland": "POL",
    "Mexico": "MEX", "Argentina": "ARG", "Chile": "CHL",
    "Indonesia": "IDN", "Malaysia": "MYS", "Thailand": "THA",
    "Vietnam": "VNM", "Philippines": "PHL", "Pakistan": "PAK",
    "Egypt": "EGY", "Morocco": "MAR", "Kenya": "KEN",
    "Ethiopia": "ETH", "Ghana": "GHA",
    # French
    "Chine": "CHN", "Japon": "JPN", "Inde": "IND",
    "Allemagne": "DEU", "Royaume-Uni": "GBR", "Russie": "RUS",
    "Corée du Sud": "KOR", "Émirats arabes unis": "ARE",
    "Afrique du Sud": "ZAF", "Brésil": "BRA",
    "Italie": "ITA", "Espagne": "ESP", "Suède": "SWE",
    "Canada": "CAN", "Australie": "AUS", "Mexique": "MEX",
    "Israël": "ISR", "Singapour": "SGP", "Indonésie": "IDN",
    "Thaïlande": "THA", "Maroc": "MAR", "Égypte": "EGY",
    "Argentine": "ARG", "Pologne": "POL", "Belgique": "BEL",
    "Pays-Bas": "NLD", "Suisse": "CHE", "Danemark": "DNK",
    "Finlande": "FIN", "Norvège": "NOR",
}

# ISO-3 → display name with flag
ISO3_TO_NAME: dict[str, str] = {
    "USA": "🇺🇸 USA", "FRA": "🇫🇷 France", "GBR": "🇬🇧 UK",
    "CHN": "🇨🇳 Chine", "DEU": "🇩🇪 Allemagne", "JPN": "🇯🇵 Japon",
    "IND": "🇮🇳 Inde", "KOR": "🇰🇷 Corée du Sud", "NLD": "🇳🇱 Pays-Bas",
    "ARE": "🇦🇪 Émirats arabes unis", "SAU": "🇸🇦 Arabie Saoudite",
    "NGA": "🇳🇬 Nigeria", "ZAF": "🇿🇦 Afrique du Sud",
    "CAN": "🇨🇦 Canada", "AUS": "🇦🇺 Australie", "RUS": "🇷🇺 Russie",
    "BRA": "🇧🇷 Brésil", "ISR": "🇮🇱 Israël", "ITA": "🇮🇹 Italie",
    "ESP": "🇪🇸 Espagne", "SWE": "🇸🇪 Suède", "SGP": "🇸🇬 Singapour",
    "TWN": "🇹🇼 Taïwan", "HKG": "🇭🇰 Hong Kong", "CHE": "🇨🇭 Suisse",
    "BEL": "🇧🇪 Belgique", "DNK": "🇩🇰 Danemark", "FIN": "🇫🇮 Finlande",
    "NOR": "🇳🇴 Norvège", "POL": "🇵🇱 Pologne", "MEX": "🇲🇽 Mexique",
    "ARG": "🇦🇷 Argentine", "IDN": "🇮🇩 Indonésie", "EGY": "🇪🇬 Égypte",
    "MAR": "🇲🇦 Maroc", "KEN": "🇰🇪 Kenya", "GHA": "🇬🇭 Ghana",
}


def _country_to_iso3(country: str) -> str | None:
    """Convert an emoji flag OR Groq text country string to ISO-3 code.
    Returns None for region-level values (Europe, Global, …)."""
    if not country:
        return None
    if country in FLAG_TO_ISO3:
        return FLAG_TO_ISO3[country]
    return TEXT_TO_ISO3.get(country)


# ---------------------------------------------------------------------------
# AI Power influence — groupings & scoring
# ---------------------------------------------------------------------------

def _compute_country_scores(articles: list[dict], top_n: int = 10) -> list[dict]:
    """Compute 3 influence scores for every country, return top N by composite.

    Couverture      = article volume × source diversity bonus
    Innovation      = count of hot articles classified hot_reason='tech' by Groq
    Influence virale = hot-article count boosted by avg mention density
    All metrics min-max normalized to 0–100 across all countries present.
    """
    by_iso: dict[str, list[dict]] = defaultdict(list)
    for a in articles:
        iso = _country_to_iso3(a.get("country", ""))
        if iso:
            by_iso[iso].append(a)

    if not by_iso:
        return []

    raw: dict[str, dict[str, float]] = {}
    for iso, arts in by_iso.items():
        n           = len(arts)
        hot_arts    = [a for a in arts if a.get("hot_topic")]
        tech_arts   = [a for a in hot_arts if a.get("hot_reason") == "tech"]
        sources     = {a.get("source") for a in arts if a.get("source")}
        avg_mention = sum(a.get("mention_count", 0) for a in arts) / n if n else 0
        raw[iso] = {
            "couverture": n * (1 + 0.3 * (len(sources) ** 0.5)),
            "innovation": float(len(tech_arts)),
            "virale":     len(hot_arts) * (1 + avg_mention * 0.3),
        }

    # Min-max normalize each metric to 0–100 across all countries
    normed: dict[str, dict[str, float]] = {iso: {} for iso in raw}
    for metric in ("couverture", "innovation", "virale"):
        vals  = [raw[iso][metric] for iso in raw]
        max_v = max(vals) if max(vals) > 0 else 1
        for iso in raw:
            normed[iso][metric] = round(raw[iso][metric] / max_v * 100, 1)

    result = []
    for iso, s in normed.items():
        composite = round((s["couverture"] + s["innovation"] + s["virale"]) / 3, 1)
        result.append({
            "iso":        iso,
            "name":       ISO3_TO_NAME.get(iso, iso),
            "couverture": s["couverture"],
            "innovation": s["innovation"],
            "virale":     s["virale"],
            "composite":  composite,
        })

    result.sort(key=lambda x: x["composite"], reverse=True)
    return result[:top_n]


# Also extend COUNTRY_TO_REGION to handle Groq text values ──────────────────
_GROQ_REGION_MAP: dict[str, str] = {
    "USA": "Amérique du Nord", "US": "Amérique du Nord", "United States": "Amérique du Nord",
    "Canada": "Amérique du Nord",
    "Brazil": "Amérique du Sud", "Brésil": "Amérique du Sud",
    "Argentina": "Amérique du Sud", "Mexico": "Amérique du Nord", "Mexique": "Amérique du Nord",
    "France": "Europe", "Germany": "Europe", "Allemagne": "Europe",
    "UK": "Europe", "United Kingdom": "Europe", "Royaume-Uni": "Europe",
    "Netherlands": "Europe", "Pays-Bas": "Europe", "Italy": "Europe", "Italie": "Europe",
    "Spain": "Europe", "Espagne": "Europe", "Sweden": "Europe", "Suède": "Europe",
    "Belgium": "Europe", "Belgique": "Europe", "Switzerland": "Europe", "Suisse": "Europe",
    "Denmark": "Europe", "Danemark": "Europe", "Finland": "Europe", "Finlande": "Europe",
    "Norway": "Europe", "Norvège": "Europe", "Poland": "Europe", "Pologne": "Europe",
    "Russia": "Europe", "Russie": "Europe",
    "China": "Asie de l'Est", "Chine": "Asie de l'Est",
    "Japan": "Asie de l'Est", "Japon": "Asie de l'Est",
    "South Korea": "Asie de l'Est", "Corée du Sud": "Asie de l'Est",
    "Taiwan": "Asie de l'Est", "Taïwan": "Asie de l'Est",
    "Hong Kong": "Asie de l'Est",
    "India": "Asie du Sud", "Inde": "Asie du Sud", "Pakistan": "Asie du Sud",
    "Singapore": "Asie du Sud-Est", "Singapour": "Asie du Sud-Est",
    "Indonesia": "Asie du Sud-Est", "Indonésie": "Asie du Sud-Est",
    "Malaysia": "Asie du Sud-Est", "Thailand": "Asie du Sud-Est",
    "Vietnam": "Asie du Sud-Est",
    "Australia": "Asie du Sud-Est", "Australie": "Asie du Sud-Est",
    "UAE": "Moyen-Orient", "Émirats arabes unis": "Moyen-Orient",
    "Saudi Arabia": "Moyen-Orient", "Arabie Saoudite": "Moyen-Orient",
    "Israel": "Moyen-Orient", "Israël": "Moyen-Orient", "Egypt": "Moyen-Orient",
    "Nigeria": "Afrique", "South Africa": "Afrique", "Afrique du Sud": "Afrique",
    "Kenya": "Afrique", "Ghana": "Afrique", "Morocco": "Afrique", "Maroc": "Afrique",
    "Ethiopia": "Afrique",
    "Global": "International", "International": "International", "Europe": "Europe",
    "Asia": "Asie du Sud-Est", "Asie": "Asie du Sud-Est",
}

# Hot reason groups — keyed by Groq's hot_reason values, order = tab display order
HOT_SOURCE_META: list[dict] = [
    {"key": "debat",    "label": "Sujets en débat",     "icon": "💬", "color": "#ff6600", "border": "#ff6600"},
    {"key": "tech",     "label": "Tech viral",          "icon": "⭐", "color": "#238636", "border": "#3fb950"},
    {"key": "societe",  "label": "Sujets de société",   "icon": "📡", "color": "#0e7490", "border": "#06b6d4"},
    {"key": "tendance", "label": "Tendances montantes", "icon": "🔮", "color": "#7b2ff7", "border": "#9d4edd"},
    {"key": "unknown",  "label": "Hot topics",          "icon": "🔥", "color": "#92400e", "border": "#f4a261"},
]
_SOURCE_ORDER = {m["key"]: i for i, m in enumerate(HOT_SOURCE_META)}

# Mapping emoji-flag → world region (covers all sources in sources.json)
COUNTRY_TO_REGION: dict[str, str] = {
    "🇺🇸": "Amérique du Nord",
    "🌎": "Amérique du Sud",
    "🇫🇷": "Europe",
    "🇬🇧": "Europe",
    "🇩🇪": "Europe",
    "🇳🇱": "Europe",
    "🇳🇬": "Afrique",
    "🌍": "Afrique",
    "🇦🇪": "Moyen-Orient",
    "🇸🇦": "Moyen-Orient",
    "🇮🇳": "Asie du Sud",
    "🇯🇵": "Asie de l'Est",
    "🇨🇳": "Asie de l'Est",
    "🇰🇷": "Asie de l'Est",
    "🌏": "Asie du Sud-Est",
    "🌐": "International",
}

def _region(country: str) -> str:
    return (
        COUNTRY_TO_REGION.get(country)
        or _GROQ_REGION_MAP.get(country)
        or "Autre"
    )

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL ou SUPABASE_KEY manquant.")
    return create_client(url, key)


def _normalize_date(s: str) -> str:
    """Keep only the YYYY-MM-DD part of any ISO date string."""
    return s[:10] if s else s


def _time_ago(published_str: str) -> str:
    """Convert a published datetime string to a human-readable elapsed time."""
    if not published_str:
        return "—"
    try:
        s = published_str.replace("Z", "+00:00")
        try:
            pub = datetime.fromisoformat(s)
        except ValueError:
            pub = datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        secs = int((now - pub).total_seconds())
        if secs < 0:
            return "—"
        if secs < 3600:
            m = max(1, secs // 60)
            return f"{m}min"
        if secs < 86400:
            return f"{secs // 3600}h"
        d = secs // 86400
        if d == 1:
            return "hier"
        return f"{d}j"
    except Exception:
        return published_str[:10]


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "as", "by", "from", "is", "are", "was", "were", "be",
    "been", "have", "has", "had", "will", "would", "could", "should",
    "that", "this", "these", "those", "its", "not", "new", "how", "why",
    "what", "when", "where", "who", "which", "more", "can", "all", "out",
    "over", "about", "into", "than", "their", "they", "there", "says",
    "said", "just", "also", "after", "amid", "here", "your", "our",
    "le", "la", "les", "de", "du", "des", "et", "en", "un", "une",
    "sur", "par", "pour", "avec", "dans", "est", "sont",
}


def _tokenize(title: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]{4,}", title.lower())
    return {w for w in words if w not in _STOPWORDS}


def _compute_mention_counts(articles: list[dict]) -> dict[str, int]:
    """For each article, count how many other articles today share ≥ 2 title keywords."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tokens = {a["url"]: _tokenize(a.get("title", "")) for a in articles}
    counts: dict[str, int] = {}
    for i, a in enumerate(articles):
        url = a["url"]
        mine = tokens[url]
        if not mine:
            counts[url] = 0
            continue
        counts[url] = sum(
            1 for j, other in enumerate(articles)
            if i != j
            and other.get("published", "") == today
            and len(mine & tokens[other["url"]]) >= 2
        )
    return counts


def _deduplicate_articles(articles: list[dict]) -> list[dict]:
    """Merge articles sharing ≥ 2 title keywords into one card with source_count."""
    tokens = {a["url"]: _tokenize(a.get("title", "")) for a in articles}
    used: set[str] = set()
    result = []
    for i, a in enumerate(articles):
        url = a["url"]
        if url in used:
            continue
        mine = tokens[url]
        group = [a]
        if mine:
            for j, other in enumerate(articles):
                if i == j or other["url"] in used:
                    continue
                if len(mine & tokens[other["url"]]) >= 2:
                    group.append(other)
                    used.add(other["url"])
        used.add(url)
        rep = dict(group[0])
        rep["source_count"]    = len(group)
        rep["merged_sources"]  = [g["source"] for g in group]
        result.append(rep)
    return result


def _primary_hot_source(a: dict) -> str:
    """Return the hot tab key for an article.
    Prefers Groq's semantic hot_reason; falls back to unknown."""
    reason = (a.get("hot_reason") or "").strip()
    if reason in _SOURCE_ORDER:
        return reason
    return "unknown"


def _hot_sort_key(a: dict):
    """Sort key: supa_hot first, then newest, then most mentioned."""
    supa = 0 if a.get("supa_hot") else 1
    raw = (a.get("published_raw") or a.get("published", ""))[:19]
    try:
        ts = -datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        ts = 0.0
    return (supa, ts, -a.get("mention_count", 0))


def load_articles(days: int) -> list[dict]:
    client = _supabase_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    resp = (
        client.table("articles")
        .select("title, source, country, published, category, sentiment, url, hot_topic, hot_source, hot_reason, summary, description")
        .gte("published", cutoff)
        .order("hot_topic", desc=True)
        .order("published", desc=True)
        .execute()
    )
    articles = resp.data or []
    for a in articles:
        if a.get("published"):
            a["published_raw"] = a["published"]
            a["published"] = _normalize_date(a["published"])
        # Migrate legacy categories
        if a.get("category") in _CATEGORY_ALIAS:
            a["category"] = _CATEGORY_ALIAS[a["category"]]

    # Compute mention counts and supa_hot flag
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    mention_counts = _compute_mention_counts(articles)
    for a in articles:
        a["mention_count"] = mention_counts.get(a.get("url", ""), 0)
        a["supa_hot"] = (
            bool(a.get("hot_topic"))
            and a["mention_count"] > 5
            and a.get("published", "") == today
        )
    return articles

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_globe(articles: list[dict]) -> go.Figure:
    """3D orthographic choropleth globe.
    Countries are colored by article count; clicking one filters the feed."""
    iso_counts: Counter = Counter()
    for a in articles:
        iso = _country_to_iso3(a.get("country", ""))
        if iso:
            iso_counts[iso] += 1

    if not iso_counts:
        fig = go.Figure()
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=460,
            title="Aucune donnée géographique disponible",
            geo=dict(
                showframe=False, showland=True, landcolor="#1a1d27",
                showocean=True, oceancolor="#0e1117",
                projection_type="orthographic", bgcolor="#0e1117",
            ),
            paper_bgcolor="#0e1117",
        )
        return fig

    iso_list  = list(iso_counts.keys())
    counts    = [iso_counts[c] for c in iso_list]
    max_count = max(counts)

    hover_text = [
        f"<b>{ISO3_TO_NAME.get(iso, iso)}</b><br>{cnt} article{'s' if cnt > 1 else ''}"
        for iso, cnt in zip(iso_list, counts)
    ]

    # Gradient: dark-blue (1 article) → cyan (brand) → orange → red (max)
    colorscale = [
        [0.0,  "#0d3b5e"],
        [0.15, "#0077b6"],
        [0.40, "#00b4d8"],
        [0.70, "#f4a261"],
        [1.0,  "#e63946"],
    ]

    fig = go.Figure(go.Choropleth(
        locations=iso_list,
        z=counts,
        locationmode="ISO-3",
        colorscale=colorscale,
        zmin=1,
        zmax=max_count,
        showscale=True,
        colorbar=dict(
            title=dict(text="Articles", font=dict(color="#adb5bd", size=12)),
            tickfont=dict(color="#adb5bd"),
            thickness=14,
            len=0.55,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
        ),
        hovertext=hover_text,
        hoverinfo="text",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text="🌍 Couverture mondiale  ·  cliquez sur un pays pour filtrer",
            font=dict(size=14, color="#adb5bd"),
            x=0.5, xanchor="center",
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#2a2d3a",
            showland=True,
            landcolor="#1a1d27",
            showocean=True,
            oceancolor="#0e1117",
            showlakes=False,
            lakecolor="#0e1117",
            showcountries=True,
            countrycolor="#2a2d3a",
            projection_type="orthographic",
            bgcolor="#0e1117",
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=0, r=0, t=50, b=0),
        height=460,
        dragmode="pan",
    )
    return fig


def fig_country_ranking(articles: list[dict]) -> go.Figure:
    """Horizontal grouped bar chart: top 10 countries ranked by composite influence score."""
    ranking = _compute_country_scores(articles, top_n=10)

    if not ranking:
        fig = go.Figure()
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=400, paper_bgcolor="#0e1117",
            title=dict(text="🏆 Top 10 pays — aucune donnée", font=dict(color="#adb5bd")),
        )
        return fig

    # Reverse so #1 appears at top of horizontal bar chart
    ranking_rev = list(reversed(ranking))
    names = [r["name"] for r in ranking_rev]

    metrics = [
        ("couverture", "📰 Couverture",       "#00b4d8"),
        ("innovation", "🚀 Innovation",        "#f4a261"),
        ("virale",     "🔥 Influence virale",  "#e63946"),
    ]
    fig = go.Figure()
    for key, label, color in metrics:
        values = [r[key] for r in ranking_rev]
        fig.add_trace(go.Bar(
            y=names,
            x=values,
            name=label,
            orientation="h",
            marker=dict(color=color, opacity=0.85),
            hovertemplate=f"<b>%{{y}}</b><br>{label} : <b>%{{x:.0f}}</b> / 100<extra></extra>",
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="group",
        title=dict(
            text="🏆 Top 10 pays — Score d'influence IA",
            font=dict(size=14, color="#adb5bd"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="Score / 100", range=[0, 108],
            gridcolor="#2a2d3a", tickfont=dict(size=10),
        ),
        yaxis=dict(tickfont=dict(size=11, color="#fafafa"), automargin=True),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.04,
            xanchor="center", x=0.5,
            font=dict(size=10, color="#adb5bd"),
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=10, r=20, t=50, b=55),
        height=450,
    )
    return fig


def _render_hot_articles(articles: list[dict], container) -> None:
    """Render hot articles as tabs grouped by detection source in Streamlit."""
    import streamlit as st
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    hot = _deduplicate_articles([
        a for a in articles if a.get("hot_topic") and a.get("published", "") >= week_ago
    ])
    container.markdown("#### 🔥 Hot Articles")
    if not hot:
        container.info("Aucun article hot topic sur la période sélectionnée.")
        return

    named_meta = [m for m in HOT_SOURCE_META if m["key"] != "unknown"]
    groups: dict[str, list[dict]] = {m["key"]: [] for m in HOT_SOURCE_META}
    for a in hot:
        src = _primary_hot_source(a)
        groups[src if src != "unknown" else "debat"].append(a)

    tab_labels = [
        f"{m['icon']} {m['label']} ({len(groups[m['key']])})"
        for m in named_meta
    ]
    tabs = container.tabs(tab_labels)
    for tab, meta in zip(tabs, named_meta):
        group = sorted(groups[meta["key"]], key=_hot_sort_key)
        if group:
            cards_html = "".join(_render_hot_card_html(a, meta) for a in group)
            tab.markdown(cards_html, unsafe_allow_html=True)
        else:
            tab.info(f"Aucun article {meta['label'].lower()} sur la période sélectionnée.")



# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def _is_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(
        layout="wide",
        page_title="Cobalt.xyz · AI Radar",
        page_icon="🤖",
    )

    @st.cache_data(ttl=300)
    def _cached_load(days: int) -> list[dict]:
        return load_articles(days)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🤖 Cobalt.xyz · AI Radar")
        st.caption("L'actualité IA mondiale : tendances, innovations et influences.")
        st.markdown("---")

        # Période rapide — preset buttons update the slider via session_state
        if "days" not in st.session_state:
            st.session_state["days"] = 7
        st.caption("Période rapide")
        p1, p2, p3 = st.columns(3)
        if p1.button("1j", use_container_width=True): st.session_state["days"] = 1
        if p2.button("3j", use_container_width=True): st.session_state["days"] = 3
        if p3.button("7j", use_container_width=True): st.session_state["days"] = 7

        days = st.slider("Fenêtre d'analyse (jours)", 1, 7, key="days")

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Chargement des données..."):
        try:
            all_articles = _cached_load(days * 2)
        except RuntimeError as e:
            st.error(str(e))
            return

    # Split current / previous period for delta KPIs
    cutoff_str    = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    articles      = [a for a in all_articles if a["published"] >= cutoff_str]
    prev_articles = [a for a in all_articles if a["published"] <  cutoff_str]

    if not articles:
        st.warning(f"Aucun article trouvé sur les {days} derniers jours.")
        return

    # ── Sidebar — global filters (data-driven) ────────────────────────────────
    with st.sidebar:
        st.markdown("---")

        all_cats     = sorted({a["category"] for a in articles if a.get("category")})
        all_sents    = ["Positif", "Neutre", "Negatif"]
        all_regions  = sorted({_region(a["country"]) for a in articles if a.get("country")})
        all_countries= sorted({a["country"] for a in articles if a.get("country")})

        sel_cats = st.multiselect(
            "Catégories", all_cats, default=all_cats,
            format_func=lambda c: f"{CATEGORY_EMOJI.get(c, '📌')} {c}",
        )
        sel_sents = st.multiselect(
            "Sentiment", all_sents, default=all_sents,
            format_func=lambda s: {"Positif": "🟢 Positif", "Neutre": "⚪ Neutre", "Negatif": "🔴 Négatif"}[s],
        )
        sel_regions = st.multiselect("Région du monde", all_regions, default=all_regions)
        sel_countries = st.multiselect("Pays", all_countries, default=all_countries)

        st.markdown("---")
        if st.button("🔄 Rafraîchir", use_container_width=True):
            _cached_load.clear()
            st.rerun()
        st.caption("Données : Supabase · Classif : Groq")

    # ── Apply global filters to both periods ──────────────────────────────────
    def _apply_global(pool: list[dict]) -> list[dict]:
        return [
            a for a in pool
            if  a.get("category")  in sel_cats
            and a.get("sentiment") in sel_sents
            and _region(a.get("country", "")) in sel_regions
            and a.get("country")   in sel_countries
        ]

    filtered      = _apply_global(articles)
    prev_filtered = _apply_global(prev_articles)

    if not filtered:
        st.warning("Aucun article ne correspond aux filtres sélectionnés.")
        return

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total      = len(filtered)
    prev_total = len(prev_filtered)
    pos_pct    = round(sum(1 for a in filtered if a["sentiment"] == "Positif") / total * 100)
    neg_pct    = round(sum(1 for a in filtered if a["sentiment"] == "Negatif") / total * 100)
    nb_src     = len({a["source"] for a in filtered})
    top_cat    = Counter(a["category"] for a in filtered).most_common(1)[0][0]
    dates      = sorted({a["published"] for a in filtered})
    date_lbl   = f"{dates[0]} → {dates[-1]}" if dates else "—"

    prev_pos_pct = round(sum(1 for a in prev_filtered if a["sentiment"] == "Positif") / max(prev_total, 1) * 100)
    prev_neg_pct = round(sum(1 for a in prev_filtered if a["sentiment"] == "Negatif") / max(prev_total, 1) * 100)
    prev_nb_src  = len({a["source"] for a in prev_filtered})

    st.markdown(f"## 📰 {total} articles · {date_lbl}")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total articles",      total,          delta=total - prev_total)
    k2.metric("Sentiment positif",   f"{pos_pct} %", delta=f"{pos_pct - prev_pos_pct} pts")
    k3.metric("Sentiment négatif",   f"{neg_pct} %", delta=f"{neg_pct - prev_neg_pct} pts", delta_color="inverse")
    k4.metric("Sources actives",     nb_src,         delta=nb_src - prev_nb_src)
    k5.metric("Catégorie dominante", f"{CATEGORY_EMOJI.get(top_cat, '📌')} {top_cat.split('/')[0].strip()}")

    st.markdown("---")

    # ── Charts : Globe 3D + Radar d'influence ────────────────────────────────
    if "globe_country" not in st.session_state:
        st.session_state["globe_country"] = None

    col_globe, col_ranking = st.columns([3, 2])
    with col_ranking:
        st.plotly_chart(fig_country_ranking(filtered), use_container_width=True)
    with col_globe:
        globe_event = st.plotly_chart(
            fig_globe(filtered),
            use_container_width=True,
            on_select="rerun",
            key="globe_chart",
            selection_mode="points",
        )
    # Capture click → toggle country filter
    if globe_event and globe_event.selection and globe_event.selection.points:
        clicked_iso = globe_event.selection.points[0].get("location")
        if clicked_iso:
            if st.session_state["globe_country"] == clicked_iso:
                st.session_state["globe_country"] = None   # second click = deselect
            else:
                st.session_state["globe_country"] = clicked_iso

    selected_iso = st.session_state["globe_country"]

    # Country filter indicator + clear button
    if selected_iso:
        country_name   = ISO3_TO_NAME.get(selected_iso, selected_iso)
        country_count  = sum(1 for a in filtered if _country_to_iso3(a.get("country", "")) == selected_iso)
        col_info, col_clear = st.columns([5, 1])
        col_info.info(f"🌍 Filtré par pays : **{country_name}** — {country_count} article{'s' if country_count != 1 else ''}")
        if col_clear.button("✕ Effacer", use_container_width=True):
            st.session_state["globe_country"] = None
            st.rerun()
        display_articles = [a for a in filtered if _country_to_iso3(a.get("country", "")) == selected_iso]
    else:
        display_articles = filtered

    _render_hot_articles(display_articles, st)

    # ── Table — local filters ─────────────────────────────────────────────────
    display_articles = _deduplicate_articles(display_articles)
    st.markdown("### 📋 Derniers articles")

    fa, fb, fc, fd = st.columns([3, 2, 1, 1])
    search   = fa.text_input("🔍 Recherche dans les titres", placeholder="ex: GPT, OpenAI, Mistral…")
    all_srcs = sorted({a["source"] for a in display_articles})
    sel_srcs = fb.multiselect("Source", all_srcs, default=all_srcs, label_visibility="visible")
    only_new = fc.checkbox("Nouveautés 24h", value=False)
    only_top = fd.checkbox("Catégorie dominante", value=False)

    # Apply local filters to the display dataframe only
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    table_rows = [
        a for a in display_articles
        if  (not search   or search.lower() in a.get("title", "").lower())
        and (a["source"]  in sel_srcs)
        and (not only_new or a["published"] >= today_str)
        and (not only_top or a.get("category") == top_cat)
    ]

    if not table_rows:
        st.info("Aucun article ne correspond aux filtres du tableau.")
    else:
        df = pd.DataFrame(table_rows)
        df["category"]  = df["category"].apply(lambda c: f"{CATEGORY_EMOJI.get(c, '📌')} {c}")
        df["lien"]      = df["url"].apply(lambda u: f"[↗]({u})")
        df["hot_topic"] = df.get("hot_topic", False).fillna(False)
        df["age"] = df.apply(
            lambda r: _time_ago(r.get("published_raw") or r.get("published", "")), axis=1
        )
        # Hot topic articles are already first (ordered by Supabase); add 🔥 badge in title
        df["title"] = df.apply(
            lambda r: f"🔥 {r['title']}" if r["hot_topic"] else r["title"], axis=1
        )
        st.dataframe(
            df[["age", "sentiment", "title", "source", "country", "category", "lien"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "age":       st.column_config.TextColumn("Publié"),
                "sentiment": st.column_config.TextColumn("Sent."),
                "title":     st.column_config.TextColumn("Titre", width="large"),
                "source":    st.column_config.TextColumn("Source"),
                "country":   st.column_config.TextColumn(""),
                "category":  st.column_config.TextColumn("Catégorie"),
                "lien":      st.column_config.LinkColumn("Lien", display_text="↗", width="small"),
            },
        )
        nb_hot = int(df["hot_topic"].sum())
        st.caption(f"{len(table_rows)} articles affichés · {nb_hot} 🔥 hot topics")

# ---------------------------------------------------------------------------
# HTML export (CI mode)
# ---------------------------------------------------------------------------

def _articles_to_html_table(articles: list[dict]) -> str:
    """Build HTML table rows with data-* attributes for JS filtering."""
    rows = []
    for a in articles:
        cat   = a.get("category", "")
        emoji = CATEGORY_EMOJI.get(cat, "📌")
        title = a.get("title", "").replace("<", "&lt;").replace(">", "&gt;")
        url   = a.get("url", "#")
        sent  = a.get("sentiment", "")
        src   = a.get("source", "")
        hot      = a.get("hot_topic", False)
        hot_badge = "🔥 " if hot else ""
        hot_class = ' class="hot"' if hot else ""
        age    = _time_ago(a.get("published_raw") or a.get("published", ""))
        n_src  = a.get("source_count", 1)
        src_display = src if n_src == 1 else f"{src} <small style='color:#888'>+{n_src-1}</small>"
        iso = _country_to_iso3(a.get("country", "")) or ""
        rows.append(
            f'<tr data-title="{title.lower()}" data-sentiment="{sent}" '
            f'data-category="{cat}" data-source="{src}" data-iso="{iso}"{hot_class}>'
            f"<td>{age}</td>"
            f"<td>{sent}</td>"
            f'<td><a href="{url}" target="_blank">{hot_badge}{title}</a></td>'
            f'<td class="col-source">{src_display}</td>'
            f'<td class="col-country">{a.get("country", "")}</td>'
            f"<td>{emoji} {cat}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _render_hot_card_html(a: dict, meta: dict) -> str:
    """Build one hot article card HTML for a given source group."""
    cat_emoji  = CATEGORY_EMOJI.get(a.get("category", ""), "📌")
    sent_color = SENTIMENT_COLORS.get(a.get("sentiment", ""), "#adb5bd")
    title      = a.get("title", "").replace("<", "&lt;").replace(">", "&gt;")
    mentions   = a.get("mention_count", 0)
    is_supra   = a.get("supa_hot", False)
    n_src      = a.get("source_count", 1)
    src_label  = ", ".join(a.get("merged_sources", [a.get("source", "")])) if n_src > 1 else a.get("source", "")
    src_badge  = (f'<span style="background:#333;color:#ccc;font-size:10px;padding:1px 5px;'
                  f'border-radius:3px;margin-left:4px;">{n_src} sources</span>') if n_src > 1 else ""
    if is_supra:
        card_style  = ("background:linear-gradient(135deg,#2a0a00,#1a0d00);"
                       "border-left:4px solid #ff4500;"
                       "box-shadow:0 0 12px rgba(255,69,0,0.4);")
        title_color = "#ff6b35"
        badge_html  = (f'<span style="background:#ff4500;color:#fff;font-size:10px;font-weight:700;'
                       f'padding:2px 6px;border-radius:4px;margin-right:6px;">🌋 SUPA HOT · {mentions} mentions</span>')
    else:
        border      = meta["border"]
        card_style  = f"background:#1a1d27;border-left:4px solid {border};"
        title_color = "#fafafa"
        badge_html  = ""
    iso = _country_to_iso3(a.get("country", "")) or ""
    return f"""
    <div data-iso="{iso}" style="border-radius:8px;padding:12px 14px;margin-bottom:8px;{card_style}">
      <div style="font-size:15px;font-weight:600;margin-bottom:6px;line-height:1.4;">
        {badge_html}<a href="{a.get('url','#')}" target="_blank"
           style="color:{title_color};text-decoration:none;">{title}</a>
      </div>
      <div style="font-size:12px;color:#888;line-height:1.6;">
        {a.get('country','')} {src_label}{src_badge} &nbsp;·&nbsp; {_time_ago(a.get('published_raw') or a.get('published',''))}
        &nbsp;·&nbsp; {cat_emoji} {a.get('category','')}
        &nbsp;·&nbsp; <span style="color:{sent_color};">{a.get('sentiment','')}</span>
      </div>
    </div>"""


def _hot_articles_html(articles: list[dict]) -> str:
    """Build hot articles as tabs grouped by detection source (CI HTML export).
    All 4 named tabs are always rendered; empty ones are disabled."""
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    hot = _deduplicate_articles([
        a for a in articles if a.get("hot_topic") and a.get("published", "") >= week_ago
    ])
    if not hot:
        return "<p style='color:#888;'>Aucun article hot topic sur la période.</p>"

    # Named tabs only (skip "unknown" from display tabs — fold into first non-empty named tab)
    named_meta = [m for m in HOT_SOURCE_META if m["key"] != "unknown"]
    groups: dict[str, list[dict]] = {m["key"]: [] for m in HOT_SOURCE_META}
    for a in hot:
        src = _primary_hot_source(a)
        groups[src if src != "unknown" else "debat"].append(a)

    # First tab with content (default selected); fall back to first named tab
    first_key = next((m["key"] for m in named_meta if groups[m["key"]]), named_meta[0]["key"])

    # ── Tab buttons — always all 4 ────────────────────────────────────────────
    buttons = ""
    for m in named_meta:
        count     = len(groups[m["key"]])
        is_active = m["key"] == first_key
        is_empty  = count == 0
        if is_empty:
            btn_style = "opacity:0.35;cursor:default;"
            onclick   = 'onclick="return false"'
        else:
            btn_style = f"border-color:{m['border']};color:{m['color']};" if is_active else ""
            onclick   = ""
        active_cls = " hot-tab--active" if is_active and not is_empty else ""
        buttons += (
            f'<button class="hot-tab{active_cls}" data-group="{m["key"]}" '
            f'style="{btn_style}" {onclick}>'
            f'{m["icon"]} {m["label"]}'
            f'<span class="hot-tab__count">{count}</span>'
            f'</button>'
        )

    # ── Tab panels ────────────────────────────────────────────────────────────
    panels = ""
    for m in named_meta:
        group   = sorted(groups[m["key"]], key=_hot_sort_key)
        display = "block" if m["key"] == first_key else "none"
        if group:
            cards = "".join(_render_hot_card_html(a, m) for a in group)
        else:
            cards = f'<p style="color:#888;padding:12px 0;">Aucun article {m["label"].lower()} sur la période.</p>'
        panels += f'<div class="hot-panel" id="hot-{m["key"]}" style="display:{display}">{cards}</div>'

    return f"""
<style>
  .hot-tabs {{
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px;
  }}
  .hot-tab {{
    background: #1a1d27; color: #adb5bd;
    border: 2px solid #2a2d3a; border-radius: 20px;
    padding: 8px 18px; font-size: 13px; cursor: pointer;
    transition: all 0.15s ease; white-space: nowrap;
  }}
  .hot-tab:not([onclick]):hover {{ background: #252836; color: #fafafa; }}
  .hot-tab--active {{ background: #252836; font-weight: 700; }}
  .hot-tab__count {{
    background: rgba(255,255,255,0.12); border-radius: 10px;
    padding: 1px 7px; font-size: 11px; margin-left: 6px; font-weight: 400;
  }}
</style>
<div class="hot-tabs">{buttons}</div>
<div>{panels}</div>
<script>
(function() {{
  var COLOR = {{
    debat:    {{ border:'#ff6600', color:'#ff6600' }},
    tech:     {{ border:'#3fb950', color:'#238636' }},
    societe:  {{ border:'#06b6d4', color:'#0e7490' }},
    tendance: {{ border:'#9d4edd', color:'#7b2ff7' }},
  }};
  document.querySelectorAll('.hot-tab:not([onclick])').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      document.querySelectorAll('.hot-tab').forEach(function(b) {{
        b.classList.remove('hot-tab--active');
        b.style.borderColor = '#2a2d3a';
        b.style.color = '';
      }});
      document.querySelectorAll('.hot-panel').forEach(function(p) {{
        p.style.display = 'none';
      }});
      btn.classList.add('hot-tab--active');
      var g = btn.dataset.group;
      if (COLOR[g]) {{ btn.style.borderColor = COLOR[g].border; btn.style.color = COLOR[g].color; }}
      var panel = document.getElementById('hot-' + g);
      if (panel) panel.style.display = 'block';
    }});
  }});
}})();
</script>"""


def run_export(days: int, output: str = "dashboard.html") -> None:
    print(f"Chargement des articles ({days} derniers jours)...")
    try:
        articles = load_articles(days)
    except RuntimeError as e:
        print(f"Erreur : {e}", file=sys.stderr)
        sys.exit(1)

    if not articles:
        print("Aucun article trouvé.")
        sys.exit(0)

    total    = len(articles)
    dates    = sorted({a["published"] for a in articles})
    date_lbl = f"{dates[0]} → {dates[-1]}" if dates else "—"
    print(f"{total} articles trouvés ({date_lbl})")

    globe_html = pio.to_html(
        fig_globe(articles),
        div_id="globe-div",
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True, "scrollZoom": False},
    )
    radar_html = pio.to_html(
        fig_country_ranking(articles),
        div_id="radar-div",
        full_html=False,
        include_plotlyjs=False,   # already bundled by globe_html
        config={"responsive": True},
    )

    deduped    = _deduplicate_articles(articles)
    hot_cards  = _hot_articles_html(deduped)
    table_rows = _articles_to_html_table(deduped)

    # Build filter option lists for the HTML selects
    def _options(values: list[str]) -> str:
        return "\n".join(f'<option value="{v}">{v}</option>' for v in sorted(set(values)))

    opt_sent = _options([a.get("sentiment", "") for a in articles if a.get("sentiment")])
    opt_cat  = _options([a.get("category",  "") for a in articles if a.get("category")])
    opt_src  = _options([a.get("source",    "") for a in articles if a.get("source")])

    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    full_html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#0e1117">
  <title>Cobalt.xyz · AI Radar</title>
  <style>
    /* ── Reset & base ───────────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      background: #0e1117; color: #fafafa;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 0; padding: 12px 12px env(safe-area-inset-bottom, 12px);
      -webkit-text-size-adjust: 100%;
    }}
    h1 {{ color: #00b4d8; margin: 0 0 4px; font-size: 1.25rem; line-height: 1.3; }}
    h2 {{ color: #fafafa; margin: 28px 0 12px; font-size: 1.05rem; }}
    p  {{ color: #888; margin: 0 0 16px; font-size: 0.82rem; }}
    a  {{ color: #00b4d8; text-decoration: none; }}
    a:active {{ opacity: 0.7; }}

    /* ── Charts grid (globe + radar) ────────────────────── */
    .charts-row {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      margin-bottom: 10px;
    }}
    @media (min-width: 640px) {{
      .charts-row {{ grid-template-columns: 3fr 2fr; }}
    }}
    .globe-card, .radar-card {{
      background: #0e1117; border-radius: 10px;
      overflow: hidden; position: relative;
    }}
    #country-filter-bar {{
      display: none; align-items: center; gap: 10px;
      background: #1a1d27; border-radius: 8px;
      padding: 10px 14px; margin-bottom: 10px; font-size: 13px;
    }}
    #country-filter-bar.visible {{ display: flex; }}
    #country-filter-label {{ color: #00b4d8; flex: 1; }}
    #country-filter-clear {{
      background: #2a2d3a; color: #adb5bd; border: none;
      border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 12px;
    }}
    #country-filter-clear:hover {{ background: #e63946; color: #fff; }}

    /* ── Hot articles ────────────────────────────────────── */
    .hot-card {{
      background: #1a1d27; border-left: 4px solid #f4a261;
      border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;
    }}
    .hot-title {{ font-size: 15px; font-weight: 600; margin-bottom: 6px; line-height: 1.4; }}
    .hot-title a {{ color: #fafafa; }}
    .hot-meta   {{ font-size: 12px; color: #888; line-height: 1.6; }}

    /* ── Toolbar — stacks vertically on mobile ───────────── */
    .toolbar {{
      display: flex; flex-direction: column; gap: 8px;
      background: #1a1d27; padding: 12px; border-radius: 10px; margin-bottom: 10px;
    }}
    .toolbar input, .toolbar select {{
      width: 100%; background: #0e1117; color: #fafafa;
      border: 1px solid #2a2d3a; border-radius: 8px;
      padding: 11px 12px; font-size: 15px; outline: none;
      -webkit-appearance: none; appearance: none;
    }}
    .toolbar input::placeholder {{ color: #555; }}
    #count {{ color: #888; font-size: 12px; text-align: right; }}

    /* ── Table — scrolls horizontally on small screens ───── */
    .table-wrap {{
      overflow-x: auto; -webkit-overflow-scrolling: touch;
      max-height: 70vh; overflow-y: auto;
      border-radius: 10px; background: #0e1117;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; min-width: 460px; }}
    th {{
      background: #1a1d27; color: #00b4d8; text-align: left;
      padding: 10px 10px; position: sticky; top: 0; z-index: 1; white-space: nowrap;
    }}
    td {{ padding: 9px 10px; border-bottom: 1px solid #2a2d3a; vertical-align: top; }}
    td:first-child {{ white-space: nowrap; }}
    tr:active td {{ background: #1a1d27; }}
    tr.hot td {{ background: #1f1a10; border-left: 3px solid #f4a261; }}

    /* Hide country & source columns on mobile to save space */
    .col-country, .col-source {{ display: none; }}

    /* ── Desktop overrides (≥ 640px) ─────────────────────── */
    @media (min-width: 640px) {{
      body  {{ padding: 20px; }}
      h1    {{ font-size: 1.6rem; }}
      h2    {{ font-size: 1.2rem; }}
      .grid {{ grid-template-columns: 1fr 1fr; gap: 16px; }}
      .toolbar {{ flex-direction: row; flex-wrap: wrap; align-items: center; }}
      .toolbar input  {{ flex: 1; min-width: 180px; width: auto; }}
      .toolbar select {{ min-width: 140px; width: auto; }}
      #count {{ margin-left: auto; }}
      tr:hover td {{ background: #1a1d27; }}
      .col-country, .col-source {{ display: table-cell; }}
    }}
  </style>
</head>
<body>
  <h1>🤖 Cobalt.xyz · AI Radar</h1>
  <p style="color:#adb5bd;font-size:0.9rem;margin-bottom:4px;">L'actualité IA mondiale : tendances, innovations et influences.</p>
  <p>{total} articles · {date_lbl} · généré le {now}</p>

  <div class="charts-row">
    <div class="globe-card">{globe_html}</div>
    <div class="radar-card">{radar_html}</div>
  </div>
  <div id="country-filter-bar">
    <span id="country-filter-label"></span>
    <button id="country-filter-clear" onclick="clearCountryFilter()">✕ Effacer le filtre</button>
  </div>

  <h2>🔥 Hot Articles</h2>
  {hot_cards}

  <h2>📋 Derniers articles</h2>
  <div class="toolbar">
    <input  id="f-search" type="search" placeholder="🔍 Recherche dans les titres…" autocomplete="off">
    <select id="f-sent"><option value="">Tous les sentiments</option>{opt_sent}</select>
    <select id="f-cat"><option value="">Toutes les catégories</option>{opt_cat}</select>
    <select id="f-src"><option value="">Toutes les sources</option>{opt_src}</select>
    <span id="count"></span>
  </div>
  <div class="table-wrap">
    <table id="articles-table">
      <thead>
        <tr>
          <th>Publié</th>
          <th>Sent.</th>
          <th>Titre</th>
          <th class="col-source">Source</th>
          <th class="col-country">Pays</th>
          <th>Catégorie</th>
        </tr>
      </thead>
      <tbody>
{table_rows}
      </tbody>
    </table>
  </div>

  <script>
    // ── ISO-3 → display name (mirrored from Python) ─────────────────────────
    const ISO3_NAME = {json.dumps(ISO3_TO_NAME)};

    // ── Globe click → country filter ────────────────────────────────────────
    let selectedIso = '';

    function updateHotTabCounts() {{
      document.querySelectorAll('.hot-tab[data-group]').forEach(function(btn) {{
        var group = btn.dataset.group;
        var panel = document.getElementById('hot-' + group);
        if (!panel) return;
        var cards = panel.querySelectorAll('[data-iso]');
        var visible = 0;
        cards.forEach(function(c) {{
          if (!selectedIso || c.dataset.iso === selectedIso) visible++;
        }});
        var span = btn.querySelector('.hot-tab__count');
        if (span) span.textContent = visible;
        // Dim button when count is 0, restore otherwise
        if (visible === 0) {{
          btn.style.opacity = '0.35';
        }} else {{
          btn.style.opacity = '';
        }}
      }});
    }}

    function applyCountryFilter() {{
      const bar   = document.getElementById('country-filter-bar');
      const label = document.getElementById('country-filter-label');
      const allCards = document.querySelectorAll('[data-iso]');

      if (selectedIso) {{
        bar.classList.add('visible');
        label.textContent = '🌍 Pays sélectionné : ' + (ISO3_NAME[selectedIso] || selectedIso);
        allCards.forEach(el => {{
          el.style.display = (el.dataset.iso === selectedIso) ? '' : 'none';
        }});
      }} else {{
        bar.classList.remove('visible');
        allCards.forEach(el => {{ el.style.display = ''; }});
      }}
      updateHotTabCounts();  // sync tab button counts with visible cards
      applyFilters();         // re-run text/sent/cat/src filters on top
    }}

    function clearCountryFilter() {{
      selectedIso = '';
      applyCountryFilter();
    }}

    // Attach Plotly globe click event after DOM is ready
    window.addEventListener('load', function() {{
      var gd = document.getElementById('globe-div');
      if (!gd) return;
      gd.on('plotly_click', function(data) {{
        if (!data || !data.points || !data.points[0]) return;
        var iso = data.points[0].location;
        if (!iso) return;
        selectedIso = (selectedIso === iso) ? '' : iso;   // toggle
        applyCountryFilter();
      }});
    }});

    // ── Table filters ────────────────────────────────────────────────────────
    const rows    = Array.from(document.querySelectorAll('#articles-table tbody tr'));
    const search  = document.getElementById('f-search');
    const fSent   = document.getElementById('f-sent');
    const fCat    = document.getElementById('f-cat');
    const fSrc    = document.getElementById('f-src');
    const counter = document.getElementById('count');

    function applyFilters() {{
      const q    = search.value.toLowerCase();
      const sent = fSent.value;
      const cat  = fCat.value;
      const src  = fSrc.value;
      let visible = 0;
      rows.forEach(row => {{
        // skip rows already hidden by country filter
        if (selectedIso && row.dataset.iso !== selectedIso) {{
          row.style.display = 'none';
          return;
        }}
        const match =
          (!q    || row.dataset.title.includes(q))    &&
          (!sent || row.dataset.sentiment === sent)    &&
          (!cat  || row.dataset.category  === cat)     &&
          (!src  || row.dataset.source    === src);
        row.style.display = match ? '' : 'none';
        if (match) visible++;
      }});
      counter.textContent = visible + ' article' + (visible !== 1 ? 's' : '') + ' affiché' + (visible !== 1 ? 's' : '');
    }}

    [search, fSent, fCat, fSrc].forEach(el => el.addEventListener('input', applyFilters));
    applyFilters();
  </script>
</body>
</html>"""

    with open(output, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Dashboard exporté → {output}")

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if _is_streamlit():
    run_streamlit()
else:
    parser = argparse.ArgumentParser(description="AI Radar Dashboard")
    parser.add_argument("--export", action="store_true", help="Générer un fichier HTML statique")
    parser.add_argument("--days",   type=int, default=7,  help="Fenêtre d'analyse en jours")
    parser.add_argument("--output", type=str, default="dashboard.html", help="Fichier de sortie")
    args = parser.parse_args()

    if args.export:
        run_export(args.days, args.output)
    else:
        print("Usage :")
        print("  Local     : streamlit run dashboard.py")
        print("  Export CI : python dashboard.py --export [--days N] [--top N]")
