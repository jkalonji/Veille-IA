"""
AI Radar - Dashboard de Veille
Usage local : streamlit run dashboard.py
Usage CI    : python dashboard.py --export [--days N] [--top N]
"""

import argparse
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

# Hot source groups — order = tab display order
HOT_SOURCE_META: list[dict] = [
    {"key": "hn",      "label": "Sujets en débat",     "icon": "💬", "color": "#ff6600", "border": "#ff6600"},
    {"key": "github",  "label": "Tech viral",          "icon": "⭐", "color": "#238636", "border": "#3fb950"},
    {"key": "db",      "label": "Sujets de société",   "icon": "📡", "color": "#0e7490", "border": "#06b6d4"},
    {"key": "trends",  "label": "Tendances montantes", "icon": "🔮", "color": "#7b2ff7", "border": "#9d4edd"},
    {"key": "unknown", "label": "Hot topics",          "icon": "🔥", "color": "#92400e", "border": "#f4a261"},
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
    return COUNTRY_TO_REGION.get(country, "Autre")

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
    """Return the highest-priority hot source key for an article."""
    raw = a.get("hot_source") or ""
    parts = [p for p in raw.split("|") if p]
    if not parts:
        return "unknown"
    return min(parts, key=lambda s: _SOURCE_ORDER.get(s, 99))


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
        .select("title, source, country, published, category, sentiment, url, hot_topic, hot_source")
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

def fig_categories(articles: list[dict]) -> go.Figure:
    counts = Counter(a["category"] for a in articles)
    cats   = list(CATEGORY_COLORS.keys())
    values = [counts.get(c, 0) for c in cats]
    labels = [f"{CATEGORY_EMOJI[c]} {c}" for c in cats]
    colors = list(CATEGORY_COLORS.values())

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=values, textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Articles par catégorie",
        xaxis_title="Nombre d'articles",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=30, t=50, b=20),
        height=320,
    )
    return fig


def fig_sentiments(articles: list[dict]) -> go.Figure:
    counts = Counter(a["sentiment"] for a in articles)
    labels = list(SENTIMENT_COLORS.keys())
    values = [counts.get(s, 0) for s in labels]
    colors = list(SENTIMENT_COLORS.values())

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=colors,
        hole=0.55,
        textinfo="label+percent",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Répartition des sentiments",
        margin=dict(l=10, r=10, t=50, b=10),
        height=320,
        showlegend=False,
    )
    return fig


def fig_trend(articles: list[dict]) -> go.Figure:
    """Stacked area chart: articles per day broken down by sentiment."""
    sentiments = ["Positif", "Neutre", "Negatif"]  # stacking order bottom→top

    # Count by (date, sentiment)
    by_date_sent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a in articles:
        by_date_sent[a["published"]][a.get("sentiment", "Neutre")] += 1

    dates = sorted(by_date_sent)

    fig = go.Figure()
    for sent in sentiments:
        values = [by_date_sent[d][sent] for d in dates]
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            name=sent,
            mode="lines",
            stackgroup="one",          # enables stacked area
            fillcolor=SENTIMENT_COLORS[sent],
            line=dict(color=SENTIMENT_COLORS[sent], width=1),
            hovertemplate="%{y} " + sent + "<extra></extra>",
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Tendance quotidienne (par sentiment)",
        xaxis_title="Date",
        yaxis_title="Articles",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=20),
        height=260,
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

    groups: dict[str, list[dict]] = {m["key"]: [] for m in HOT_SOURCE_META}
    for a in hot:
        groups[_primary_hot_source(a)].append(a)

    active = [m for m in HOT_SOURCE_META if groups[m["key"]]]
    if not active:
        container.info("Aucun article hot topic sur la période sélectionnée.")
        return

    tab_labels = [f"{m['icon']} {m['label']} ({len(groups[m['key']])})" for m in active]
    tabs = container.tabs(tab_labels)
    for tab, meta in zip(tabs, active):
        group = sorted(groups[meta["key"]], key=_hot_sort_key)
        cards_html = "".join(_render_hot_card_html(a, meta) for a in group)
        tab.markdown(cards_html, unsafe_allow_html=True)


def fig_heatmap(articles: list[dict]) -> go.Figure:
    """Articles by day-of-week × category."""
    day_names = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    cats = list(CATEGORY_COLORS.keys())
    matrix = {cat: [0] * 7 for cat in cats}

    for a in articles:
        try:
            dow = datetime.strptime(a["published"], "%Y-%m-%d").weekday()
            matrix[a["category"]][dow] += 1
        except (ValueError, KeyError):
            pass

    z       = [matrix[c] for c in cats]
    ylabels = [f"{CATEGORY_EMOJI[c]} {c}" for c in cats]

    fig = go.Figure(go.Heatmap(
        z=z, x=day_names, y=ylabels,
        colorscale="Blues",
        text=z, texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Heatmap : catégorie × jour de la semaine",
        margin=dict(l=10, r=10, t=50, b=20),
        height=300,
    )
    return fig

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
        page_title="AI Radar",
        page_icon="🤖",
    )

    @st.cache_data(ttl=300)
    def _cached_load(days: int) -> list[dict]:
        return load_articles(days)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🤖 AI Radar")
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

    # ── Charts ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig_categories(filtered), use_container_width=True)
    c2.plotly_chart(fig_sentiments(filtered), use_container_width=True)

    c3, c4 = st.columns([2, 3])
    c3.plotly_chart(fig_trend(filtered),   use_container_width=True)
    c4.plotly_chart(fig_heatmap(filtered), use_container_width=True)

    _render_hot_articles(filtered, st)

    # ── Table — local filters ─────────────────────────────────────────────────
    filtered = _deduplicate_articles(filtered)
    st.markdown("### 📋 Derniers articles")

    fa, fb, fc, fd = st.columns([3, 2, 1, 1])
    search   = fa.text_input("🔍 Recherche dans les titres", placeholder="ex: GPT, OpenAI, Mistral…")
    all_srcs = sorted({a["source"] for a in filtered})
    sel_srcs = fb.multiselect("Source", all_srcs, default=all_srcs, label_visibility="visible")
    only_new = fc.checkbox("Nouveautés 24h", value=False)
    only_top = fd.checkbox("Catégorie dominante", value=False)

    # Apply local filters to the display dataframe only
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    table_rows = [
        a for a in filtered
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
        rows.append(
            f'<tr data-title="{title.lower()}" data-sentiment="{sent}" '
            f'data-category="{cat}" data-source="{src}"{hot_class}>'
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
    return f"""
    <div style="border-radius:8px;padding:12px 14px;margin-bottom:8px;{card_style}">
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
    """Build hot articles as tabs grouped by detection source (CI HTML export)."""
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    hot = _deduplicate_articles([
        a for a in articles if a.get("hot_topic") and a.get("published", "") >= week_ago
    ])
    if not hot:
        return "<p style='color:#888;'>Aucun article hot topic sur la période.</p>"

    groups: dict[str, list[dict]] = {m["key"]: [] for m in HOT_SOURCE_META}
    for a in hot:
        groups[_primary_hot_source(a)].append(a)

    # Only show tabs that have content
    active = [m for m in HOT_SOURCE_META if groups[m["key"]]]
    if not active:
        return "<p style='color:#888;'>Aucun article hot topic sur la période.</p>"

    first_key = active[0]["key"]

    # ── Tab buttons ───────────────────────────────────────────────────────────
    buttons = ""
    for m in active:
        count     = len(groups[m["key"]])
        is_active = m["key"] == first_key
        buttons += (
            f'<button class="hot-tab{"  hot-tab--active" if is_active else ""}" '
            f'data-group="{m["key"]}" '
            f'style="{"border-color:"+m["border"]+";color:"+m["color"] if is_active else ""}">'
            f'{m["icon"]} {m["label"]}'
            f'<span class="hot-tab__count">{count}</span>'
            f'</button>'
        )

    # ── Tab panels ────────────────────────────────────────────────────────────
    panels = ""
    for m in active:
        group   = sorted(groups[m["key"]], key=_hot_sort_key)
        display = "block" if m["key"] == first_key else "none"
        cards   = "".join(_render_hot_card_html(a, m) for a in group)
        panels += f'<div class="hot-panel" id="hot-{m["key"]}" style="display:{display}">{cards}</div>'

    return f"""
<style>
  .hot-tabs {{
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px;
  }}
  .hot-tab {{
    background: #1a1d27; color: #adb5bd;
    border: 2px solid #2a2d3a; border-radius: 20px;
    padding: 7px 16px; font-size: 13px; cursor: pointer;
    transition: all 0.15s ease; white-space: nowrap;
  }}
  .hot-tab:hover {{ background: #252836; color: #fafafa; }}
  .hot-tab--active {{ background: #252836; color: #fafafa; font-weight: 700; }}
  .hot-tab__count {{
    background: #333; border-radius: 10px;
    padding: 1px 7px; font-size: 11px; margin-left: 6px; font-weight: 400;
  }}
</style>
<div class="hot-tabs">{buttons}</div>
<div>{panels}</div>
<script>
(function() {{
  var tabs   = document.querySelectorAll('.hot-tab');
  var panels = document.querySelectorAll('.hot-panel');
  tabs.forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      var meta = {{
        hn:      {{ border:'#ff6600', color:'#ff6600' }},
        github:  {{ border:'#3fb950', color:'#238636' }},
        db:      {{ border:'#06b6d4', color:'#0e7490' }},
        trends:  {{ border:'#9d4edd', color:'#7b2ff7' }},
        unknown: {{ border:'#f4a261', color:'#92400e' }},
      }};
      tabs.forEach(function(b) {{
        b.classList.remove('hot-tab--active');
        b.style.borderColor = '';
        b.style.color = '';
      }});
      panels.forEach(function(p) {{ p.style.display = 'none'; }});
      btn.classList.add('hot-tab--active');
      var g = btn.dataset.group;
      if (meta[g]) {{ btn.style.borderColor = meta[g].border; btn.style.color = meta[g].color; }}
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

    figs = [
        fig_categories(articles),
        fig_sentiments(articles),
        fig_trend(articles),
        fig_heatmap(articles),
    ]

    # Combine all figures into a single self-contained HTML
    # FIX 8: embed Plotly bundle locally (first fig only) — works offline/CI without CDN
    html_parts = []
    for i, fig in enumerate(figs):
        html_parts.append(pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=True if i == 0 else False,
            config={"responsive": True},
        ))

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
  <title>AI Radar — Dashboard</title>
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

    /* ── Charts grid — 1 col mobile, 2 col desktop ──────── */
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
    .card {{ background: #1a1d27; border-radius: 10px; padding: 6px; overflow: hidden; }}

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
  <h1>🤖 AI Radar — Dashboard de Veille</h1>
  <p>{total} articles · {date_lbl} · généré le {now}</p>

  <div class="grid">
    <div class="card">{html_parts[0]}</div>
    <div class="card">{html_parts[1]}</div>
    <div class="card">{html_parts[2]}</div>
    <div class="card">{html_parts[3]}</div>
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
