"""
AI Radar - Dashboard de Veille
Usage local : streamlit run dashboard.py
Usage CI    : python dashboard.py --export [--days N] [--top N]
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import plotly.express as px
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
    "Geopolitique":           "#6a0572",
}
CATEGORY_EMOJI = {
    "Innovation / Tech":      "🚀",
    "Politique / Regulation": "⚖️",
    "Business / Industrie":   "💼",
    "Societe / Ethique":      "🤝",
    "Recherche Academique":   "🎓",
    "Drama / Controverses":   "💥",
    "Geopolitique":           "🌍",
}
SENTIMENT_COLORS = {
    "Positif": "#2a9d8f",
    "Negatif": "#e63946",
    "Neutre":  "#adb5bd",
}

PLOTLY_TEMPLATE = "plotly_dark"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL ou SUPABASE_KEY manquant.")
    return create_client(url, key)


def load_articles(days: int) -> list[dict]:
    client = _supabase_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    resp = (
        client.table("articles")
        .select("title, source, country, published, category, sentiment, url")
        .gte("published", cutoff)
        .order("published", desc=True)
        .execute()
    )
    return resp.data or []

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
    by_date: dict[str, int] = defaultdict(int)
    for a in articles:
        by_date[a["published"]] += 1

    dates  = sorted(by_date)
    values = [by_date[d] for d in dates]

    fig = go.Figure(go.Scatter(
        x=dates, y=values,
        mode="lines+markers",
        fill="tozeroy",
        line=dict(color="#00b4d8", width=2),
        marker=dict(size=6),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Tendance quotidienne",
        xaxis_title="Date",
        yaxis_title="Articles",
        margin=dict(l=10, r=10, t=50, b=20),
        height=260,
    )
    return fig


def fig_sources(articles: list[dict], top_n: int = 10) -> go.Figure:
    counts = Counter(f"{a['country']} {a['source']}" for a in articles)
    top    = counts.most_common(top_n)
    labels = [t[0] for t in reversed(top)]
    values = [t[1] for t in reversed(top)]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color="#f4a261",
        text=values, textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Top {top_n} sources",
        xaxis_title="Articles",
        margin=dict(l=10, r=30, t=50, b=20),
        height=max(280, top_n * 30),
    )
    return fig


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

    z      = [matrix[c] for c in cats]
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

    # Sidebar
    with st.sidebar:
        st.title("🤖 AI Radar")
        st.markdown("---")
        days  = st.slider("Fenêtre d'analyse (jours)", 1, 90, 7)
        top_n = st.slider("Nb de sources", 5, 20, 10)
        st.markdown("---")
        st.caption("Données : Supabase · Classif : Groq")

    # Load
    with st.spinner("Chargement des données..."):
        try:
            articles = load_articles(days)
        except RuntimeError as e:
            st.error(str(e))
            return

    if not articles:
        st.warning(f"Aucun article trouvé sur les {days} derniers jours.")
        return

    # KPIs
    total    = len(articles)
    pos_pct  = round(sum(1 for a in articles if a["sentiment"] == "Positif") / total * 100)
    neg_pct  = round(sum(1 for a in articles if a["sentiment"] == "Negatif") / total * 100)
    top_cat  = Counter(a["category"] for a in articles).most_common(1)[0][0]
    nb_src   = len({a["source"] for a in articles})
    dates    = sorted({a["published"] for a in articles})
    date_lbl = f"{dates[0]} → {dates[-1]}" if dates else "—"

    st.markdown(f"## 📰 {total} articles · {date_lbl}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total articles",     total)
    k2.metric("Sentiment positif",  f"{pos_pct} %")
    k3.metric("Sentiment négatif",  f"{neg_pct} %")
    k4.metric("Sources actives",    nb_src)

    st.markdown("---")

    # Row 1: categories + sentiments
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig_categories(articles), use_container_width=True)
    c2.plotly_chart(fig_sentiments(articles), use_container_width=True)

    # Row 2: trend + heatmap
    c3, c4 = st.columns([2, 3])
    c3.plotly_chart(fig_trend(articles),   use_container_width=True)
    c4.plotly_chart(fig_heatmap(articles), use_container_width=True)

    # Row 3: sources
    st.plotly_chart(fig_sources(articles, top_n), use_container_width=True)

    # Row 4: latest articles table
    st.markdown(f"### 📋 Derniers articles")
    import pandas as pd
    df = pd.DataFrame(articles)
    df["category"] = df["category"].apply(
        lambda c: f"{CATEGORY_EMOJI.get(c, '📌')} {c}"
    )
    df["lien"] = df["url"].apply(lambda u: f"[↗]({u})")
    st.dataframe(
        df[["published", "sentiment", "title", "source", "country", "category", "lien"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "published":  st.column_config.DateColumn("Date"),
            "sentiment":  st.column_config.TextColumn("Sent."),
            "title":      st.column_config.TextColumn("Titre", width="large"),
            "source":     st.column_config.TextColumn("Source"),
            "country":    st.column_config.TextColumn(""),
            "category":   st.column_config.TextColumn("Catégorie"),
            "lien":       st.column_config.LinkColumn("Lien", width="small"),
        },
    )

# ---------------------------------------------------------------------------
# HTML export (CI mode)
# ---------------------------------------------------------------------------

def run_export(days: int, top_n: int, output: str = "dashboard.html") -> None:
    print(f"Chargement des articles ({days} derniers jours)...")
    try:
        articles = load_articles(days)
    except RuntimeError as e:
        print(f"Erreur : {e}", file=sys.stderr)
        sys.exit(1)

    if not articles:
        print("Aucun article trouvé.")
        sys.exit(0)

    total   = len(articles)
    dates   = sorted({a["published"] for a in articles})
    date_lbl = f"{dates[0]} → {dates[-1]}" if dates else "—"
    print(f"{total} articles trouvés ({date_lbl})")

    figs = [
        fig_categories(articles),
        fig_sentiments(articles),
        fig_trend(articles),
        fig_heatmap(articles),
        fig_sources(articles, top_n),
    ]

    # Combine all figures into a single self-contained HTML
    html_parts = []
    for i, fig in enumerate(figs):
        include_js = "cdn" if i == 0 else False
        html_parts.append(pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=include_js,
            config={"responsive": True},
        ))

    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    full_html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Radar — Dashboard</title>
  <style>
    body {{ background: #0e1117; color: #fafafa; font-family: sans-serif; margin: 0; padding: 20px; }}
    h1   {{ color: #00b4d8; margin-bottom: 4px; }}
    p    {{ color: #888; margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .full {{ grid-column: 1 / -1; }}
    .card {{ background: #1a1d27; border-radius: 8px; padding: 8px; }}
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
    <div class="card full">{html_parts[4]}</div>
  </div>
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
    parser.add_argument("--top",    type=int, default=10, help="Nb de sources à afficher")
    parser.add_argument("--output", type=str, default="dashboard.html", help="Fichier de sortie")
    args = parser.parse_args()

    if args.export:
        run_export(args.days, args.top, args.output)
    else:
        print("Usage :")
        print("  Local     : streamlit run dashboard.py")
        print("  Export CI : python dashboard.py --export [--days N] [--top N]")
