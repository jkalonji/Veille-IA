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


def load_articles(days: int) -> list[dict]:
    client = _supabase_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    resp = (
        client.table("articles")
        .select("title, source, country, published, category, sentiment, url, hot_topic")
        .gte("published", cutoff)
        .order("hot_topic", desc=True)
        .order("published", desc=True)
        .execute()
    )
    articles = resp.data or []
    # FIX 2: normalize published to YYYY-MM-DD regardless of what Supabase returns
    for a in articles:
        if a.get("published"):
            a["published"] = _normalize_date(a["published"])
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
        p1, p2, p3, p4 = st.columns(4)
        if p1.button("1j",  use_container_width=True): st.session_state["days"] = 1
        if p2.button("7j",  use_container_width=True): st.session_state["days"] = 7
        if p3.button("30j", use_container_width=True): st.session_state["days"] = 30
        if p4.button("90j", use_container_width=True): st.session_state["days"] = 90

        days  = st.slider("Fenêtre d'analyse (jours)", 1, 90, key="days")
        top_n = st.slider("Nb de sources", 5, 20, 10)

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

    st.plotly_chart(fig_sources(filtered, top_n), use_container_width=True)

    # ── Table — local filters ─────────────────────────────────────────────────
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
        # Hot topic articles are already first (ordered by Supabase); add 🔥 badge in title
        df["title"] = df.apply(
            lambda r: f"🔥 {r['title']}" if r["hot_topic"] else r["title"], axis=1
        )
        st.dataframe(
            df[["published", "sentiment", "title", "source", "country", "category", "lien"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "published": st.column_config.DateColumn("Date"),
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
        rows.append(
            f'<tr data-title="{title.lower()}" data-sentiment="{sent}" '
            f'data-category="{cat}" data-source="{src}"{hot_class}>'
            f"<td>{a.get('published', '')}</td>"
            f"<td>{sent}</td>"
            f'<td><a href="{url}" target="_blank">{hot_badge}{title}</a></td>'
            f"<td>{src}</td>"
            f"<td>{a.get('country', '')}</td>"
            f"<td>{emoji} {cat}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


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

    total    = len(articles)
    dates    = sorted({a["published"] for a in articles})
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
    # FIX 8: embed Plotly bundle locally (first fig only) — works offline/CI without CDN
    html_parts = []
    for i, fig in enumerate(figs):
        html_parts.append(pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=True if i == 0 else False,
            config={"responsive": True},
        ))

    table_rows = _articles_to_html_table(articles)

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
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Radar — Dashboard</title>
  <style>
    body       {{ background: #0e1117; color: #fafafa; font-family: sans-serif; margin: 0; padding: 20px; }}
    h1         {{ color: #00b4d8; margin-bottom: 4px; }}
    h2         {{ color: #fafafa; margin-top: 32px; }}
    p          {{ color: #888; margin-top: 0; }}
    .grid      {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .full      {{ grid-column: 1 / -1; }}
    .card      {{ background: #1a1d27; border-radius: 8px; padding: 8px; }}
    table      {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th         {{ background: #1a1d27; color: #00b4d8; text-align: left; padding: 8px 10px; position: sticky; top: 0; z-index: 1; }}
    td         {{ padding: 6px 10px; border-bottom: 1px solid #2a2d3a; vertical-align: top; }}
    tr:hover td {{ background: #1a1d27; }}
    a          {{ color: #00b4d8; text-decoration: none; }}
    a:hover    {{ text-decoration: underline; }}
    .table-wrap {{ max-height: 560px; overflow-y: auto; border-radius: 8px; background: #0e1117; }}
    tr.hot td  {{ background: #1f1a10; border-left: 3px solid #f4a261; }}
    .toolbar   {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
                  background: #1a1d27; padding: 10px 12px; border-radius: 8px; margin-bottom: 8px; }}
    .toolbar input, .toolbar select {{
      background: #0e1117; color: #fafafa; border: 1px solid #2a2d3a;
      border-radius: 6px; padding: 6px 10px; font-size: 13px; outline: none; }}
    .toolbar input  {{ flex: 1; min-width: 200px; }}
    .toolbar select {{ min-width: 140px; }}
    .toolbar input::placeholder {{ color: #555; }}
    #count     {{ margin-left: auto; color: #888; font-size: 12px; white-space: nowrap; }}
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

  <h2>📋 Derniers articles</h2>
  <div class="toolbar">
    <input  id="f-search" type="text"   placeholder="🔍 Recherche dans les titres…">
    <select id="f-sent">  <option value="">Tous les sentiments</option>{opt_sent}</select>
    <select id="f-cat">   <option value="">Toutes les catégories</option>{opt_cat}</select>
    <select id="f-src">   <option value="">Toutes les sources</option>{opt_src}</select>
    <span   id="count"></span>
  </div>
  <div class="table-wrap">
    <table id="articles-table">
      <thead>
        <tr><th>Date</th><th>Sent.</th><th>Titre</th><th>Source</th><th>Pays</th><th>Catégorie</th></tr>
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
          (!q    || row.dataset.title.includes(q))     &&
          (!sent || row.dataset.sentiment === sent)     &&
          (!cat  || row.dataset.category  === cat)      &&
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
    parser.add_argument("--top",    type=int, default=10, help="Nb de sources à afficher")
    parser.add_argument("--output", type=str, default="dashboard.html", help="Fichier de sortie")
    args = parser.parse_args()

    if args.export:
        run_export(args.days, args.top, args.output)
    else:
        print("Usage :")
        print("  Local     : streamlit run dashboard.py")
        print("  Export CI : python dashboard.py --export [--days N] [--top N]")
