"""
AI layer cake section of the dashboard ("ia" domain only), shared by the
Streamlit app and the static export of dashboard.py. See SPECS_LAYER_CAKE.md §7.

The section is one self-contained HTML block (CSS + inline SVG + a little JS):
the static export inlines it, Streamlit renders it with components.html.
"""

import sys
from collections import Counter
from datetime import date, timedelta
from html import escape
from urllib.parse import urlparse

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import cake
import cake_market
import chart_info

PERIODS = (7, 30, 90)
MAX_ARROWS = 12   # per period view, highest scores first
COLORS = {"montee": "#f4a261", "descente": "#7eb8f7"}
LAYER_COLORS = {"energie": "#e9c46a", "puces": "#9b5de5", "infrastructure": "#2a9d8f",
                "modeles": "#4488ff", "applications": "#e76f51"}
STALE_PRICE_DAYS = 4


def _template() -> str:
    """dashboard.py registers the "cobalt" chart theme; fall back when used alone."""
    return "cobalt" if "cobalt" in pio.templates else "plotly_dark"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def layer_counts(articles: list[dict]) -> Counter:
    """Articles per layer code (an article can count in up to 3 layers)."""
    return Counter(code for a in articles for code in (a.get("layers") or []))


def load_cake_data(client, days: int = 180) -> dict | None:
    """Events, actors and the layers of recent articles. None when the
    Phase B tables are not there yet (the section falls back to counts)."""
    if client is None:
        return None
    try:
        actors = client.table("actors").select(
            "id, name, home_layers, weight, weight_breakdown, weight_confidence, needs_review").execute().data or []
        since = (date.today() - timedelta(days=days)).isoformat()
        events = cake.fetch_all_rows(lambda a, b: client.table("layer_events").select(cake.EVENT_COLS)
                                     .gte("last_seen", since).order("id").range(a, b).execute())
    except Exception as e:
        print(f"Cake: tables absentes ({e}) — migration Phase B non appliquée.", file=sys.stderr)
        return None
    by_id = {a["id"]: a for a in actors}
    for e in events:
        e["actor"] = by_id.get(e.get("actor_id")) or {}
        e["actor_name"] = e["actor"].get("name") or "—"
    try:
        since90 = (date.today() - timedelta(days=max(PERIODS))).isoformat()
        layer_articles = cake.fetch_all_rows(lambda a, b: client.table("articles").select("url, published, layers")
                                             .eq("domain", "ia").gte("published", since90)
                                             .order("url").range(a, b).execute())
    except Exception:
        layer_articles = []
    for a in layer_articles:
        a["published"] = (a.get("published") or "")[:10]
        a["layers"] = a.get("layers") or []
    return {"events": events, "actors": actors, "articles": layer_articles}


def load_cake_market(client, days: int = 180) -> dict | None:
    """Baskets, token price index, scarcity z-scores and bottleneck state.
    None when the Phase C table (layer_daily) is not there yet."""
    if client is None:
        return None
    today = date.today().isoformat()
    since = (date.today() - timedelta(days=days)).isoformat()
    try:
        daily = cake.fetch_all_rows(lambda a, b: client.table("layer_daily").select(
            "day, layer, scarcity_z, baseline_rebuilding, bottleneck_status").gte("day", since)
            .order("day").range(a, b).execute())
        synthetic = client.table("layer_events").select("id, what, first_seen").eq("event_type", "goulot") \
            .gte("first_seen", (date.today() - timedelta(days=7)).isoformat()).execute().data or []
    except Exception as e:
        print(f"Cake: layer_daily absente ({e}) — migration Phase C non appliquée.", file=sys.stderr)
        return None
    symbols = [cake_market.basket_symbol(l) for l in cake.layer_codes()] + [
        cake_market.BENCHMARK, cake_market.TOKEN_IN, cake_market.TOKEN_OUT]
    try:
        series = cake_market.load_series(client, symbols, since)
    except Exception:
        series = {}
    return build_market_view(daily, series, synthetic, today)


def build_market_view(daily: list[dict], series: dict, synthetic: list[dict], today: str) -> dict:
    """Shape the stored market and scarcity data for the section (pure, testable)."""
    by_layer: dict[str, list[dict]] = {}
    for r in daily:
        by_layer.setdefault(r["layer"], []).append(r)
    last_day = max((r["day"] for r in daily), default=None)
    bottleneck = None
    if last_day:
        hit = next((r for r in daily if r["day"] == last_day and r.get("bottleneck_status")), None)
        bottleneck = ({"layer": hit["layer"], "status": "confirmé" if hit["bottleneck_status"] == "confirme" else "signalé"}
                      if hit else {"layer": None})
    extras: dict[int, dict] = {}
    for p in PERIODS:
        since = (date.fromisoformat(today) - timedelta(days=p)).isoformat()
        extras[p] = {}
        for code in cake.layer_codes():
            rows = by_layer.get(code) or []
            latest = rows[-1] if rows else {}
            basket = [v for d, v in series.get(cake_market.basket_symbol(code), []) if d >= since]
            ex = {"gauge": cake_market.gauge(latest.get("scarcity_z")),
                  "rebuilding": bool(latest.get("baseline_rebuilding")),
                  "spark": basket,
                  "change_pct": (basket[-1] / basket[0] - 1) * 100 if len(basket) >= 2 and basket[0] else None}
            if code == "modeles":
                ex["spark2"] = [v for d, v in series.get(cake_market.TOKEN_OUT, []) if d >= since]
            extras[p][code] = ex
    last_price = max((d for d, _ in series.get(cake_market.basket_symbol("puces"), [])), default=None)
    stale = ""
    if last_price and (date.fromisoformat(today) - date.fromisoformat(last_price)).days > STALE_PRICE_DAYS:
        stale = f"⚠️ cours au {last_price}"
    return {"bottleneck": bottleneck, "layer_extras": extras, "series": series, "daily": daily,
            "synthetic": synthetic, "stale": stale}


def fig_matrix(events: list[dict], today: str) -> go.Figure:
    """Origin → target heatmap of cross-layer moves over 90 days (diagonal
    empty), with buttons: count / sum of scores, and the change vs the
    previous 90 days."""
    codes = cake.layer_codes()
    meta = cake.layer_meta()
    t = date.fromisoformat(today)
    cur_since, prev_since = (t - timedelta(days=90)).isoformat(), (t - timedelta(days=180)).isoformat()
    tomorrow = (t + timedelta(days=1)).isoformat()

    def matrix(since, until, weight):
        m = [[0.0] * len(codes) for _ in codes]
        for e in events:
            if e.get("event_type") == "integration" and since <= (e.get("first_seen") or "") < until \
                    and e.get("origin_layer") in codes and e.get("target_layer") in codes:
                m[codes.index(e["origin_layer"])][codes.index(e["target_layer"])] += weight(e)
        for i in range(len(codes)):
            m[i][i] = None
        return m

    def diff(a, b):
        return [[None if x is None else x - y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]

    def count(e):
        return 1

    def total(e):
        return float(e.get("score") or 0)

    cur_n, cur_s = matrix(cur_since, tomorrow, count), matrix(cur_since, tomorrow, total)
    prev_n, prev_s = matrix(prev_since, cur_since, count), matrix(prev_since, cur_since, total)
    labels = [meta[c]["label"] for c in codes]
    seq = [[0, "#0c1222"], [0.4, "#0047ab"], [0.75, "#4488ff"], [1, "#7eb8f7"]]
    div = [[0, "#7eb8f7"], [0.5, "#0c1222"], [1, "#f4a261"]]
    fig = go.Figure(go.Heatmap(z=cur_n, x=labels, y=labels, colorscale=seq, zmin=0, xgap=3, ygap=3,
                               hovertemplate="%{y} → %{x}<br>%{z}<extra></extra>", texttemplate="%{z:.0f}",
                               colorbar=dict(thickness=10, outlinewidth=0)))
    views = [("Nombre", cur_n, seq, False), ("Somme des scores", cur_s, seq, False),
             ("Évolution (nombre)", diff(cur_n, prev_n), div, True),
             ("Évolution (scores)", diff(cur_s, prev_s), div, True)]
    buttons = [dict(label=lbl, method="restyle",
                    args=[{"z": [z], "colorscale": [cs], "zmid": [0 if centred else None],
                           "zmin": [None if centred else 0]}]) for lbl, z, cs, centred in views]
    fig.update_layout(
        template=_template(), height=420,
        title=dict(text="Mouvements origine → cible",
                   subtitle=dict(text="90 derniers jours · ligne = origine, colonne = cible")),
        xaxis=dict(side="bottom"), yaxis=dict(ticklabelstandoff=6),
        updatemenus=[dict(type="buttons", direction="right", buttons=buttons, x=0, y=-0.22, xanchor="left",
                          bgcolor="#0c1222", bordercolor="#1a2540", font=dict(size=10, color="#c8d4ec"))],
        margin=dict(l=10, r=10, t=70, b=80),
    )
    return fig


def fig_layer_curves(series: dict, daily: list[dict]) -> go.Figure:
    """The five baskets rebased to 100 and, underneath on the same time axis,
    each layer's scarcity z-score with the threshold and the confirmed
    bottleneck periods shaded."""
    meta = cake.layer_meta()
    cfg = cake.load_config()["bottleneck"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06)
    for code in cake.layer_codes():
        pts = series.get(cake_market.basket_symbol(code)) or []
        if pts:
            base = pts[0][1] or 1
            fig.add_trace(go.Scatter(x=[d for d, _ in pts], y=[v / base * 100 for _, v in pts], name=meta[code]["label"],
                                     line=dict(color=LAYER_COLORS[code], width=1.8), legendgroup=code), row=1, col=1)
    bench = series.get(cake_market.BENCHMARK) or []
    if bench:
        fig.add_trace(go.Scatter(x=[d for d, _ in bench], y=[v / bench[0][1] * 100 for _, v in bench], name="QQQ",
                                 line=dict(color="#5a6a8a", width=1.2, dash="dot")), row=1, col=1)
    for code in cake.layer_codes():
        rows = [r for r in daily if r["layer"] == code and r.get("scarcity_z") is not None]
        if rows:
            fig.add_trace(go.Scatter(x=[r["day"] for r in rows], y=[r["scarcity_z"] for r in rows],
                                     name=f"z {meta[code]['label']}", line=dict(color=LAYER_COLORS[code], width=1.4),
                                     legendgroup=code, showlegend=False), row=2, col=1)
    fig.add_hline(y=cfg["z_threshold"], line=dict(color="#e63946", width=1, dash="dash"), row=2, col=1)
    confirmed = sorted(r["day"] for r in daily if r.get("bottleneck_status") == "confirme")
    spans, start, prev = [], None, None
    for d in confirmed:
        if prev and (date.fromisoformat(d) - date.fromisoformat(prev)).days > 1:
            spans.append((start, prev))
            start = None
        start = start or d
        prev = d
    if start:
        spans.append((start, prev))
    for a, b in spans:
        fig.add_vrect(x0=a, x1=b, fillcolor="#f4a261", opacity=0.12, line_width=0)
    fig.update_layout(template=_template(), height=520,
                      title=dict(text="Paniers par couche et tensions", subtitle=dict(
                          text="base 100 · en dessous : z-score du solde de rareté (seuil 1,5), goulots confirmés surlignés")),
                      legend=dict(orientation="h", y=-0.08), margin=dict(l=10, r=10, t=70, b=40))
    fig.update_yaxes(title_text="base 100", row=1, col=1)
    fig.update_yaxes(title_text="z", row=2, col=1)
    return fig


def fig_value_staircase(value: dict) -> go.Figure | None:
    """§7.1 (5) — price of a MWh at the meter at each floor (log scale), with
    value added and multiplier above each bar; low-confidence floors lighter
    and prefixed with ≈, negative value added in red with ▼. A button switches
    to the monthly value added of each floor."""
    floors = [f for f in value.get("floors") or [] if f.get("price_usd_mwh")]
    if not floors:
        return None
    meta = cake.layer_meta()

    def money(v):
        return f"{v:,.0f}".replace(",", " ")

    labels, prices, colors, texts, hovers = [], [], [], [], []
    for f in floors:
        low = f["confidence"] == "low"
        labels.append(f"{meta[f['layer']]['emoji']} {meta[f['layer']]['label']}")
        prices.append(f["price_usd_mwh"])
        colors.append("rgba(68,136,255,0.45)" if low else "#4488ff")
        pre = "≈ " if low else ""
        line = f"{pre}{money(f['price_usd_mwh'])} $/MWh"
        hover = f"{pre}{money(f['price_usd_mwh'])} $/MWh au compteur"
        if f.get("value_added_usd_mwh") is not None:
            va = f["value_added_usd_mwh"]
            mult = f"×{f['multiplier']:.1f}".replace(".", ",") if f.get("multiplier") else ""
            if f["value_added_negative"]:
                line += f"<br><span style='color:#e63946'>▼ {money(va)} {mult}</span>"
                hover += "<br>vend sous le coût des couches inférieures"
            else:
                line += f"<br>+{money(va)} {mult}"
            hover += f"<br>valeur ajoutée {money(va)} $/MWh {mult}"
        texts.append(line)
        hovers.append(hover + "<br>sources : " + ", ".join(sorted({i['source'] or i['key'] for i in f['inputs'] if i.get('source_url')})))
    fig = go.Figure(go.Bar(x=labels, y=prices, marker=dict(color=colors, cornerradius=4), text=texts,
                           textposition="outside", cliponaxis=False, hovertext=hovers, hoverinfo="text"))
    hist = value.get("history") or []
    months = [h["month"] for h in hist]
    for f in floors:
        fig.add_trace(go.Scatter(x=months, y=[(h["floors"].get(f["layer"]) or {}).get("value_added") for h in hist],
                                 name=meta[f["layer"]]["label"], mode="lines+markers", visible=False,
                                 line=dict(color=LAYER_COLORS[f["layer"]])))
    n = len(floors)
    urls = sorted({i["source_url"] for f in floors for i in f["inputs"] if i.get("source_url")})
    domains = list(dict.fromkeys(urlparse(u).netloc.removeprefix("www.") for u in urls))
    links = [f'<a href="{next(u for u in urls if d in u)}">{d}</a>' for d in domains]
    links = " · ".join(links[:5]) + ("<br>" + " · ".join(links[5:]) if len(links) > 5 else "")
    fig.update_layout(
        template=_template(), height=480, showlegend=False,
        title=dict(text="Valeur d'un MWh à chaque couche",
                   subtitle=dict(text="$ par MWh au compteur, échelle log · ≈ = estimation · ▼ = valeur ajoutée négative")),
        yaxis=dict(type="log", title="$/MWh"),
        updatemenus=[dict(type="buttons", direction="right", x=0, y=-0.14, xanchor="left", yanchor="top",
                          bgcolor="#0c1222", bordercolor="#1a2540", font=dict(size=10, color="#c8d4ec"),
                          buttons=[dict(label="Escalier", method="update",
                                        args=[{"visible": [True] + [False] * n},
                                              {"yaxis.type": "log", "showlegend": False, "yaxis.title.text": "$/MWh"}]),
                                   dict(label="Évolution de la valeur ajoutée", method="update",
                                        args=[{"visible": [False] + [True] * n},
                                              {"yaxis.type": "linear", "showlegend": True,
                                               "yaxis.title.text": "valeur ajoutée $/MWh"}])])],
        annotations=[dict(text=f"Mise à jour {value.get('as_of', '?')} · {links}", showarrow=False, xref="paper",
                          yref="paper", x=0.5, y=-0.32, font=dict(size=9, color="#5a6a8a"))],
        margin=dict(l=10, r=10, t=80, b=120),
    )
    return fig


def cake_figures(data: dict | None, market: dict | None, today: str | None = None) -> list[tuple[str, go.Figure]]:
    """(chart_info key, Plotly figure) shown under the cake: matrix, value staircase, layer curves."""
    import layer_value
    today = today or date.today().isoformat()
    figs = []
    if data is not None:
        figs.append(("matrix", fig_matrix(data["events"], today)))
    staircase = fig_value_staircase(layer_value.load_output())
    if staircase is not None:
        figs.append(("value_staircase", staircase))
    if market and (market.get("series") or market.get("daily")):
        figs.append(("layer_curves", fig_layer_curves(market["series"], market["daily"])))
    return figs


def _hot_within(days: list[str], since: str) -> bool:
    """A story was hot at some point of the period: ≥ 2 articles over two consecutive days."""
    per_day = Counter(d for d in days if d >= since)
    for d, n in per_day.items():
        prev = (date.fromisoformat(d) - timedelta(days=1)).isoformat()
        if n + per_day.get(prev, 0) >= 2:
            return True
    return False


def strong_signals(data: dict, stories_lookup: dict | None, period: int, today: str) -> list[dict]:
    """§7.2 — hot stories whose articles mostly sit in a layer (≥ 50 %, or
    ≥ 30 % each for a story spanning several layers), and events scoring
    ≥ strong_signal_threshold. Each item: {kind, label, layers, date, ref}."""
    since = (date.fromisoformat(today) - timedelta(days=period)).isoformat()
    threshold = cake.load_config()["event_score"]["strong_signal_threshold"]
    by_url = {a["url"]: a for a in data["articles"]}
    out = []
    for sid, s in (stories_lookup or {}).items():
        if s.get("status") != "open":
            continue
        arts = [by_url[u] for u in s.get("article_urls") or [] if u in by_url]
        if not arts or not _hot_within([a["published"] for a in arts], since):
            continue
        share = Counter(l for a in arts for l in a["layers"])
        main_layers = [l for l, n in share.items() if n >= 0.5 * len(arts)]
        balanced = [l for l, n in share.items() if n >= 0.3 * len(arts)]
        layers = main_layers if main_layers else (balanced if len(balanced) >= 2 else [])
        if layers:
            label = s["label"] + (f" — {s['summary']}" if s.get("summary") else "")
            out.append({"kind": "story", "label": label, "layers": sorted(set(balanced) | set(layers)),
                        "date": max(a["published"] for a in arts), "ref": f"story-ia-{sid}"})
    for e in data["events"]:
        if e.get("event_type") == "integration" and (e.get("last_seen") or "") >= since \
                and (e.get("score") or 0) >= threshold:
            out.append({"kind": "event", "label": f"{e['actor_name']} : {e.get('what') or ''}",
                        "layers": [e["origin_layer"], e["target_layer"]], "date": e["last_seen"],
                        "ref": f"cake-ev-{e['id']}"})
    return sorted(out, key=lambda x: x["date"], reverse=True)


def _letters(n: int) -> list[str]:
    out = []
    for i in range(n):
        s, i = "", i + 1
        while i:
            i, r = divmod(i - 1, 26)
            s = chr(65 + r) + s
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt_move(e: dict) -> str:
    meta = cake.layer_meta()
    arrow = "⬆️" if e["direction"] == "montee" else "⬇️"
    return (f"{arrow} {escape(e['actor_name'])} · {meta[e['origin_layer']]['label']} → "
            f"{meta[e['target_layer']]['label']} · {cake.ENGAGEMENT_LABEL.get(e['engagement'], '').lower()} · "
            f"{cake.fmt_score(e['score'] or 0)}")


def banner_html(events: list[dict], today: str, bottleneck: str = "", synthetic: list[dict] | None = None) -> str:
    """Major moves of the last 7 days (and confirmed bottleneck migrations), plus the bottleneck state."""
    since = (date.fromisoformat(today) - timedelta(days=7)).isoformat()
    majors = sorted((e for e in events if e.get("event_type") == "integration" and e.get("is_major")
                     and (e.get("last_seen") or "") >= since), key=lambda e: e["last_seen"], reverse=True)
    items = [f'<a class="cake-banner__item" href="#cake-ev-{e["id"]}" data-ev="cake-ev-{e["id"]}">{_fmt_move(e)}</a>'
             for e in majors]
    items += [f'<span class="cake-banner__item">🎂 {escape(s["what"])}</span>' for s in synthetic or []]
    state = escape(bottleneck) if bottleneck else ""
    if not items and not state:
        return ""
    return (f'<div class="cake-banner"><div class="cake-banner__moves">{"".join(items) or "<span class=cake-muted>Aucun mouvement majeur sur 7 jours</span>"}</div>'
            + (f'<div class="cake-banner__state">{state}</div>' if state else "") + "</div>")


def _sparkline(values: list[float], x: float, y: float, w: float, h: float, color: str) -> str:
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    pts = " ".join(f"{x + i * w / (len(values) - 1):.1f},{y + h - (v - lo) / span * h:.1f}" for i, v in enumerate(values))
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.4"/>'


def cake_svg(period_events: list[dict], letters: dict, signal_counts: Counter, article_counts: Counter,
             layer_extras: dict | None = None, bottleneck: dict | None = None, uid: str = "") -> str:
    """The cake: five stacked bands (energy at the bottom) and one vertical
    arrow per cross-layer event, from origin to target band — solid for a
    descent, dashed for a climb, width by score. `layer_extras[code]` may hold
    {gauge (0-1), rebuilding (bool), spark: [..], change_pct, spark2: [..]}
    (Phase C); `bottleneck` = {layer, status}. `uid` keeps the arrowhead
    marker ids unique when several cakes share a page."""
    meta = cake.layer_meta()
    codes = cake.layer_codes()
    W, band_h, gap, left_w = 760, 58, 6, 540
    H = len(codes) * (band_h + gap)
    y_of = {c: (len(codes) - 1 - i) * (band_h + gap) for i, c in enumerate(codes)}
    extras = layer_extras or {}
    parts = [f'<svg viewBox="0 0 {W} {H}" class="cake-svg" role="img" aria-label="AI layer cake">',
             '<defs>' + "".join(
                 f'<marker id="cake-head-{d}{uid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" '
                 f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
                 for d, c in COLORS.items()) + '</defs>']
    for c in codes:
        y = y_of[c]
        is_bn = bottleneck and bottleneck.get("layer") == c
        stroke = "#f4a261" if is_bn else "#1a2540"
        parts.append(f'<g class="cake-band-g" data-layer="{c}" style="cursor:pointer">'
                     f'<rect x="0" y="{y}" width="{W}" height="{band_h}" rx="8" fill="#0c1222" stroke="{stroke}"/>'
                     f'<text x="14" y="{y + 24}" class="cake-t-name">{meta[c]["emoji"]} {escape(meta[c]["label"])}</text>'
                     f'<text x="14" y="{y + 44}" class="cake-t-sub">{signal_counts.get(c, 0)} signaux forts · '
                     f'{article_counts.get(c, 0)} articles</text>')
        if is_bn:
            parts.append(f'<text x="250" y="{y + 24}" class="cake-t-bn">goulot {escape(bottleneck["status"])}</text>')
        ex = extras.get(c) or {}
        if ex.get("gauge") is not None:
            gx, gw = 250, 80
            dash = ' stroke-dasharray="3 3"' if ex.get("rebuilding") else ""
            parts.append(f'<rect x="{gx}" y="{y + 36}" width="{gw}" height="8" rx="4" fill="#060810" stroke="#1a2540"{dash}/>'
                         f'<rect x="{gx}" y="{y + 36}" width="{gw * ex["gauge"]:.0f}" height="8" rx="4" fill="#e63946" opacity="0.8"/>'
                         f'<title>Rareté (z-score ramené à 0–1){" — normale en reconstruction" if ex.get("rebuilding") else ""}</title>')
        if ex.get("spark"):
            parts.append(_sparkline(ex["spark"], 345, y + 10, 115, band_h - 22, "#4488ff"))
            if ex.get("spark2"):
                parts.append(_sparkline(ex["spark2"], 345, y + 10, 115, band_h - 22, "#e9c46a"))
            chg = ex.get("change_pct")
            if chg is not None:
                color = "#3fb950" if chg >= 0 else "#e63946"
                parts.append(f'<text x="470" y="{y + 34}" class="cake-t-sub" fill="{color}" style="fill:{color}">'
                             f'{chg:+.1f}%</text>')
        parts.append("</g>")
    # arrows
    slot = (W - left_w - 20) / max(1, len(period_events))
    for i, e in enumerate(period_events):
        x = left_w + 10 + slot * (i + 0.5)
        y1 = y_of[e["origin_layer"]] + band_h / 2
        y2 = y_of[e["target_layer"]] + band_h / 2
        y2 += -band_h / 2 + 8 if y2 < y1 else band_h / 2 - 8
        color = COLORS[e["direction"]]
        dash = ' stroke-dasharray="6 4"' if e["direction"] == "montee" else ""
        width = 1.5 + (e.get("score") or 0) / 2.5
        parts.append(
            f'<g class="cake-arrow" data-ev="cake-ev-{e["id"]}" style="cursor:pointer">'
            f'<title>{escape(_fmt_move(e))}</title>'
            f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.1f}"'
            f'{dash} marker-end="url(#cake-head-{e["direction"]}{uid})"/>'
            f'<circle cx="{x:.1f}" cy="{y1:.1f}" r="9" fill="#060810" stroke="{color}"/>'
            f'<text x="{x:.1f}" y="{y1 + 4:.1f}" text-anchor="middle" class="cake-t-letter">{letters[e["id"]]}</text></g>')
    parts.append("</svg>")
    return "".join(parts)


def _event_card(e: dict, letter: str, story_refs: dict[str, str], link_stories: bool) -> str:
    meta = cake.layer_meta()
    codes = cake.layer_codes()
    lo, hi = sorted((codes.index(e["origin_layer"]), codes.index(e["target_layer"])))
    spanned = [meta[c]["label"] for c in codes[lo + 1:hi]]
    hist = {h["engagement"]: h["date"] for h in e.get("engagement_history") or []}
    frise = " → ".join(
        f'<span class="cake-step{" cake-step--on" if g in hist else ""}">{cake.ENGAGEMENT_LABEL[g]}'
        f'{" · " + hist[g] if g in hist else ""}</span>' for g in cake.ENGAGEMENTS)
    bd = e.get("score_breakdown") or {}
    actor = e.get("actor") or {}
    wb = actor.get("weight_breakdown") or {}
    comp = " · ".join(f"{k} {wb[k]:.2f}" if isinstance(wb.get(k), (int, float)) else f"{k} ?"
                      for k in ("size", "influence", "longevity", "implantation"))
    weight = actor.get("weight")
    low = actor.get("weight_confidence") == "low"
    weight_txt = ("≈ " if low else "") + (f"{weight:.2f}" if isinstance(weight, (int, float)) else "?")
    lists = "".join(
        f"<div><b>{label}</b> : {escape(', '.join(e.get(key) or [])) or '—'}</div>"
        for key, label in (("bypassed", "Court-circuités"), ("winners", "Gagnants"), ("losers", "Perdants")))
    links = "".join(f'<li><a href="{escape(u)}" target="_blank" rel="noopener">{escape(u[:90])}</a></li>'
                    for u in e.get("article_urls") or [])
    story = next((story_refs[u] for u in e.get("article_urls") or [] if u in story_refs), None)
    story_html = (f'<div><a href="#{story}" class="cake-story-link">📖 Voir l\'histoire dans « Suivi d\'histoires »</a></div>'
                  if story and link_stories else "")
    reading = "conquête (montée)" if e["direction"] == "montee" else "sécurisation (descente)"
    return f"""
<details class="cake-card" id="cake-ev-{e['id']}" data-layers="{e['origin_layer']} {e['target_layer']}">
  <summary><span class="cake-letter">{letter}</span> {_fmt_move(e)}{' <span class="cake-major">majeur</span>' if e.get('is_major') else ''}</summary>
  <div class="cake-card__body">
    <div>📌 {escape(e.get('what') or '')}</div>
    <div><b>Lecture</b> : {reading} · {e['distance']} couche{'s' if e['distance'] > 1 else ''}{' · enjambe ' + ', '.join(spanned) if spanned else ''}</div>
    <div class="cake-frise">{frise}</div>
    <div><b>Score</b> {cake.fmt_score(e.get('score') or 0)} = D {bd.get('D', '?')} · E {bd.get('E', '?')} · W {bd.get('W', '?')} · N {bd.get('N', '?')}</div>
    <div><b>Poids de l'acteur</b> {weight_txt} ({comp}){' · acteur à relire' if actor.get('needs_review') else ''}</div>
    {lists}
    {story_html}
    <ul class="cake-links">{links}</ul>
  </div>
</details>"""


CSS = """
<style>
  .cake-wrap { margin-bottom:14px; font-family:'DM Mono', ui-monospace, Menlo, Consolas, monospace; color:#c8d4ec; }
  .cake-muted { color:#5a6a8a; }
  .cake-banner { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; justify-content:space-between;
    background:#1f1a10; border:1px solid #f4a261; border-radius:10px; padding:10px 14px; margin-bottom:10px; font-size:13px; }
  .cake-banner__moves { display:flex; flex-wrap:wrap; gap:6px 16px; }
  .cake-banner__item { color:#e8eef8; text-decoration:none; }
  .cake-banner__state { color:#f4a261; white-space:nowrap; }
  .cake-periods { display:flex; gap:6px; margin-bottom:8px; }
  .cake-periods button { background:#0c1222; color:#c8d4ec; border:1px solid #1a2540; border-radius:6px;
    padding:5px 12px; cursor:pointer; font-family:inherit; font-size:12px; }
  .cake-periods button.active { background:#4488ff; color:#060810; border-color:#4488ff; }
  .cake-view { display:none; }
  .cake-view.active { display:block; }
  .cake-svg { width:100%; height:auto; display:block; }
  .cake-t-name { fill:#e8eef8; font-size:15px; font-family:'DM Serif Display', Georgia, serif; }
  .cake-t-sub { fill:#7d8db0; font-size:11px; }
  .cake-t-bn { fill:#f4a261; font-size:11px; }
  .cake-t-letter { fill:#e8eef8; font-size:10px; font-weight:600; }
  .cake-band-g.selected rect:first-child { stroke:#4488ff; }
  .cake-legend { font-size:11px; color:#5a6a8a; margin:6px 0 10px; }
  .cake-signals { list-style:none; padding:0; margin:0 0 12px; font-size:12px; max-height:220px; overflow-y:auto; }
  .cake-signals li { padding:4px 0; border-bottom:1px solid #1a2540; }
  .cake-signals a { color:#c8d4ec; }
  .cake-signals .cake-kind { color:#5a6a8a; margin-right:6px; }
  .cake-card { background:#0c1222; border:1px solid #1a2540; border-radius:10px; margin-bottom:8px; padding:0 14px; }
  .cake-card summary { cursor:pointer; padding:10px 0; font-size:13px; color:#e8eef8; list-style:none; }
  .cake-card summary::-webkit-details-marker { display:none; }
  .cake-card__body { font-size:12px; line-height:1.7; padding-bottom:10px; }
  .cake-letter { display:inline-block; min-width:22px; text-align:center; border:1px solid #4488ff; border-radius:10px;
    font-size:10px; margin-right:6px; }
  .cake-major { color:#060810; background:#f4a261; border-radius:8px; padding:0 6px; font-size:10px; margin-left:6px; }
  .cake-step { color:#5a6a8a; }
  .cake-step--on { color:#e8eef8; }
  .cake-links { margin:4px 0 0; padding-left:18px; }
  .cake-links a, .cake-story-link { color:#4488ff; }
  .cake-h3 { font-family:'DM Serif Display', Georgia, serif; color:#e8eef8; font-size:1.05rem; margin:14px 0 8px; }
</style>"""

JS = """
<script>
(function () {
  var root = document.currentScript.closest('.cake-wrap') || document;
  function openCard(id) {
    var card = root.querySelector('#' + id) || document.getElementById(id);
    if (!card) return;
    card.open = true;
    card.scrollIntoView({behavior: 'smooth', block: 'center'});
  }
  root.querySelectorAll('.cake-periods button').forEach(function (b) {
    b.addEventListener('click', function () {
      root.querySelectorAll('.cake-periods button').forEach(function (x) { x.classList.toggle('active', x === b); });
      root.querySelectorAll('.cake-view').forEach(function (v) { v.classList.toggle('active', v.dataset.period === b.dataset.period); });
    });
  });
  root.querySelectorAll('[data-ev]').forEach(function (el) {
    el.addEventListener('click', function (ev) { ev.preventDefault(); openCard(el.dataset.ev); });
  });
  root.querySelectorAll('.cake-band-g').forEach(function (band) {
    band.addEventListener('click', function () {
      var view = band.closest('.cake-view');
      var on = !band.classList.contains('selected');
      view.querySelectorAll('.cake-band-g').forEach(function (b) { b.classList.remove('selected'); });
      if (on) band.classList.add('selected');
      view.querySelectorAll('.cake-signals li').forEach(function (li) {
        li.style.display = (!on || (' ' + li.dataset.layers + ' ').indexOf(' ' + band.dataset.layer + ' ') >= 0) ? '' : 'none';
      });
    });
  });
})();
</script>"""


def cake_section_html(data: dict | None, articles: list[dict], stories_lookup: dict | None = None,
                      link_stories: bool = True, market: dict | None = None, today: str | None = None) -> str:
    """The whole 🎂 section. Without the Phase B tables (`data` None), only
    the articles-per-layer counter. `market` (Phase C) carries per-period
    layer extras, the bottleneck state and the extra HTML blocks."""
    if data is None:
        return cake_counts_html(articles)
    today = today or date.today().isoformat()
    market = market or {}
    meta = cake.layer_meta()
    integ = [e for e in data["events"] if e.get("event_type") == "integration"
             and e.get("origin_layer") in meta and e.get("target_layer") in meta]
    since90 = (date.fromisoformat(today) - timedelta(days=max(PERIODS))).isoformat()
    listed = sorted((e for e in integ if (e.get("last_seen") or "") >= since90),
                    key=lambda e: (e["last_seen"], e.get("score") or 0), reverse=True)
    letters = dict(zip([e["id"] for e in listed], _letters(len(listed))))
    bn = market.get("bottleneck")
    bn_label = ""
    if bn:
        bn_label = f"Goulot : {meta[bn['layer']]['label']} ({bn['status']})" if bn.get("layer") else "Pas de goulot net"
    if market.get("stale"):
        bn_label = " · ".join(x for x in (bn_label, market["stale"]) if x)

    views, buttons = "", ""
    for p in PERIODS:
        since = (date.fromisoformat(today) - timedelta(days=p)).isoformat()
        p_events = sorted((e for e in listed if e["last_seen"] >= since),
                          key=lambda e: e.get("score") or 0, reverse=True)[:MAX_ARROWS]
        p_events.sort(key=lambda e: e["last_seen"])
        signals = strong_signals(data, stories_lookup, p, today)
        sig_counts = Counter(l for s in signals for l in s["layers"])
        art_counts = layer_counts([a for a in data["articles"] if a["published"] >= since])
        bn_svg = {"layer": bn["layer"], "status": bn["status"]} if bn and bn.get("layer") else None
        svg = cake_svg(p_events, letters, sig_counts, art_counts,
                       (market.get("layer_extras") or {}).get(p), bn_svg, uid=f"-{p}")
        items = "".join(
            f'<li data-layers="{" ".join(s["layers"])}"><span class="cake-kind">{"📖" if s["kind"] == "story" else "🎂"} {s["date"]}</span>'
            + (f'<a href="#{s["ref"]}"' + (f' data-ev="{s["ref"]}"' if s["kind"] == "event" else "") + f'>{escape(s["label"])}</a>'
               if (s["kind"] == "event" or link_stories) else escape(s["label"]))
            + f' <span class="cake-muted">({", ".join(meta[l]["label"] for l in s["layers"])})</span></li>'
            for s in signals[:60])
        active = " active" if p == 30 else ""
        buttons += f'<button class="{active.strip()}" data-period="{p}">{p} j</button>'
        views += (f'<div class="cake-view{active}" data-period="{p}">{svg}'
                  f'<div class="cake-legend">Flèche pleine = descente (sécurisation) · pointillée = montée (conquête) · '
                  f'épaisseur = score · clic sur une bande = filtre des signaux forts</div>'
                  f'<div class="cake-h3">Signaux forts</div>'
                  f'<ul class="cake-signals">{items or "<li class=cake-muted>Aucun signal fort sur la période</li>"}</ul></div>')

    story_refs = {}
    if link_stories:
        for sid, s in (stories_lookup or {}).items():
            if s.get("status") == "open":
                for u in s.get("article_urls") or []:
                    story_refs.setdefault(u, f"story-ia-{sid}")
    cards = "".join(_event_card(e, letters[e["id"]], story_refs, link_stories) for e in listed)
    cards_html = (f'<div class="cake-h3">Fiches événements</div>{chart_info.info_html("cake_events")}{cards}' if cards
                  else '<p class="cake-muted">Aucun mouvement inter-couches sur 90 jours.</p>')
    return (CSS + chart_info.INFO_CSS + '<div class="cake-wrap">'
            + banner_html(data["events"], today, bn_label, market.get("synthetic"))
            + chart_info.info_html("cake")
            + f'<div class="cake-periods">{buttons}</div>{views}{cards_html}'
            + "".join(market.get("blocks_html") or [])
            + "</div>" + JS)


def cake_counts_html(articles: list[dict]) -> str:
    """Five stacked bands, energy at the bottom, with the article count of each layer."""
    counts = layer_counts(articles)
    if not counts:
        return ""
    meta = cake.layer_meta()
    top = max(counts.values())
    bands = ""
    for code in reversed(cake.layer_codes()):
        n = counts.get(code, 0)
        width = 8 + 92 * n / top
        bands += f"""
    <div class="cake-band">
      <span class="cake-band__name">{meta[code]['emoji']} {escape(meta[code]['label'])}</span>
      <span class="cake-band__bar"><span style="width:{width:.0f}%"></span></span>
      <span class="cake-band__count">{n}</span>
    </div>"""
    return chart_info.INFO_CSS + chart_info.info_html("cake_counts") + f"""
<style>
  .cake {{ background:#0c1222; border:1px solid #1a2540; border-radius:10px; padding:12px 14px; margin-bottom:14px; }}
  .cake-band {{ display:flex; align-items:center; gap:10px; padding:5px 0; font-size:13px; color:#c8d4ec; }}
  .cake-band__name {{ flex:0 0 140px; }}
  .cake-band__bar {{ flex:1; height:10px; background:#060810; border-radius:5px; overflow:hidden; }}
  .cake-band__bar span {{ display:block; height:100%; background:linear-gradient(90deg,#0047ab,#7eb8f7); border-radius:5px; }}
  .cake-band__count {{ flex:0 0 40px; text-align:right; color:#e8eef8; }}
  .cake-caption {{ font-size:11px; color:#5a6a8a; margin-top:6px; }}
</style>
<div class="cake">{bands}
  <div class="cake-caption">Articles par couche sur la période · un article peut couvrir jusqu'à 3 couches</div>
</div>"""
