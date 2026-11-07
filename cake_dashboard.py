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

import cake

PERIODS = (7, 30, 90)
MAX_ARROWS = 12   # per period view, highest scores first
COLORS = {"montee": "#f4a261", "descente": "#7eb8f7"}


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


def load_cake_market(client) -> dict | None:
    """Market and scarcity data of the section (Phase C); None until then."""
    return None


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
            parts.append(f'<text x="190" y="{y + 24}" class="cake-t-bn">goulot {escape(bottleneck["status"])}</text>')
        ex = extras.get(c) or {}
        if ex.get("gauge") is not None:
            gx, gw = 190, 110
            dash = ' stroke-dasharray="3 3"' if ex.get("rebuilding") else ""
            parts.append(f'<rect x="{gx}" y="{y + 36}" width="{gw}" height="8" rx="4" fill="#060810" stroke="#1a2540"{dash}/>'
                         f'<rect x="{gx}" y="{y + 36}" width="{gw * ex["gauge"]:.0f}" height="8" rx="4" fill="#e63946" opacity="0.8"/>'
                         f'<title>Rareté (z-score ramené à 0–1){" — normale en reconstruction" if ex.get("rebuilding") else ""}</title>')
        if ex.get("spark"):
            parts.append(_sparkline(ex["spark"], 320, y + 10, 120, band_h - 22, "#4488ff"))
            if ex.get("spark2"):
                parts.append(_sparkline(ex["spark2"], 320, y + 10, 120, band_h - 22, "#e9c46a"))
            chg = ex.get("change_pct")
            if chg is not None:
                color = "#3fb950" if chg >= 0 else "#e63946"
                parts.append(f'<text x="450" y="{y + 34}" class="cake-t-sub" fill="{color}" style="fill:{color}">'
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

    views, buttons = "", ""
    for p in PERIODS:
        since = (date.fromisoformat(today) - timedelta(days=p)).isoformat()
        p_events = sorted((e for e in listed if e["last_seen"] >= since),
                          key=lambda e: e.get("score") or 0, reverse=True)[:MAX_ARROWS]
        p_events.sort(key=lambda e: e["last_seen"])
        signals = strong_signals(data, stories_lookup, p, today)
        sig_counts = Counter(l for s in signals for l in s["layers"])
        art_counts = layer_counts([a for a in data["articles"] if a["published"] >= since])
        svg = cake_svg(p_events, letters, sig_counts, art_counts,
                       (market.get("layer_extras") or {}).get(p), bn, uid=f"-{p}")
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
    cards_html = (f'<div class="cake-h3">Fiches événements</div>{cards}' if cards
                  else '<p class="cake-muted">Aucun mouvement inter-couches sur 90 jours.</p>')
    return (CSS + '<div class="cake-wrap">'
            + banner_html(data["events"], today, bn_label, market.get("synthetic"))
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
    return f"""
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
