"""
AI layer cake — reading the AI news layer by layer (energy → chips →
infrastructure → models → applications). See SPECS_LAYER_CAKE.md.

This module holds the config loader, the article-level tagging helpers used
by main.py (classification prompt, output validation, relevance filter) and
the cross-layer events pipeline (actors, extraction, scoring, Telegram).
"""

import itertools
import json
import logging
import math
import re
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from html import escape
from pathlib import Path

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "data" / "cake_layers.json"

SCARCITY_VALUES = ("tension", "detente", "aucune")


@lru_cache(maxsize=1)
def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def layer_codes() -> list[str]:
    """Layer codes, bottom to top."""
    return [l["code"] for l in sorted(load_config()["layers"], key=lambda l: l["rank"])]


def layer_meta() -> dict[str, dict]:
    return {l["code"]: l for l in load_config()["layers"]}


# ---------------------------------------------------------------------------
# Article tagging (Phase A)
# ---------------------------------------------------------------------------

# Appended to the Groq classification prompt of the "ia" domain.
CAKE_PROMPT = """

- "layers": liste de 0 a 3 couches dont parle l'article, parmi ["energie", "puces", "infrastructure", "modeles", "applications"] :
  energie = production et reseau electriques, nucleaire, gaz, uranium, raccordement des data centers ;
  puces = conception et fabrication de semi-conducteurs, GPU, memoire HBM, fonderies, lithographie ;
  infrastructure = data centers, refroidissement, serveurs, cloud, location de GPU ;
  modeles = entrainement et publication de modeles d'IA, laboratoires d'IA ;
  applications = produits et services construits sur les modeles (assistants, agents, logiciels, robots).
  Liste vide si aucune ne s'applique (ex: debat ethique general).

- "cross_layer": true si l'article decrit un acteur qui ENTRE dans une couche ou il n'est pas encore actif (il construit, achete, concoit, produit, investit ou signe pour y operer) ; false pour une activite dans sa couche habituelle.
  Exemples true : "OpenAI construit sa propre centrale electrique", "Un energeticien publie un modele d'IA".
  Exemples false : "Nvidia presente son nouveau GPU", "OpenAI lance un nouveau modele".

- "scarcity": une valeur parmi ["tension", "detente", "aucune"]. "tension" si l'article decrit une penurie, un rationnement, un allongement des delais ou une capacite saturee dans une des couches de "layers" ; "detente" si une telle tension se resorbe (ex: "la penurie de HBM s'attenue") ; "aucune" sinon."""


def parse_cake_fields(result: dict) -> tuple[list[str], bool, str]:
    """Validate the cake keys of a Groq classification: unknown layer codes
    are dropped, at most 3 layers are kept, defaults are [] / False / 'aucune'."""
    codes = set(layer_codes())
    raw = result.get("layers")
    layers: list[str] = []
    if isinstance(raw, list):
        for code in raw:
            if isinstance(code, str) and code.strip().lower() in codes and code.strip().lower() not in layers:
                layers.append(code.strip().lower())
    layers = layers[:3]
    cross = result.get("cross_layer") is True or str(result.get("cross_layer")).lower() == "true"
    scarcity = str(result.get("scarcity") or "").strip().lower().replace("é", "e")
    if scarcity not in SCARCITY_VALUES:
        scarcity = "aucune"
    # A move or a shortage needs at least one layer to make sense
    if not layers:
        cross, scarcity = False, "aucune"
    return layers, cross, scarcity


@lru_cache(maxsize=None)
def _keyword_re(layer: str) -> re.Pattern | None:
    kws = load_config()["relevance_keywords"].get(layer) or []
    if not kws:
        return None
    # Matched at the start of a word, so plurals ("data centers") still match
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(k.lower()) for k in kws) + ")")


def matches_layer_keywords(text: str, layer: str) -> bool:
    """True if `text` mentions one of the relevance keywords of `layer`."""
    pattern = _keyword_re(layer)
    return bool(pattern and pattern.search(text.lower()))


# ---------------------------------------------------------------------------
# Cross-layer events (Phase B) — pure logic, tested in tests/test_cake_events.py
# ---------------------------------------------------------------------------

ENGAGEMENTS = ("annonce", "contrat", "mise_en_service")
ENGAGEMENT_LABEL = {"annonce": "Annonce", "contrat": "Contrat", "mise_en_service": "Mise en service"}
RELATIONS = ("fournit", "utilise", "partenaire", "investit")
DISTANCE_FACTOR = {1: 0.40, 2: 0.60, 3: 0.85, 4: 1.00}
ENGAGEMENT_FACTOR = {"annonce": 0.40, "contrat": 0.70, "mise_en_service": 1.00}


def normalize_name(name: str) -> str:
    """Same normalisation as main._topic_key, with accents folded ("Électricité" → "electricite")."""
    folded = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")


def actor_index(actors: list[dict]) -> dict[str, dict]:
    """normalized name or alias -> actor."""
    index: dict[str, dict] = {}
    for a in actors:
        for n in [a["name"], re.sub(r"\(.*?\)", "", a["name"])] + list(a.get("aliases") or []):
            key = normalize_name(n)
            if key:
                index.setdefault(key, a)
    return index


def resolve_actor(name: str, index: dict[str, dict]) -> dict | None:
    return index.get(normalize_name(name)) or index.get(normalize_name(re.sub(r"\(.*?\)", "", name or "")))


def locate_move(home_layers: list[str], target: str) -> tuple[str, str, int, bool] | None:
    """(origin_layer, direction, distance, tie) of an actor moving into `target`,
    or None when it is not a cross-layer move (target already a home layer, or
    unknown layers). The origin is the home layer closest to the target, so a
    move is never overstated; on a tie (one home layer above, one below) the
    lower one wins and the move is a climb."""
    rank = {l["code"]: l["rank"] for l in load_config()["layers"]}
    homes = [h for h in home_layers or [] if h in rank]
    if target not in rank or not homes or target in homes:
        return None
    best = min(abs(rank[h] - rank[target]) for h in homes)
    closest = sorted((h for h in homes if abs(rank[h] - rank[target]) == best), key=rank.get)
    origin = closest[0]
    direction = "montee" if rank[target] > rank[origin] else "descente"
    return origin, direction, best, len(closest) > 1


def event_score(distance: int, engagement: str, weight: float | None, novel: bool) -> tuple[float, dict]:
    """score = min(10, round(10 × (0.35·D + 0.30·E + 0.35·W) × N, 1)), and its breakdown."""
    cfg = load_config()["event_score"]
    d = DISTANCE_FACTOR.get(distance, 1.0)
    e = ENGAGEMENT_FACTOR.get(engagement, ENGAGEMENT_FACTOR["annonce"])
    w = weight or 0.0
    n = cfg["novelty_bonus"] if novel else 1.0
    score = min(10.0, round(10 * (0.35 * d + 0.30 * e + 0.35 * w) * n, 1))
    return score, {"D": d, "E": e, "W": round(w, 3), "N": n, "W_missing": weight is None}


def days_before(day: str, n: int) -> str:
    return (date.fromisoformat(day) - timedelta(days=n)).isoformat()


def is_novel(origin: str, target: str, events: list[dict], today: str) -> bool:
    """No integration event crossed the same origin → target pair recently."""
    since = days_before(today, load_config()["event_score"]["novelty_window_days"])
    return not any(
        e.get("event_type", "integration") == "integration"
        and e.get("origin_layer") == origin and e.get("target_layer") == target
        and (e.get("last_seen") or "") >= since
        for e in events
    )


def _clean_list(raw, limit: int = 5) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [" ".join(str(x).split())[:80] for x in raw if isinstance(x, str) and x.strip()][:limit]


def clean_extraction(raw) -> dict | None:
    """Validate one event returned by Groq; None if unusable."""
    if not isinstance(raw, dict):
        return None
    actor = " ".join(str(raw.get("actor") or "").split())
    target = str(raw.get("target_layer") or "").strip().lower()
    if not actor or target not in layer_codes():
        return None
    engagement = str(raw.get("engagement") or "").strip().lower()
    if engagement not in ENGAGEMENTS:
        engagement = "annonce"
    relations = []
    for r in raw.get("relations") or []:
        if isinstance(r, dict) and r.get("relation") in RELATIONS and r.get("from") and r.get("to"):
            relations.append({"from": str(r["from"])[:80], "relation": r["relation"], "to": str(r["to"])[:80]})
    homes = raw.get("actor_home_layers") if isinstance(raw.get("actor_home_layers"), list) else []
    return {
        "actor": actor[:80],
        "actor_home_layers": [l for l in homes if l in layer_codes()],
        "target_layer": target,
        "engagement": engagement,
        "what": " ".join(str(raw.get("what") or "").split())[:160],
        "bypassed": _clean_list(raw.get("bypassed")),
        "winners": _clean_list(raw.get("winners")),
        "losers": _clean_list(raw.get("losers")),
        "relations": relations[:5],
    }


def new_actor_from(ext: dict) -> dict:
    """An actor Groq named but the actor list does not know: kept for review,
    seeded with the home layers Groq gave for it."""
    return {"name": ext["actor"], "aliases": [], "home_layers": ext["actor_home_layers"],
            "needs_review": True, "weight": None}


def apply_extraction(ext: dict, url: str, actor: dict, events: list[dict], today: str) -> tuple[str, dict | None]:
    """Fold one extracted move into `events` (mutated). Returns (status, event):
    "not_cross_layer" when the target is already a home layer of the actor,
    "new" for a new event, "merged" when it joins an event of the same actor
    and target seen within merge_window_days, "progressed" when it also moves
    that event's engagement forward (annonce → contrat → mise_en_service)."""
    cfg = load_config()["event_score"]
    located = locate_move(actor.get("home_layers") or [], ext["target_layer"])
    if located is None:
        return "not_cross_layer", None
    origin, direction, distance, _ = located
    since = days_before(today, cfg["merge_window_days"])
    existing = next(
        (e for e in events
         if e.get("event_type", "integration") == "integration" and e.get("actor_name") == actor["name"]
         and e["target_layer"] == ext["target_layer"] and (e.get("last_seen") or "") >= since),
        None,
    )
    if existing is None:
        novel = is_novel(origin, ext["target_layer"], events, today)
        event = {
            "event_type": "integration", "actor_name": actor["name"], "actor_id": actor.get("id"),
            "origin_layer": origin, "target_layer": ext["target_layer"], "direction": direction,
            "distance": distance, "engagement": ext["engagement"],
            "engagement_history": [{"engagement": ext["engagement"], "date": today}],
            "what": ext["what"], "bypassed": ext["bypassed"], "winners": ext["winners"], "losers": ext["losers"],
            "article_urls": [url], "first_seen": today, "last_seen": today, "progressed_at": None,
            "notified_at": None,
        }
        events.append(event)
        status = "new"
    else:
        event = existing
        novel = (event.get("score_breakdown") or {}).get("N", 1) > 1  # novelty is judged at first sight
        if url not in event["article_urls"]:
            event["article_urls"] = event["article_urls"] + [url]
        event["last_seen"] = max(event["last_seen"], today)
        for key in ("bypassed", "winners", "losers"):
            if not event.get(key):
                event[key] = ext[key]
        status = "merged"
        if ENGAGEMENTS.index(ext["engagement"]) > ENGAGEMENTS.index(event["engagement"]):
            event["engagement"] = ext["engagement"]
            event["engagement_history"] = (event.get("engagement_history") or []) + [
                {"engagement": ext["engagement"], "date": today}]
            event["progressed_at"] = today
            if ext["what"]:
                event["what"] = ext["what"]
            status = "progressed"
    event["score"], event["score_breakdown"] = event_score(
        event["distance"], event["engagement"], actor.get("weight"), novel)
    event["is_major"] = event["score"] >= cfg["major_threshold"]
    return status, event


def needs_notification(event: dict) -> bool:
    """A major event is announced once, and once more per engagement progression."""
    if not event.get("is_major"):
        return False
    if not event.get("notified_at"):
        return True
    return bool(event.get("progressed_at")) and event["progressed_at"] > event["notified_at"][:10]


def promote_home_layers(events: list[dict], actors_by_name: dict[str, dict], today: str) -> list[dict]:
    """Once a move has been in service for home_layer_delay_days, its target
    becomes a home layer of the actor (later news about it is no longer a move).
    Returns the actors that changed."""
    cutoff = days_before(today, load_config()["event_score"]["home_layer_delay_days"])
    changed = []
    for e in events:
        if e.get("event_type", "integration") != "integration" or e.get("engagement") != "mise_en_service":
            continue
        reached = next((h["date"] for h in e.get("engagement_history") or [] if h["engagement"] == "mise_en_service"), None)
        actor = actors_by_name.get(e.get("actor_name"))
        if actor and reached and reached <= cutoff and e["target_layer"] not in actor["home_layers"]:
            actor["home_layers"] = actor["home_layers"] + [e["target_layer"]]
            changed.append(actor)
    return changed


# ---------------------------------------------------------------------------
# Actor weight (§5)
# ---------------------------------------------------------------------------

def size_component(revenue_usd_bn: float | None) -> float | None:
    if revenue_usd_bn is None or revenue_usd_bn < 0:
        return None
    return min(1.0, math.log10(1 + revenue_usd_bn) / math.log10(1 + 500))


def longevity_component(founded: int | None, year: int, parent_founded: int | None = None) -> float | None:
    """clip((year − founded) / 15, 0.2, 1); a spin-off takes 0.8 × its parent's."""
    if parent_founded:
        return 0.8 * min(1.0, max(0.2, (year - parent_founded) / 15))
    if not founded:
        return None
    return min(1.0, max(0.2, (year - founded) / 15))


def percentile_ranks(values: dict) -> dict:
    """key -> percentile rank in [0, 1] (ties share their average rank)."""
    if not values:
        return {}
    if len(values) == 1:
        return {k: 1.0 for k in values}
    ordered = sorted(values.items(), key=lambda kv: kv[1])
    ranks: dict = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = (i + j) / 2 / (len(ordered) - 1)
        i = j + 1
    return ranks


def pagerank(nodes: list, edges: dict[tuple, float], damping: float = 0.85, iterations: int = 60) -> dict:
    """PageRank over a weighted directed graph {(src, dst): weight}. Dangling
    nodes spread their rank evenly."""
    n = len(nodes)
    if not n:
        return {}
    out_weight: dict = defaultdict(float)
    incoming: dict = defaultdict(list)
    for (src, dst), w in edges.items():
        if w > 0 and src != dst:
            out_weight[src] += w
            incoming[dst].append((src, w))
    rank = {v: 1 / n for v in nodes}
    for _ in range(iterations):
        dangling = sum(rank[v] for v in nodes if not out_weight.get(v))
        rank = {
            v: (1 - damping) / n + damping * (
                dangling / n + sum(rank[s] * w / out_weight[s] for s, w in incoming.get(v, ())))
            for v in nodes
        }
    return rank


def comention_edges(article_actor_sets: list[set]) -> dict[tuple, float]:
    """Co-mention graph, both directions: weight = number of articles naming both actors."""
    edges: dict[tuple, float] = defaultdict(float)
    for names in article_actor_sets:
        for a, b in itertools.permutations(sorted(names), 2):
            edges[(a, b)] += 1
    return edges


def combine_weight(components: dict[str, float | None], cold_start: bool = False) -> tuple[float | None, str]:
    """Weighted mean over the known components (coefficients renormalised).
    Confidence is low when a component is missing or influence is a cold start."""
    coeffs = load_config()["actor_weight_coefficients"]
    known = {k: v for k, v in components.items() if v is not None and k in coeffs}
    if not known:
        return None, "low"
    total = sum(coeffs[k] for k in known)
    weight = sum(coeffs[k] * v for k, v in known.items()) / total
    confidence = "high" if len(known) == len(coeffs) and not cold_start else "low"
    return round(weight, 4), confidence


# ---------------------------------------------------------------------------
# Telegram formatting
# ---------------------------------------------------------------------------

def fmt_score(score: float) -> str:
    return f"{score:.1f}".replace(".", ",")


def major_event_message(event: dict, dashboard_url: str = "") -> str:
    meta = layer_meta()
    day = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    title = ("🎂 <b>Mouvement majeur — progression</b>" if event.get("notified_at")
             else f"🎂 <b>Mouvement majeur — {day}</b>")
    arrow = "⬆️" if event["direction"] == "montee" else "⬇️"
    novel = (event.get("score_breakdown") or {}).get("N", 1) > 1
    dist = event["distance"]
    lines = [
        title,
        f"{arrow} {escape(event['actor_name'])} : {meta[event['origin_layer']]['label']} → "
        f"{meta[event['target_layer']]['label']} ({dist} couche{'s' if dist > 1 else ''})",
    ]
    if event.get("what"):
        lines.append(f"📌 {escape(event['what'])}")
    lines.append(f"🔖 {ENGAGEMENT_LABEL[event['engagement']]} · Score {fmt_score(event['score'])}"
                 + (" (nouveau franchissement)" if novel else ""))
    if event.get("losers"):
        lines.append(f"📉 Perdants : {escape(', '.join(event['losers']))}")
    if event.get("winners"):
        lines.append(f"📈 Gagnants : {escape(', '.join(event['winners']))}")
    if event.get("article_urls"):
        lines.append(f'🔗 <a href="{escape(event["article_urls"][-1])}">Lire l\'article</a>')
    if dashboard_url:
        lines.append(f'📊 <a href="{dashboard_url.rstrip("/")}#cake">Voir le gâteau</a>')
    return "\n".join(lines)


def digest_line(touched: list[dict], bottleneck: str = "") -> str:
    """One line for the ia digest: the run's moves, plus the bottleneck state."""
    moves = [e for e in touched if e.get("event_type", "integration") == "integration"]
    if not moves and not bottleneck:
        return ""
    majors = sum(1 for e in moves if e.get("is_major"))
    line = f"🎂 Cake : {len(moves)} mouvement{'s' if len(moves) != 1 else ''} ({majors} majeur{'s' if majors != 1 else ''})"
    return line + (f" · goulot : {bottleneck}" if bottleneck else "")


# ---------------------------------------------------------------------------
# Supabase I/O and the daily events pipeline (Phase B)
# ---------------------------------------------------------------------------

ACTORS_MANUAL_PATH = ROOT / "data" / "actors_manual.json"
ACTOR_COLS = (
    "id, name, aliases, home_layers, auto_home_layers, home_sublayers, parent, parent_founded_year, "
    "founded_year, ticker, market_cap_slug, revenue_usd_bn, revenue_source, size, influence, longevity, "
    "implantation, weight, weight_breakdown, weight_confidence, needs_review"
)
EVENT_COLS = (
    "id, event_type, actor_id, origin_layer, target_layer, direction, distance, engagement, "
    "engagement_history, what, bypassed, winners, losers, article_urls, score, score_breakdown, "
    "is_major, first_seen, last_seen, progressed_at, notified_at"
)
EXTRACT_BATCH_SIZE = 10
EVENTS_LOOKBACK_DAYS = 180  # novelty window; the merge window (90 days) fits in it


def load_manual_actors() -> list[dict]:
    try:
        return json.loads(ACTORS_MANUAL_PATH.read_text(encoding="utf-8"))["actors"]
    except (OSError, ValueError, KeyError):
        return []


def fetch_all_rows(query_fn, page_size: int = 1000) -> list[dict]:
    """Page through a Supabase query past its 1000-row cap. `query_fn(start, end)` runs one page."""
    rows: list[dict] = []
    while True:
        page = query_fn(len(rows), len(rows) + page_size - 1).data or []
        if not page:
            return rows
        rows.extend(page)


def _effective_homes(manual_homes: list[str], auto_homes: list[str]) -> list[str]:
    return list(dict.fromkeys(list(manual_homes or []) + list(auto_homes or [])))


def sync_actors(client) -> list[dict] | None:
    """Push data/actors_manual.json into `actors` and return every actor row
    (manual ones plus those created from unknown names). home_layers = the
    manual layers + `auto_home_layers` (targets promoted after going into
    service), so editing the manual file never loses a promotion and vice
    versa. None if the table does not exist yet (cake events then skipped)."""
    try:
        existing = {r["name"]: r for r in client.table("actors").select(ACTOR_COLS).execute().data or []}
    except Exception as e:
        logging.warning(f"Cake: actors table unavailable ({e}) — run the Phase B migration. Skipping events.")
        return None
    rows = []
    for m in load_manual_actors():
        prev = existing.get(m["name"], {})
        row = {
            "name": m["name"],
            "aliases": m.get("aliases") or [],
            "home_layers": _effective_homes(m.get("home_layers"), prev.get("auto_home_layers")),
            "home_sublayers": m.get("home_sublayers") or [],
            "parent": m.get("parent"),
            "parent_founded_year": m.get("parent_founded_year"),
            "founded_year": m.get("founded_year"),
            "ticker": m.get("ticker"),
            "market_cap_slug": m.get("market_cap_slug"),
            "needs_review": bool(m.get("needs_review", True)),
        }
        if m.get("revenue"):
            row["revenue_usd_bn"] = m["revenue"]["usd_bn"]
            row["revenue_source"] = m["revenue"].get("source_url")
        rows.append(row)
    try:
        for i in range(0, len(rows), 200):
            client.table("actors").upsert(rows[i:i + 200], on_conflict="name").execute()
        return client.table("actors").select(ACTOR_COLS).execute().data or []
    except Exception as e:
        logging.error(f"Cake: actors sync failed: {e}")
        return None


def create_actor(client, actor: dict) -> dict:
    row = {k: actor[k] for k in ("name", "aliases", "home_layers", "needs_review")}
    saved = client.table("actors").upsert(row, on_conflict="name").execute().data or [row]
    return {**actor, **saved[0]}


def fetch_events(client, actors_by_id: dict[int, dict], days: int = EVENTS_LOOKBACK_DAYS) -> list[dict]:
    since = (date.today() - timedelta(days=days)).isoformat()
    rows = fetch_all_rows(lambda a, b: client.table("layer_events").select(EVENT_COLS)
                          .gte("last_seen", since).order("id").range(a, b).execute())
    for r in rows:
        r["actor_name"] = (actors_by_id.get(r.get("actor_id")) or {}).get("name")
    return rows


_EVENT_WRITE_COLS = (
    "event_type", "actor_id", "origin_layer", "target_layer", "direction", "distance", "engagement",
    "engagement_history", "what", "bypassed", "winners", "losers", "article_urls", "score",
    "score_breakdown", "is_major", "first_seen", "last_seen", "progressed_at", "notified_at",
)


def save_events(client, events: list[dict]) -> None:
    """Insert new events (no id yet) and update the others; ids are written back."""
    for e in events:
        row = {k: e.get(k) for k in _EVENT_WRITE_COLS}
        try:
            if e.get("id"):
                client.table("layer_events").update(row).eq("id", e["id"]).execute()
            else:
                saved = client.table("layer_events").insert(row).execute().data or []
                if saved:
                    e["id"] = saved[0]["id"]
        except Exception as ex:
            logging.error(f"Cake: saving event {e.get('actor_name')} → {e.get('target_layer')} failed: {ex}")


def mark_notified(client, events: list[dict]) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for e in events:
        e["notified_at"] = now
        if e.get("id"):
            try:
                client.table("layer_events").update({"notified_at": now}).eq("id", e["id"]).execute()
            except Exception as ex:
                logging.error(f"Cake: marking event {e['id']} notified failed: {ex}")


def _extract_prompt(items: list[dict], actor_names: list[str]) -> str:
    return (
        "Each article below may describe a company or organisation ENTERING a layer of the AI stack "
        "where it is not active yet (it builds, buys, designs, produces, invests or signs to operate there).\n"
        "Layers, bottom to top: energie (power generation, grid, nuclear, gas, uranium), puces "
        "(semiconductor design and manufacturing, GPUs, HBM memory, foundries, lithography), "
        "infrastructure (data centers, cooling, servers, cloud, GPU rental), modeles (training and "
        "releasing AI models), applications (products built on models: assistants, agents, software, robots).\n\n"
        "For each article return either null or ONE event:\n"
        "- actor: the organisation that moves, reusing a name of KNOWN_ACTORS exactly when it is one of them\n"
        "- actor_home_layers: the layers where that actor ALREADY operates (1 to 4 codes)\n"
        "- target_layer: the layer it enters (one code)\n"
        "- engagement: \"annonce\" (intention, press release, rumour), \"contrat\" (signed deal, firm order, "
        "investment paid, closed acquisition) or \"mise_en_service\" (product shipped, plant connected, model "
        "available, factory producing)\n"
        "- what: French, 15 words max, what happens\n"
        "- bypassed: French, who is short-circuited by the move (0-3 items)\n"
        "- winners / losers: French, who gains / loses from it (0-3 items each)\n"
        "- relations: [{\"from\": <org>, \"relation\": \"fournit\"|\"utilise\"|\"partenaire\"|\"investit\", \"to\": <org>}] "
        "between the organisations named (0-3 items)\n"
        "Return null when the article only describes activity in the actor's usual layer "
        "(a chipmaker releasing a GPU, a lab releasing a model) or no clear actor moves.\n\n"
        "Return JSON: {\"articles\": [{\"id\": <id>, \"event\": null | {...}}]}\n\n"
        f"KNOWN_ACTORS = {json.dumps(actor_names, ensure_ascii=False)}\n\n"
        f"ARTICLES = {json.dumps(items, ensure_ascii=False)}"
    )


async def extract_layer_events(articles: list, groq_client, model: str, actor_names: list[str],
                               retries: int = 6, on_usage=None, batch_pause: float = 3.0) -> list[tuple]:
    """One Groq call per batch of EXTRACT_BATCH_SIZE cross-layer articles.
    Returns [(article, cleaned event or None)]; a failed batch yields Nones."""
    import asyncio
    client = groq_client.with_options(max_retries=retries)
    out: list[tuple] = []
    for start in range(0, len(articles), EXTRACT_BATCH_SIZE):
        batch = articles[start:start + EXTRACT_BATCH_SIZE]
        items = [{"id": i, "title": a.title, "description": (a.description or "")[:300], "source": a.source}
                 for i, a in enumerate(batch)]
        found: dict[int, dict | None] = {}
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _extract_prompt(items, actor_names)}],
                temperature=0.1,
                max_tokens=4000,
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )
            if on_usage:
                on_usage(resp)
            for row in json.loads(resp.choices[0].message.content).get("articles", []):
                i = row.get("id")
                if isinstance(i, int) and 0 <= i < len(batch):
                    found[i] = clean_extraction(row.get("event"))
        except Exception as e:
            logging.warning(f"Cake: event extraction failed for a batch of {len(batch)}: {e}")
        out.extend((a, found.get(i)) for i, a in enumerate(batch))
        if start + EXTRACT_BATCH_SIZE < len(articles):
            await asyncio.sleep(batch_pause)
    return out


async def process_layer_events(articles: list, client, groq_client, model: str, retries: int = 6,
                               on_usage=None, today: str | None = None) -> list[dict]:
    """Daily step, after classification: turn this run's cross_layer articles
    into layer events (new, merged or progressed), save them and their
    relations, promote home layers. Returns the events touched by this run."""
    today = today or date.today().isoformat()
    if client is None:
        return []
    actors = sync_actors(client)
    if actors is None:
        return []
    if date.fromisoformat(today).weekday() == 0 or any(a.get("weight") is None for a in actors):
        refresh_actor_weights(client, actors, refresh_revenue=date.fromisoformat(today).weekday() == 0)

    try:
        by_id = {a["id"]: a for a in actors}
        events = fetch_events(client, by_id)
    except Exception as e:
        logging.warning(f"Cake: layer_events table unavailable ({e}) — run the Phase B migration.")
        return []

    cross = [a for a in articles if getattr(a, "domain", "ia") == "ia" and getattr(a, "cross_layer", False)]
    touched: dict[int, dict] = {}
    relations: list[dict] = []
    if cross:
        extracted = await extract_layer_events(cross, groq_client, model, [a["name"] for a in actors],
                                               retries=retries, on_usage=on_usage)
        index = actor_index(actors)
        for art, ext in extracted:
            if not ext:
                continue
            actor = resolve_actor(ext["actor"], index)
            if actor is None:
                if not ext["actor_home_layers"]:
                    logging.info(f"Cake: unknown actor {ext['actor']!r} without home layers — skipped")
                    continue
                actor = create_actor(client, new_actor_from(ext))
                actors.append(actor)
                index = actor_index(actors)
                logging.info(f"Cake: new actor {actor['name']!r} (needs review), home layers {actor['home_layers']}")
            status, event = apply_extraction(ext, art.url, actor, events, today)
            if event is None:
                logging.info(f"Cake: not a cross-layer move ({actor['name']} → {ext['target_layer']}, "
                             f"home layers {actor['home_layers']}): {art.title[:80]}")
                continue
            touched[id(event)] = event
            relations += [{"from_actor": r["from"], "relation": r["relation"], "to_actor": r["to"],
                           "article_url": art.url, "seen_at": today} for r in ext["relations"]]
            logging.info(f"Cake: {status} — {actor['name']}: {event['origin_layer']} → {event['target_layer']} "
                         f"({event['engagement']}, score {event['score']})")
    save_events(client, list(touched.values()))
    if relations:
        try:
            client.table("actor_relations").insert(relations).execute()
        except Exception as e:
            logging.warning(f"Cake: saving actor relations failed: {e}")

    by_name = {a["name"]: a for a in actors}
    for actor in promote_home_layers(events, by_name, today):
        manual = next((m for m in load_manual_actors() if m["name"] == actor["name"]), {})
        auto = [l for l in actor["home_layers"] if l not in (manual.get("home_layers") or [])]
        try:
            client.table("actors").update({"home_layers": actor["home_layers"], "auto_home_layers": auto}) \
                .eq("id", actor["id"]).execute()
            logging.info(f"Cake: {actor['name']} now operates in {actor['home_layers']}")
        except Exception as e:
            logging.warning(f"Cake: promoting home layers of {actor['name']} failed: {e}")
    return list(touched.values())


def refresh_actor_weights(client, actors: list[dict], refresh_revenue: bool = False,
                          influence: dict[str, float] | None = None) -> None:
    """Recompute the four weight components and W of every actor (§5), and
    store them. `influence` (actor name -> median abnormal return, Phase C)
    replaces the cold-start influence for the actors it covers."""
    from country_indicators import _cmc_usd_bn, load_indicators

    year = date.today().year
    # Listed actors: trailing-12-month revenue, weekly (or when never fetched)
    for a in actors:
        if a.get("market_cap_slug") and (refresh_revenue or a.get("revenue_usd_bn") is None):
            try:
                a["revenue_usd_bn"] = _cmc_usd_bn(a["market_cap_slug"], "revenue")
                a["revenue_source"] = f"https://companiesmarketcap.com/{a['market_cap_slug']}/revenue/"
            except Exception as e:
                logging.warning(f"Cake: revenue of {a['name']} unavailable: {e}")

    index = actor_index(actors)
    # Influence, cold start: percentile of market cap / valuation (country indicators)
    values = {}
    for row in (load_indicators().get("company_valuations") or {}).get("rows", []):
        name = row["name"].split(" ", 1)[-1] if not row["name"][:1].isascii() else row["name"]
        actor = resolve_actor(name, index)
        if actor:
            values[actor["name"]] = row["value"]
    cold = percentile_ranks(values)
    full = percentile_ranks(influence or {})

    # Implantation: PageRank of the co-mention graph over 180 days of article topics
    since = (date.today() - timedelta(days=EVENTS_LOOKBACK_DAYS)).isoformat()
    try:
        rows = fetch_all_rows(lambda a, b: client.table("articles").select("url, topics").eq("domain", "ia")
                              .gte("published", since).order("url").range(a, b).execute())
    except Exception as e:
        logging.warning(f"Cake: articles for implantation unavailable: {e}")
        rows = []
    sets = []
    for r in rows:
        names = {act["name"] for t in r.get("topics") or [] if (act := resolve_actor(t, index))}
        if len(names) >= 2:
            sets.append(names)
    names_all = [a["name"] for a in actors]
    implantation = percentile_ranks(pagerank(names_all, comention_edges(sets))) if sets else {}

    updates = []
    for a in actors:
        comps = {
            "size": size_component(a.get("revenue_usd_bn")),
            "influence": full.get(a["name"], cold.get(a["name"])),
            "longevity": longevity_component(a.get("founded_year"), year, a.get("parent_founded_year")),
            "implantation": implantation.get(a["name"]),
        }
        cold_start = a["name"] not in full
        weight, confidence = combine_weight(comps, cold_start=cold_start)
        a.update(comps, weight=weight, weight_confidence=confidence,
                 weight_breakdown={**{k: (round(v, 3) if v is not None else None) for k, v in comps.items()},
                                   "influence_cold_start": cold_start})
        updates.append({k: a.get(k) for k in (
            "name", "home_layers", "revenue_usd_bn", "revenue_source", "size", "influence", "longevity",
            "implantation", "weight", "weight_breakdown", "weight_confidence")})
    try:
        for i in range(0, len(updates), 200):
            client.table("actors").upsert(updates[i:i + 200], on_conflict="name").execute()
        logging.info(f"Cake: weights refreshed for {len(updates)} actors "
                     f"({len(sets)} co-mention articles, {len(values)} valuations)")
    except Exception as e:
        logging.error(f"Cake: saving actor weights failed: {e}")


def review_reminder_lines(client) -> list[str]:
    """Monday reminder: actors created from unknown names, and manual actors still to validate."""
    manual = load_manual_actors()
    lines = []
    pending = sum(1 for m in manual if m.get("needs_review", True))
    if pending:
        lines.append(f"• <b>{pending} acteurs</b> de <code>data/actors_manual.json</code> à valider (home_layers, panier)")
    if client is not None:
        try:
            names = {m["name"] for m in manual}
            rows = client.table("actors").select("name, home_layers").eq("needs_review", True).execute().data or []
            new = [r for r in rows if r["name"] not in names]
            if new:
                lines.append("• Nouveaux acteurs détectés : " + ", ".join(
                    f"{escape(r['name'])} ({'/'.join(r['home_layers'] or []) or '?'})" for r in new[:15]))
        except Exception as e:
            logging.warning(f"Cake: actors reminder unavailable: {e}")
    return lines
