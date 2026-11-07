"""
AI layer cake — market and scarcity signals (Phase C, see SPECS_LAYER_CAKE.md §6).

- Daily closes of every actor ticker and the equal-weight basket of each layer
  (yfinance, unofficial: failures are logged and never block the pipeline).
- Token price index: median $/M tokens of the 20 most used OpenRouter models.
- Scarcity balance per layer and day, its z-score against the layer's own last
  90 days, signalled / confirmed bottlenecks and their migrations.
- Actor influence: median absolute abnormal return around the actor's moves.

Everything lands in `market_data` and `layer_daily`; the dashboard reads them back.
"""

import hashlib
import logging
import re
import statistics
from datetime import date, datetime, timedelta, timezone

import requests

import cake

BENCHMARK = "QQQ"
SOURCE_YF = "yfinance"
TOKEN_IN, TOKEN_OUT = "TOKEN_PRICE_IDX_IN", "TOKEN_PRICE_IDX_OUT"
PRICE_HISTORY_DAYS = 400
_CURRENCY_BY_SUFFIX = {"PA": "EUR", "DE": "EUR", "AS": "EUR", "MI": "EUR", "L": "GBp", "SW": "CHF",
                       "HK": "HKD", "KS": "KRW", "T": "JPY", "TW": "TWD", "SS": "CNY"}


def basket_symbol(layer: str) -> str:
    return f"CAKE_{layer.upper()}"


def baskets() -> dict[str, list[str]]:
    return {k: v for k, v in (cake.load_config().get("baskets") or {}).items() if not k.startswith("_")}


def _currency(ticker: str) -> str:
    return _CURRENCY_BY_SUFFIX.get(ticker.rsplit(".", 1)[-1], "USD") if "." in ticker else "USD"


# ---------------------------------------------------------------------------
# Pure computations (tested in tests/test_cake_market.py)
# ---------------------------------------------------------------------------

def basket_returns(closes: dict[str, dict[str, float]], tickers: list[str]) -> dict[str, float]:
    """Equal-weight daily return of a basket: mean of the members' returns on
    each date where at least one member traded. `closes` = ticker -> {day: close}.
    Currencies do not matter, only returns."""
    by_day: dict[str, list[float]] = {}
    for t in tickers:
        series = sorted((closes.get(t) or {}).items())
        for (d0, p0), (d1, p1) in zip(series, series[1:]):
            if p0:
                by_day.setdefault(d1, []).append(p1 / p0 - 1)
    return {d: sum(v) / len(v) for d, v in sorted(by_day.items())}


def chain_index(returns: dict[str, float], start_value: float = 100.0, after: str = "") -> dict[str, float]:
    """Compound daily returns (only those dated after `after`) from start_value."""
    out, value = {}, start_value
    for d, r in sorted(returns.items()):
        if d > after:
            value *= 1 + r
            out[d] = value
    return out


def value_at(series: list[tuple[str, float]], day: str) -> tuple[str, float] | None:
    """Last (date, value) on or before `day` of a date-sorted series."""
    best = None
    for d, v in series:
        if d > day:
            break
        best = (d, v)
    return best


def change_pct(series: list[tuple[str, float]], day: str, days: int) -> float | None:
    """% change of a series between `days` days before `day` and `day`."""
    end = value_at(series, day)
    start = value_at(series, (date.fromisoformat(day) - timedelta(days=days)).isoformat())
    if not end or not start or not start[1] or start[0] == end[0]:
        return None
    return (end[1] / start[1] - 1) * 100


def abnormal_return(series: list[tuple[str, float]], bench: list[tuple[str, float]], day: str) -> float | None:
    """J0–J+1 return around `day` (from the last close before it to the second
    close from it) minus the benchmark's over the same window."""
    def window(s):
        before = [v for d, v in s if d < day]
        after = [v for d, v in s if d >= day]
        if not before or len(after) < 2 or not before[-1]:
            return None
        return after[1] / before[-1] - 1
    r, rb = window(series), window(bench)
    if r is None or rb is None:
        return None
    return r - rb


def daily_layer_counts(articles: list[dict]) -> dict[str, dict[str, list[float]]]:
    """layer -> day -> [articles, tension, detente]. Articles classified before
    the `scarcity` key (None) fall back on the scarcity vocabulary, counted as a
    weak tension (0.5)."""
    vocab = [w.lower() for w in cake.load_config().get("scarcity_vocabulary") or []]
    out: dict[str, dict[str, list[float]]] = {}
    for a in articles:
        day = (a.get("published") or "")[:10]
        scarcity = a.get("scarcity")
        if scarcity is None:
            text = f"{a.get('title') or ''} {a.get('description') or ''}".lower()
            tension, detente = (0.5 if any(w in text for w in vocab) else 0.0), 0.0
        else:
            tension, detente = float(scarcity == "tension"), float(scarcity == "detente")
        for layer in a.get("layers") or []:
            row = out.setdefault(layer, {}).setdefault(day, [0.0, 0.0, 0.0])
            row[0] += 1
            row[1] += tension
            row[2] += detente
    return out


def scarcity_series(counts: dict[str, list[float]], days: list[str], smooth: int = 7,
                    baseline: int = 90, min_baseline: int = 30) -> dict[str, dict]:
    """For each day: the 7-day balance (Σtension − Σdetente) / Σarticles and its
    z-score against the previous `baseline` days of the same layer (None until
    `min_baseline` days of baseline exist, or when the baseline is flat)."""
    balance: dict[str, float | None] = {}
    for d in days:
        window = [(date.fromisoformat(d) - timedelta(days=k)).isoformat() for k in range(smooth)]
        n = sum(counts.get(w, [0, 0, 0])[0] for w in window)
        t = sum(counts.get(w, [0, 0, 0])[1] for w in window)
        de = sum(counts.get(w, [0, 0, 0])[2] for w in window)
        balance[d] = (t - de) / n if n else None
    out = {}
    for i, d in enumerate(days):
        prev = [balance[p] for p in days[max(0, i - baseline):i] if balance[p] is not None]
        z = None
        if balance[d] is not None and len(prev) >= min_baseline:
            sd = statistics.pstdev(prev)
            if sd > 0:
                z = (balance[d] - statistics.fmean(prev)) / sd
        c = counts.get(d, [0, 0, 0])
        out[d] = {"article_count": int(c[0]), "tension_count": round(c[1], 1), "detente_count": round(c[2], 1),
                  "scarcity_balance": balance[d], "scarcity_z": z}
    return out


def gauge(z: float | None) -> float | None:
    """z ≤ 0 → 0, z ≥ 3 → 1."""
    return None if z is None else max(0.0, min(1.0, z / 3))


def signalled_layer(z_by_layer: dict[str, list[float | None]], rebuilding: set[str]) -> str | None:
    """Layer with the highest z today, provided it stayed above z_threshold for
    the last min_days days and its baseline is not being rebuilt.
    `z_by_layer[layer]` = z of the last days, oldest first, today last."""
    cfg = cake.load_config()["bottleneck"]
    today = {l: zs[-1] for l, zs in z_by_layer.items() if zs and zs[-1] is not None and l not in rebuilding}
    if not today:
        return None
    top = max(today, key=today.get)
    recent = z_by_layer[top][-cfg["min_days"]:]
    if len(recent) == cfg["min_days"] and all(z is not None and z > cfg["z_threshold"] for z in recent):
        return top
    return None


def price_confirmation(layer: str, series: dict[str, list[tuple[str, float]]], day: str) -> str | None:
    """A price signal of the same layer pointing the same way over the window
    (§6.4 step 3), described in French; None if none."""
    cfg = cake.load_config()["bottleneck"]["price_confirmation"]
    window = cfg["window_days"]
    label = cake.layer_meta()[layer]["label"].lower()

    def outperf():
        b = change_pct(series.get(basket_symbol(layer), []), day, window)
        q = change_pct(series.get(BENCHMARK, []), day, window)
        if b is not None and q is not None and b - q > cfg["basket_outperformance_pts"]:
            return f"panier {label} {b - q:+.1f} pts vs {BENCHMARK} sur {window} j".replace(".", ",")
        return None

    if layer == "energie":
        for sym, name in (("ELEC_US", "États-Unis"), ("ELEC_EU", "Europe")):
            c = change_pct(series.get(sym, []), day, window)
            if c is not None and c > cfg["electricity_rise_pct"]:
                return f"prix de l'électricité ({name}) {c:+.0f} % sur {window} j"
        return outperf()
    if layer == "puces":
        return outperf()
    if layer == "infrastructure":
        c = change_pct(series.get("GPU_HOUR_H100", []), day, window)
        if c is not None and c > 0:
            return f"prix du GPU-heure {c:+.0f} % sur {window} j"
        return outperf()
    if layer == "modeles":
        for sym, name in ((TOKEN_OUT, "sortie"), (TOKEN_IN, "entrée")):
            c = change_pct(series.get(sym, []), day, window)
            if c is not None and c > 0:
                return f"indice de prix des tokens ({name}) {c:+.0f} % sur {window} j"
    return None  # applications: no reliable price signal in v1


def migration(confirmed_by_day: list[tuple[str, str | None]], hold_days: int) -> tuple[str, str] | None:
    """(previous, new) when the confirmed bottleneck moved to a new layer and
    held there for the last `hold_days` days. `confirmed_by_day` oldest first."""
    if len(confirmed_by_day) < hold_days + 1:
        return None
    recent = [l for _, l in confirmed_by_day[-hold_days:]]
    new = recent[0]
    if new is None or any(l != new for l in recent):
        return None
    previous = next((l for _, l in reversed(confirmed_by_day[:-hold_days]) if l is not None), None)
    return (previous, new) if previous and previous != new else None


def sources_fingerprint(sources: list[dict], layer: str) -> str:
    """Hash of the enabled ia sources that can feed a layer: the layer's own
    cake sources plus the generic ones (no cake_layer)."""
    names = sorted(s["name"] for s in sources if s.get("domain", "ia") == "ia" and s.get("enabled", True)
                   and s.get("cake_layer") in (None, layer))
    return hashlib.sha1("|".join(names).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def load_series(client, symbols: list[str], since: str) -> dict[str, list[tuple[str, float]]]:
    """symbol -> [(day, value)] from market_data, one value per day (the last)."""
    out: dict[str, dict[str, float]] = {}
    for i in range(0, len(symbols), 50):
        rows = cake.fetch_all_rows(lambda a, b: client.table("market_data").select("symbol, value, collected_at")
                                   .eq("domain", "ia").in_("symbol", symbols[i:i + 50]).gte("collected_at", since)
                                   .order("collected_at").range(a, b).execute())
        for r in rows:
            out.setdefault(r["symbol"], {})[r["collected_at"][:10]] = float(r["value"])
    return {s: sorted(v.items()) for s, v in out.items()}


def _insert_market_rows(client, rows: list[dict]) -> int:
    """Insert rows whose (symbol, day) is not stored yet. Returns the count inserted."""
    if not rows:
        return 0
    since = min(r["collected_at"] for r in rows)
    symbols = sorted({r["symbol"] for r in rows})
    have = set()
    for i in range(0, len(symbols), 50):
        for r in cake.fetch_all_rows(lambda a, b: client.table("market_data").select("symbol, collected_at")
                                     .eq("domain", "ia").in_("symbol", symbols[i:i + 50])
                                     .gte("collected_at", since).order("collected_at").range(a, b).execute()):
            have.add((r["symbol"], r["collected_at"][:10]))
    new = [r for r in rows if (r["symbol"], r["collected_at"][:10]) not in have]
    for i in range(0, len(new), 500):
        client.table("market_data").insert(new[i:i + 500]).execute()
    return len(new)


def _row(symbol: str, label: str, day: str, value: float, unit: str, variation: float | None, source: str) -> dict:
    return {"domain": "ia", "symbol": symbol, "label": label, "value": round(value, 6), "unit": unit,
            "variation_pct": None if variation is None else round(variation, 4), "source": source,
            "collected_at": f"{day}T00:00:00+00:00"}


# ---------------------------------------------------------------------------
# Daily collection
# ---------------------------------------------------------------------------

def tracked_tickers() -> list[str]:
    actors = [a["ticker"] for a in cake.load_manual_actors() if a.get("ticker")]
    members = [t for ts in baskets().values() for t in ts]
    return sorted(set(actors) | set(members) | {BENCHMARK})


def download_closes(tickers: list[str], period: str) -> dict[str, dict[str, float]]:
    """ticker -> {day: close} via yfinance; tickers without data are logged and left out."""
    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    df = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False,
                     group_by="column", threads=True)
    closes = {}
    for t in tickers:
        try:
            s = df["Close"][t].dropna()
        except KeyError:
            s = None
        if s is None or s.empty:
            logging.warning(f"Cake market: no price for {t} — left out of its basket")
            continue
        closes[t] = {d.strftime("%Y-%m-%d"): float(v) for d, v in s.items()}
    return closes


def collect_prices(client, backfill: bool = False) -> None:
    """Store the closes of every tracked ticker and each layer's basket index.
    `backfill` downloads a year of history (first run: `python main.py --backfill-baskets`)."""
    closes = download_closes(tracked_tickers(), "1y" if backfill else "15d")
    if not closes:
        logging.error("Cake market: yfinance returned nothing — keeping the last stored prices")
        return
    today = date.today().isoformat()
    rows = []
    for t, series in closes.items():
        items = sorted(series.items())
        for (d0, p0), (d1, p1) in zip([(None, None)] + items, items):
            if d1 >= today:  # today's bar is not a close yet
                continue
            rows.append(_row(t, t, d1, p1, _currency(t), (p1 / p0 - 1) * 100 if p0 else None, SOURCE_YF))
    meta = cake.layer_meta()
    for layer, members in baskets().items():
        sym = basket_symbol(layer)
        stored = load_series(client, [sym], (date.today() - timedelta(days=PRICE_HISTORY_DAYS)).isoformat()).get(sym, [])
        last_day, last_value = stored[-1] if stored else ("", 100.0)
        present = [m for m in members if m in closes]
        rets = {d: r for d, r in basket_returns(closes, present).items() if d < today}
        if not stored and rets:
            last_day = min(d for m in present for d in closes[m])  # base 100 on the first day of history
            rows.append(_row(sym, f"Panier {meta[layer]['label']}", last_day, 100.0, "index", None, SOURCE_YF))
        prev = last_value
        for d, v in chain_index(rets, last_value, after=last_day).items():
            rows.append(_row(sym, f"Panier {meta[layer]['label']}", d, v, "index", (v / prev - 1) * 100, SOURCE_YF))
            prev = v
    logging.info(f"Cake market: {_insert_market_rows(client, rows)} new price rows ({len(closes)} tickers)")


def fetch_token_price_index() -> tuple[float, float, list[str]]:
    """(median input $/M tokens, median output $/M tokens, model ids) of the 20
    most used paid models of the previous day on OpenRouter. The ranking comes
    from the data embedded in openrouter.ai/rankings (no public API), the
    prices from the public /api/v1/models."""
    page = requests.get("https://openrouter.ai/rankings", timeout=30,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; Veille-IA)"}).text.replace('\\"', '"')
    usage: dict[str, int] = {}
    for m in re.finditer(r'\{[^{}]*"model_permaslug":"([^"]+)"[^{}]*\}', page):
        obj = m.group(0)
        if '"variant":"free"' in obj:
            continue
        tok = sum(int(x) for x in re.findall(r'"total_(?:prompt|completion)_tokens":(\d+)', obj))
        usage[m.group(1)] = usage.get(m.group(1), 0) + tok
    if not usage:
        raise RuntimeError("OpenRouter rankings layout changed: no model usage found")
    models = requests.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
    by_slug = {}
    for m in models:
        by_slug[m.get("canonical_slug") or m["id"]] = m
        by_slug.setdefault(m["id"], m)
    top = [s for s, _ in sorted(usage.items(), key=lambda kv: -kv[1]) if s in by_slug][:20]
    prices_in, prices_out = [], []
    for s in top:
        p = by_slug[s].get("pricing") or {}
        try:
            pin, pout = float(p["prompt"]) * 1e6, float(p["completion"]) * 1e6
        except (KeyError, TypeError, ValueError):
            continue
        if pin > 0 or pout > 0:
            prices_in.append(pin)
            prices_out.append(pout)
    if len(prices_out) < 5:
        raise RuntimeError(f"OpenRouter: only {len(prices_out)} priced models among the top ones")
    return statistics.median(prices_in), statistics.median(prices_out), top


def collect_token_prices(client) -> None:
    today = date.today().isoformat()
    try:
        pin, pout, top = fetch_token_price_index()
    except Exception as e:
        logging.warning(f"Cake market: token price index unavailable: {e}")
        return
    rows = [_row(TOKEN_IN, "Indice prix tokens — entrée (médiane top 20 OpenRouter)", today, pin, "USD/M tokens", None, "openrouter"),
            _row(TOKEN_OUT, "Indice prix tokens — sortie (médiane top 20 OpenRouter)", today, pout, "USD/M tokens", None, "openrouter")]
    _insert_market_rows(client, rows)
    logging.info(f"Cake market: token price index {pin:.2f} / {pout:.2f} $/M tokens ({len(top)} models)")


def yf_revenue_usd_bn(ticker: str) -> float | None:
    """Trailing revenue from Yahoo Finance, converted to USD bn — fallback for
    listed actors without a companiesmarketcap slug."""
    import yfinance as yf
    info = yf.Ticker(ticker).info
    revenue, cur = info.get("totalRevenue"), (info.get("financialCurrency") or "USD").upper()
    if not revenue:
        return None
    rate = 1.0
    if cur != "USD":
        fx = download_closes([f"{cur}USD=X"], "5d").get(f"{cur}USD=X")
        if not fx:
            return None
        rate = fx[max(fx)]
    return revenue * rate / 1e9


# ---------------------------------------------------------------------------
# Actor influence (§5.2)
# ---------------------------------------------------------------------------

def influence_values(client, actors: list[dict]) -> dict[str, float]:
    """actor name -> median absolute abnormal return J0–J+1 around its moves
    (layer events) and, for history, its hot articles; own stock vs QQQ when
    listed, and always the baskets of the layers involved. Actors with fewer
    than 3 observation days are left out (cold start)."""
    since = (date.today() - timedelta(days=PRICE_HISTORY_DAYS)).isoformat()
    symbols = sorted({a["ticker"] for a in actors if a.get("ticker")} | {basket_symbol(l) for l in baskets()} | {BENCHMARK})
    series = load_series(client, symbols, since)
    bench = series.get(BENCHMARK) or []
    if not bench:
        return {}
    index = cake.actor_index(actors)
    obs: dict[str, dict[str, set[str]]] = {}  # actor -> day -> layers involved

    by_id = {a["id"]: a for a in actors if a.get("id")}
    try:
        events = cake.fetch_all_rows(lambda a, b: client.table("layer_events").select(
            "actor_id, origin_layer, target_layer, first_seen, progressed_at").eq("event_type", "integration")
            .gte("first_seen", since).order("id").range(a, b).execute())
    except Exception:
        events = []
    for e in events:
        actor = by_id.get(e.get("actor_id"))
        if actor:
            for d in filter(None, (e["first_seen"], e.get("progressed_at"))):
                obs.setdefault(actor["name"], {}).setdefault(d, set()).update(
                    l for l in (e["origin_layer"], e["target_layer"]) if l)
    try:
        hot = cake.fetch_all_rows(lambda a, b: client.table("articles").select("published, topics").eq("domain", "ia")
                                  .eq("hot_topic", True).gte("published", since).order("url").range(a, b).execute())
    except Exception:
        hot = []
    for r in hot:
        for t in r.get("topics") or []:
            actor = cake.resolve_actor(t, index)
            if actor:
                obs.setdefault(actor["name"], {}).setdefault(r["published"][:10], set()).update(actor.get("home_layers") or [])

    by_name = {a["name"]: a for a in actors}
    out = {}
    for name, days in obs.items():
        values, counted = [], 0
        for d, layers in days.items():
            day_values = []
            ticker = by_name[name].get("ticker")
            if ticker and series.get(ticker):
                ar = abnormal_return(series[ticker], bench, d)
                if ar is not None:
                    day_values.append(abs(ar))
            for l in layers:
                if series.get(basket_symbol(l)):
                    ar = abnormal_return(series[basket_symbol(l)], bench, d)
                    if ar is not None:
                        day_values.append(abs(ar))
            if day_values:
                counted += 1
                values += day_values
        if counted >= 3:
            out[name] = statistics.median(values)
    return out


# ---------------------------------------------------------------------------
# Scarcity, bottleneck, migration (daily, after the articles are saved)
# ---------------------------------------------------------------------------

RECOMPUTE_DAYS = 30  # layer_daily rows rewritten each run (self-healing)


def update_scarcity(client, sources: list[dict], today: str | None = None) -> dict:
    """Recompute layer_daily for the last RECOMPUTE_DAYS days and detect a
    confirmed bottleneck migration. Returns {"bottleneck": label for the
    digest, "migration": synthetic goulot event or None}."""
    today = today or date.today().isoformat()
    cfg = cake.load_config()["bottleneck"]
    codes = cake.layer_codes()
    meta = cake.layer_meta()
    first = (date.fromisoformat(today) - timedelta(days=RECOMPUTE_DAYS + cfg["baseline_days"] + 7)).isoformat()
    try:
        articles = cake.fetch_all_rows(lambda a, b: client.table("articles")
                                       .select("published, layers, scarcity, title, description")
                                       .eq("domain", "ia").gte("published", first).order("url").range(a, b).execute())
        stored = cake.fetch_all_rows(lambda a, b: client.table("layer_daily").select("day, layer, sources_fingerprint, bottleneck_status")
                                     .gte("day", (date.fromisoformat(today) - timedelta(days=120)).isoformat())
                                     .order("day").range(a, b).execute())
    except Exception as e:
        logging.warning(f"Cake: scarcity unavailable ({e}) — run the Phase C migration.")
        return {"bottleneck": "", "migration": None}

    counts = daily_layer_counts(articles)
    all_days = [(date.fromisoformat(first) + timedelta(days=k)).isoformat()
                for k in range((date.fromisoformat(today) - date.fromisoformat(first)).days + 1)]
    recent_days = all_days[-RECOMPUTE_DAYS:]
    stats = {l: scarcity_series(counts.get(l, {}), all_days, baseline=cfg["baseline_days"]) for l in codes}

    # Source fingerprints: kept as stored for past days, computed for today
    fp_by_layer: dict[str, list[tuple[str, str]]] = {l: [] for l in codes}
    for r in stored:
        if r["day"] < today and r.get("sources_fingerprint") and r["layer"] in fp_by_layer:
            fp_by_layer[r["layer"]].append((r["day"], r["sources_fingerprint"]))
    for l in codes:
        fp_by_layer[l].append((today, sources_fingerprint(sources, l)))

    def rebuilding(layer: str, day: str) -> bool:
        fps = [(d, f) for d, f in fp_by_layer[layer] if d <= day]
        change = None
        for (d0, f0), (d1, f1) in zip(fps, fps[1:]):
            if f1 != f0:
                change = d1
        return change is not None and (date.fromisoformat(day) - date.fromisoformat(change)).days < cfg["rebuild_days"]

    since = (date.fromisoformat(today) - timedelta(days=PRICE_HISTORY_DAYS)).isoformat()
    price_series = load_series(client, [basket_symbol(l) for l in codes] + [BENCHMARK, "ELEC_US", "ELEC_EU",
                                                                            "GPU_HOUR_H100", TOKEN_IN, TOKEN_OUT], since)
    rows, confirmed_by_day, labels = [], [], {}
    for day in recent_days:
        rebuild = {l for l in codes if rebuilding(l, day)}
        idx = all_days.index(day)
        z_hist = {l: [stats[l][d]["scarcity_z"] for d in all_days[max(0, idx - cfg["min_days"] + 1):idx + 1]] for l in codes}
        top = signalled_layer(z_hist, rebuild)
        proof = price_confirmation(top, price_series, day) if top else None
        status = ("confirme" if proof else "signale") if top else None
        confirmed_by_day.append((day, top if status == "confirme" else None))
        labels[day] = (top, status, proof)
        for l in codes:
            s = stats[l][day]
            rows.append({"day": day, "layer": l, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()},
                         "sources_fingerprint": None,
                         "baseline_rebuilding": l in rebuild,
                         "bottleneck_status": status if l == top else None})
    # keep the fingerprint history of past days as stored
    stored_fp = {(r["day"], r["layer"]): r.get("sources_fingerprint") for r in stored}
    for r in rows:
        if r["day"] < today:
            r["sources_fingerprint"] = stored_fp.get((r["day"], r["layer"]))
        else:
            r["sources_fingerprint"] = sources_fingerprint(sources, r["layer"])
    try:
        for i in range(0, len(rows), 500):
            client.table("layer_daily").upsert(rows[i:i + 500], on_conflict="day,layer").execute()
    except Exception as e:
        logging.error(f"Cake: saving layer_daily failed: {e}")

    top, status, proof = labels[today]
    label = f"{meta[top]['label']} ({'confirmé' if status == 'confirme' else 'signalé'})" if top else "pas de goulot net"

    # Migration of the confirmed bottleneck (stored history + this run's days)
    history = {r["day"]: (r["layer"] if r.get("bottleneck_status") == "confirme" else None)
               for r in stored if r.get("bottleneck_status") == "confirme"}
    days_hist = sorted({r["day"] for r in stored} | set(recent_days))
    merged = [(d, dict(confirmed_by_day).get(d, history.get(d))) for d in days_hist]
    move = migration(merged, cfg["migration_days"])
    event = None
    if move:
        prev, new = move
        try:
            recent = client.table("layer_events").select("id").eq("event_type", "goulot").eq("origin_layer", prev) \
                .eq("target_layer", new).gte("first_seen", (date.fromisoformat(today) - timedelta(days=30)).isoformat()) \
                .execute().data or []
        except Exception:
            recent = [1]
        if not recent:
            proof_now = price_confirmation(new, price_series, today) or "prix"
            event = {"event_type": "goulot", "origin_layer": prev, "target_layer": new,
                     "what": f"Le goulot passe de {meta[prev]['label']} à {meta[new]['label']} (confirmé par : {proof_now})",
                     "is_major": True, "first_seen": today, "last_seen": today, "engagement_history": [],
                     "article_urls": [], "bypassed": [], "winners": [], "losers": []}
            cake.save_events(client, [event])
            logging.info(f"Cake: bottleneck migration {prev} → {new}")
    return {"bottleneck": label, "migration": event}


def migration_message(event: dict) -> str:
    return f"🎂 {event['what']}"
