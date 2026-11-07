"""
Value of one MWh at each layer of the AI cake (SPECS_LAYER_CAKE.md §6.5).

Bottom-up: at each floor an observable unit price is converted into dollars
per MWh of electricity at the meter (cooling and losses included):

  P1 energy         = wholesale electricity price
  P2 chips          = P1 + accelerator price / (TDP × PUE × lifetime hours × utilisation)
  P3 infrastructure = GPU-hour price / (TDP × PUE)
  P4 models         = price per million tokens / energy per million tokens
  P5 applications   = revenue per user per month / energy per user per month

P3-P5 are market prices that already include the floors below, so they are not
summed. Value added = P_k − P_(k−1) (negative when a floor sells below the
cost of the floors underneath: a signal, never clipped to zero), multiplier =
P_k / P_(k−1).

Usage: python layer_value.py [--telegram]   # writes data/layer_value.json
"""

import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
MANUAL_PATH = ROOT / "data" / "layer_value_manual.json"
OUTPUT_PATH = ROOT / "data" / "layer_value.json"
LAYERS = ("energie", "puces", "infrastructure", "modeles", "applications")
_RANK = {"low": 0, "medium": 1, "high": 2}


def load_manual() -> dict:
    return json.loads(MANUAL_PATH.read_text(encoding="utf-8"))


def load_output() -> dict:
    try:
        return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _value(inputs: dict, key: str) -> float | None:
    v = (inputs.get(key) or {}).get("value")
    return float(v) if isinstance(v, (int, float)) else None


def _at_meter(inputs: dict, key: str, pue: float | None) -> float | None:
    """An energy input at the meter: chip-level values are multiplied by the PUE."""
    v = _value(inputs, key)
    if v is None:
        return None
    if (inputs.get(key) or {}).get("level") == "chip":
        return v * pue if pue else None
    return v


EIA_STEO_URL = "https://api.eia.gov/v2/steo/data/"
EIA_HUBS = ("ELWHU_CA", "ELWHU_FL", "ELWHU_MW", "ELWHU_NE", "ELWHU_NW", "ELWHU_NY", "ELWHU_PJ",
            "ELWHU_SE", "ELWHU_SP", "ELWHU_SW", "ELWHU_TX")


def eia_us_wholesale(api_key: str, months: int = 24) -> list[tuple[str, float]]:
    """Monthly US wholesale electricity price ($/MWh): simple mean of the
    EIA STEO hub series (ELWHU_*), completed months only (STEO also carries
    forecasts). EIA's own national figure is load-weighted; the weights are
    not in the API, hence the simple mean."""
    import requests
    from calendar import monthrange
    first = date.today().replace(day=1)
    start = date(first.year - (months // 12 + 1), first.month, 1).strftime("%Y-%m")
    rows, offset = [], 0
    while True:
        resp = requests.get(EIA_STEO_URL, timeout=30, params={
            "api_key": api_key, "frequency": "monthly", "data[0]": "value", "start": start,
            "end": first.strftime("%Y-%m"), "offset": offset, "length": 5000, "facets[seriesId][]": list(EIA_HUBS)})
        resp.raise_for_status()
        page = resp.json()["response"]["data"]
        rows += [r for r in page if str(r.get("seriesId", "")).startswith("ELWHU_")]
        if len(page) < 5000:
            break
        offset += 5000
    by_month: dict[str, list[float]] = {}
    for r in rows:
        if r["period"] < first.strftime("%Y-%m") and r.get("value") is not None:
            by_month.setdefault(r["period"], []).append(float(r["value"]))
    out = []
    for m, values in sorted(by_month.items()):
        y, mo = map(int, m.split("-"))
        out.append((f"{m}-{monthrange(y, mo)[1]:02d}", sum(values) / len(values)))
    return out[-months:]


def _apply_eia(inputs: dict) -> None:
    """Replace the hand-entered US electricity price by EIA data when EIA_API_KEY is set."""
    key = os.environ.get("EIA_API_KEY")
    if not key:
        return
    try:
        series = eia_us_wholesale(key)
    except Exception as e:
        logging.warning(f"EIA unavailable, keeping the manual electricity price: {e}")
        return
    if series:
        day, value = series[-1]
        inputs["electricity_us"] = {
            "value": round(value, 2), "unit": "USD/MWh", "as_of": day, "confidence": "high",
            "source": "EIA STEO, moyenne simple des hubs de gros (ELWHU_*)",
            "source_url": "https://www.eia.gov/outlooks/steo/",
            "note": "Mois écoulé le plus récent ; moyenne simple des hubs (la moyenne EIA est pondérée par la charge).",
            "series": series,
        }


def compute_floors(inputs: dict) -> list[dict]:
    """One dict per floor: layer, price_usd_mwh (None when an input is missing),
    value_added_usd_mwh, multiplier, value_added_negative, confidence, inputs."""
    pue = _value(inputs, "pue")
    tdp = _value(inputs, "accelerator_tdp_kw")
    p = {}
    used = {}

    used["energie"] = ["electricity_us"]
    p["energie"] = _value(inputs, "electricity_us")

    used["puces"] = ["electricity_us", "accelerator_price_usd", "accelerator_tdp_kw", "pue", "lifetime_years", "utilization"]
    price, life, util = (_value(inputs, k) for k in ("accelerator_price_usd", "lifetime_years", "utilization"))
    if None not in (p["energie"], price, tdp, pue, life, util) and tdp * pue * life * util:
        lifetime_mwh = tdp * pue * life * 8760 * util / 1000
        p["puces"] = p["energie"] + price / lifetime_mwh
    else:
        p["puces"] = None

    used["infrastructure"] = ["gpu_hour_usd", "accelerator_tdp_kw", "pue"]
    gpu_hour = _value(inputs, "gpu_hour_usd")
    p["infrastructure"] = gpu_hour / (tdp * pue / 1000) if None not in (gpu_hour, tdp, pue) and tdp * pue else None

    used["modeles"] = ["token_price_usd_per_m", "energy_per_m_tokens_mwh", "pue"]
    tokens = _value(inputs, "token_price_usd_per_m")
    energy_m = _at_meter(inputs, "energy_per_m_tokens_mwh", pue)
    p["modeles"] = tokens / energy_m if None not in (tokens, energy_m) and energy_m else None

    used["applications"] = ["app_revenue_per_user_month_usd", "energy_per_prompt_wh", "prompts_per_user_month", "pue"]
    arpu, prompts = _value(inputs, "app_revenue_per_user_month_usd"), _value(inputs, "prompts_per_user_month")
    wh = _at_meter(inputs, "energy_per_prompt_wh", pue)
    user_mwh = wh * prompts / 1e6 if None not in (wh, prompts) else None
    p["applications"] = arpu / user_mwh if None not in (arpu, user_mwh) and user_mwh else None

    floors = []
    prev = None
    for layer in LAYERS:
        price_k = p[layer]
        confs = [(inputs.get(k) or {}).get("confidence", "low") for k in used[layer]]
        confidence = min(confs, key=lambda c: _RANK.get(c, 0)) if confs else "low"
        if layer in ("modeles", "applications"):
            confidence = "low"  # always: energy per token / per user are rough estimates
        va = mult = None
        if price_k is not None and prev is not None:
            va = price_k - prev
            mult = price_k / prev if prev else None
        floors.append({
            "layer": layer,
            "price_usd_mwh": None if price_k is None else round(price_k, 2),
            "value_added_usd_mwh": None if va is None else round(va, 2),
            "multiplier": None if mult is None else round(mult, 2),
            "value_added_negative": va is not None and va < 0,
            "confidence": confidence,
            "inputs": [{"key": k, "source_url": (inputs.get(k) or {}).get("source_url"),
                        "source": (inputs.get(k) or {}).get("source", "")} for k in used[layer]],
        })
        prev = price_k
    return floors


def refresh(today: str | None = None) -> dict:
    """Compute the floors from the manual inputs (+ today's token price index)
    and write data/layer_value.json, keeping one history entry per month."""
    today = today or date.today().isoformat()
    manual = load_manual()
    inputs = {k: dict(v) for k, v in manual["inputs"].items()}
    _apply_eia(inputs)
    token = inputs.get("token_price_usd_per_m") or {}
    if token.get("auto") == "openrouter":
        try:
            import cake_market
            _, out_price, _ = cake_market.fetch_token_price_index()
            token["value"] = round(out_price, 4)
            token["as_of"] = today
        except Exception as e:
            logging.error(f"Token price index unavailable: {e}")
    floors = compute_floors(inputs)
    previous = load_output()
    month = today[:7]
    history = [h for h in previous.get("history", []) if h["month"] != month]
    history.append({"month": month, "floors": {f["layer"]: {"price": f["price_usd_mwh"],
                                                             "value_added": f["value_added_usd_mwh"]} for f in floors}})
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": today,
        "updated": manual.get("updated"),
        "next_review": manual.get("next_review"),
        "token_price_usd_per_m": token.get("value"),
        "floors": floors,
        "history": sorted(history, key=lambda h: h["month"])[-36:],
    }
    OUTPUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def monthly_message(out: dict) -> str:
    """Telegram report: the floor adding the most value per MWh, its change
    vs last month, and the floors in negative value added."""
    import cake
    meta = cake.layer_meta()
    floors = [f for f in out["floors"] if f["value_added_usd_mwh"] is not None]
    if not floors:
        return "🎂 <b>Valeur du MWh par couche</b> : données insuffisantes ce mois-ci."
    best = max(floors, key=lambda f: f["value_added_usd_mwh"])
    lines = [f"🎂 <b>Valeur du MWh par couche — {out['as_of'][:7]}</b>",
             f"🏆 {meta[best['layer']]['label']} crée le plus de valeur : "
             f"{'≈ ' if best['confidence'] == 'low' else ''}{best['value_added_usd_mwh']:,.0f} $/MWh ajoutés "
             f"(×{best['multiplier']:.1f} sur la couche du dessous)".replace(",", " ").replace(".", ",")]
    hist = out.get("history") or []
    if len(hist) >= 2:
        before = (hist[-2]["floors"].get(best["layer"]) or {}).get("value_added")
        if before:
            lines.append(f"📈 Variation vs {hist[-2]['month']} : {(best['value_added_usd_mwh'] / before - 1) * 100:+.0f} %")
    negative = [f for f in out["floors"] if f["value_added_negative"]]
    for f in negative:
        lines.append(f"▼ {meta[f['layer']]['label']} vend sous le coût des couches inférieures "
                     f"({f['value_added_usd_mwh']:,.0f} $/MWh)".replace(",", " "))
    return "\n".join(lines)


def stale_reason(out: dict | None = None, today: date | None = None) -> str:
    """Why the manual inputs need a review, or ''."""
    out = out if out is not None else (load_output() or {"next_review": load_manual().get("next_review")})
    nr = out.get("next_review")
    if nr and (today or date.today()) > date.fromisoformat(nr):
        return f"entrées manuelles à revoir depuis le {nr}"
    return ""


def market_series() -> list[tuple[str, str, dict]]:
    """Hand-entered monthly series that also feed market_data (bottleneck price confirmation)."""
    inputs = {k: dict(v) for k, v in load_manual()["inputs"].items()}
    _apply_eia(inputs)
    out = []
    for day, value in (inputs["electricity_us"].get("series") or [])[:-1]:  # history; the last month is added below
        out.append(("ELEC_US", "Prix de gros électricité États-Unis",
                    {"value": value, "as_of": day, "unit": "USD/MWh", "source_url": EIA_STEO_URL}))
    for key, sym, label in (("electricity_us", "ELEC_US", "Prix de gros électricité États-Unis"),
                            ("electricity_eu", "ELEC_EU", "Prix de gros électricité Europe"),
                            ("gpu_hour_usd", "GPU_HOUR_H100", "Prix location H100 (GPU-heure)")):
        item = inputs.get(key) or {}
        if isinstance(item.get("value"), (int, float)) and item.get("as_of"):
            out.append((sym, label, item))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = refresh()
    for f in result["floors"]:
        logging.info(f"{f['layer']:15} {f['price_usd_mwh']} $/MWh · VA {f['value_added_usd_mwh']} · ×{f['multiplier']}"
                     f"{' · NEGATIVE' if f['value_added_negative'] else ''} ({f['confidence']})")
    if "--telegram" in sys.argv and os.environ.get("TELEGRAM_BOT_TOKEN"):
        import requests
        requests.post(f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage",
                      json={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": monthly_message(result),
                            "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=10)
