"""
Country-level AI investment indicators ("Who really invests in AI?").

Builds four rankings, each normalised by GDP so that small countries that
commit heavily stand out as much as large ones:
  - private_investment : private AI investment (Stanford AI Index, hand-entered)
  - hardware_imports   : net imports of servers + computer parts (UN Comtrade)
  - public_compute     : GPU-accelerated public supercomputers (TOP500)
  - announcements      : public AI budgets announced per year (hand-entered)

Usage: python country_indicators.py   # refresh data/country_indicators.json
"""

import html
import json
import logging
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).parent
MANUAL_PATH = ROOT / "data" / "country_indicators_manual.json"
OUTPUT_PATH = ROOT / "data" / "country_indicators.json"

USER_AGENT = "Mozilla/5.0 (compatible; Veille-IA country indicators)"

# Auto-refreshed sources are considered stale past this age (refresh job broken).
AUTO_MAX_AGE_DAYS = 45

# HS 847150 = processing units (servers), HS 847330 = parts incl. GPU boards.
HS_CODES = ["847150", "847330"]
# A Comtrade year is used only once enough countries have reported it.
COMTRADE_MIN_REPORTERS = 100

TOP500_PUBLIC_SEGMENTS = {"Research", "Academic", "Government"}
# GPUs / AI accelerators as they appear in TOP500 system descriptions. Plain
# "NVIDIA" is not enough: Grace CPUs and InfiniBand interconnects carry it too.
TOP500_ACCEL_RE = re.compile(
    r"NVIDIA\s+(?:Tesla\s+|DGX\s+|HGX\s+)?(?:[ABHLV]\d{2,3}\b|GH\d{3}|GB\d{3}|P100|K\d{2}\b|Grace Hopper)"
    r"|Instinct|Ascend|Gaudi|Data Center GPU|TPU",
    re.IGNORECASE,
)

# TOP500 country names that differ from World Bank names.
TOP500_COUNTRY_ALIASES = {
    "United States": "USA", "Korea, South": "KOR", "South Korea": "KOR",
    "Russia": "RUS", "Taiwan": "TWN", "Czech Republic": "CZE", "Czechia": "CZE",
    "Hong Kong": "HKG", "Turkey": "TUR", "Türkiye": "TUR", "Slovakia": "SVK",
    "Vietnam": "VNM", "Iran": "IRN", "Egypt": "EGY",
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, retries: int = 5) -> requests.Response:
    """GET with retry on rate limits and transient errors."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60, headers={"User-Agent": USER_AGENT})
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise
            wait = 3 * (attempt + 1)
            logging.warning(f"{url} failed ({e}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _flag(iso2: str) -> str:
    """ISO-2 code → emoji flag."""
    if len(iso2) != 2 or not iso2.isalpha():
        return ""
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in iso2.upper())


# ---------------------------------------------------------------------------
# World Bank — GDP, exchange rates, country names
# ---------------------------------------------------------------------------

def fetch_world_bank(indicator: str) -> dict[str, dict]:
    """Most recent value per country: {iso3: {value, year, name, iso2}}."""
    resp = _get(
        f"https://api.worldbank.org/v2/country/all/indicator/{indicator}",
        params={"format": "json", "mrnev": 1, "per_page": 500},
    )
    out = {}
    for r in resp.json()[1]:
        iso3 = r.get("countryiso3code")
        if iso3 and r.get("value") is not None:
            out[iso3] = {
                "value": float(r["value"]),
                "year": int(r["date"]),
                "name": r["country"]["value"],
                "iso2": r["country"]["id"],
            }
    return out


def _country_label(iso3: str, countries: dict[str, dict]) -> str:
    c = countries.get(iso3)
    if not c:
        return iso3
    name = {
        "Korea, Rep.": "South Korea", "Russian Federation": "Russia", "Turkiye": "Turkey",
        "Egypt, Arab Rep.": "Egypt", "Iran, Islamic Rep.": "Iran", "Hong Kong SAR, China": "Hong Kong",
        "Czechia": "Czechia", "Slovak Republic": "Slovakia", "Viet Nam": "Vietnam",
    }.get(c["name"], c["name"])
    return f"{_flag(c['iso2'])} {name}".strip()


# ---------------------------------------------------------------------------
# 1. Private AI investment (hand-entered)
# ---------------------------------------------------------------------------

def build_private_investment(manual: dict, gdp: dict, countries: dict) -> dict:
    m = manual["private_investment"]
    rows = []
    for iso3, usd_bn in m["values_usd_bn"].items():
        g = gdp.get(iso3)
        if not g:
            continue
        rows.append({
            "iso3": iso3,
            "name": _country_label(iso3, countries),
            "value": usd_bn * 1e9 / g["value"] * 100,
            "detail": f"${usd_bn:,.1f}bn private AI investment in {m['year']}",
        })
    return {
        "title": "Private AI investment",
        "unit": "% of GDP",
        "source": m["source"],
        "source_url": m["source_url"],
        "as_of": str(m["year"]),
        "updated": m["updated"],
        "next_review": m["next_review"],
        "auto": False,
        "note": m.get("note", ""),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 2. Hardware imports (UN Comtrade)
# ---------------------------------------------------------------------------

def _comtrade(year: int, cmd: str, flow: str) -> list[dict]:
    resp = _get(
        "https://comtradeapi.un.org/public/v1/preview/C/A/HS",
        params={"period": year, "cmdCode": cmd, "flowCode": flow, "partnerCode": 0},
    )
    time.sleep(2)  # public endpoint rate limit
    data = resp.json().get("data") or []
    # Keep the plain world total, not breakdowns by customs regime / transport mode.
    return [r for r in data if r["customsCode"] == "C00" and r["motCode"] == 0 and r["partner2Code"] == 0]


def build_hardware_imports(gdp: dict, countries: dict) -> dict:
    ref = _get("https://comtradeapi.un.org/files/v1/app/reference/Reporters.json").json()["results"]
    code_to_iso3 = {r["id"]: r.get("reporterCodeIsoAlpha3") for r in ref}

    # Most recent year with enough reporting countries.
    this_year = date.today().year
    for year in (this_year - 1, this_year - 2, this_year - 3):
        imports = _comtrade(year, HS_CODES[0], "M")
        if len({r["reporterCode"] for r in imports}) >= COMTRADE_MIN_REPORTERS:
            break
    else:
        raise RuntimeError("No Comtrade year with enough reporters")

    totals: dict[str, list[float]] = {}  # iso3 → [imports, exports]
    for cmd in HS_CODES:
        for flow, idx in (("M", 0), ("X", 1)):
            records = imports if (cmd, flow) == (HS_CODES[0], "M") else _comtrade(year, cmd, flow)
            for r in records:
                iso3 = code_to_iso3.get(r["reporterCode"])
                if iso3:
                    totals.setdefault(iso3, [0.0, 0.0])[idx] += r["primaryValue"] or 0

    rows = []
    for iso3, (imp, exp) in totals.items():
        g = gdp.get(iso3)
        # Tiny economies make the ratio noisy; transit hubs come out ≤ 0.
        if not g or g["value"] < 50e9 or imp <= exp:
            continue
        rows.append({
            "iso3": iso3,
            "name": _country_label(iso3, countries),
            "value": (imp - exp) / g["value"] * 100,
            "detail": f"${(imp - exp) / 1e9:,.1f}bn net imports (${imp / 1e9:,.1f}bn in, ${exp / 1e9:,.1f}bn out) in {year}",
        })
    return {
        "title": "Net imports of servers & GPU parts",
        "unit": "% of GDP",
        "source": "UN Comtrade (HS 847150 + 847330, imports minus exports)",
        "source_url": "https://comtradeplus.un.org/",
        "as_of": str(year),
        "updated": date.today().isoformat(),
        "auto": True,
        "note": "Includes non-AI servers. Countries that do not report customs data (e.g. UAE) are missing.",
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 3. Public AI supercomputers (TOP500)
# ---------------------------------------------------------------------------

_TOP500_ROW_RE = re.compile(
    r'<a href="/system/(\d+)/?">\s*<b>(.*?)</b>(.*?)</a>.*?'
    r'<a href="/site/(\d+)/?">(.*?)</a><br>\s*(.*?)\s*</td>\s*'
    r'<td[^>]*>[\d,]+</td>\s*<td[^>]*>([\d,.]+)</td>',
    re.S,
)


def _latest_top500_list() -> tuple[int, int]:
    """TOP500 lists come out in June and November."""
    today = date.today()
    candidates = []
    for y in (today.year, today.year - 1):
        for m in (11, 6):
            if (y, m) <= (today.year, today.month):
                candidates.append((y, m))
    for y, m in candidates:
        page = _get(f"https://top500.org/lists/top500/list/{y}/{m:02d}/", params={"page": 1}).text
        if _TOP500_ROW_RE.search(page):
            return y, m
    raise RuntimeError("No TOP500 list found")


def _site_segment(site_id: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", _get(f"https://top500.org/site/{site_id}/").text))
    time.sleep(0.5)
    m = re.search(r"Segment\s+(\w+)", text)
    return m.group(1) if m else ""


def build_public_compute(gdp: dict, countries: dict, previous: dict | None) -> dict:
    year, month = _latest_top500_list()
    list_id = f"{year}/{month:02d}"
    name_to_iso3 = {c["name"]: iso3 for iso3, c in countries.items()} | TOP500_COUNTRY_ALIASES

    systems = []
    for page in range(1, 6):
        text = _get(f"https://top500.org/lists/top500/list/{year}/{month:02d}/", params={"page": page}).text
        for sys_id, name, desc, site_id, site, country, rmax in _TOP500_ROW_RE.findall(text):
            desc = html.unescape(re.sub(r"\s+", " ", desc))
            if TOP500_ACCEL_RE.search(desc):
                systems.append({
                    "name": html.unescape(name.strip()),
                    "site_id": site_id,
                    "site": html.unescape(site.strip()),
                    "country": html.unescape(country.strip()),
                    "rmax_pf": float(rmax.replace(",", "")),
                })
        time.sleep(0.5)
    if not systems:
        raise RuntimeError("TOP500 page layout changed: no accelerated system parsed")

    # Segments rarely change: reuse those fetched for earlier lists.
    segments = dict((previous or {}).get("site_segments", {}))
    for s in systems:
        if s["site_id"] not in segments:
            segments[s["site_id"]] = _site_segment(s["site_id"])

    per_country: dict[str, dict] = {}
    for s in systems:
        if segments.get(s["site_id"]) not in TOP500_PUBLIC_SEGMENTS:
            continue
        iso3 = name_to_iso3.get(s["country"])
        if not iso3:
            logging.warning(f"TOP500: unknown country {s['country']!r}")
            continue
        c = per_country.setdefault(iso3, {"rmax": 0.0, "systems": []})
        c["rmax"] += s["rmax_pf"]
        c["systems"].append(s["name"])

    rows = []
    for iso3, c in per_country.items():
        g = gdp.get(iso3)
        if not g:
            continue
        top = ", ".join(c["systems"][:3]) + ("…" if len(c["systems"]) > 3 else "")
        rows.append({
            "iso3": iso3,
            "name": _country_label(iso3, countries),
            "value": c["rmax"] / (g["value"] / 1e12),
            "detail": f"{c['rmax']:,.0f} PFlop/s across {len(c['systems'])} systems ({top})",
        })
    return {
        "title": "Public AI supercomputers",
        "unit": "PFlop/s per $1tn GDP",
        "source": f"TOP500 list {list_id} (GPU-accelerated, research/academic/government sites)",
        "source_url": f"https://top500.org/lists/top500/{year}/{month:02d}/",
        "as_of": list_id,
        "updated": date.today().isoformat(),
        "auto": True,
        "note": "HPL (FP64) performance; private clusters are rarely listed in TOP500.",
        "rows": rows,
        "site_segments": segments,
    }


# ---------------------------------------------------------------------------
# 4. Announced public AI budgets (hand-entered)
# ---------------------------------------------------------------------------

def build_announcements(manual: dict, gdp: dict, fx: dict, countries: dict) -> dict:
    m = manual["announcements"]
    rows = []
    for e in m["entries"]:
        g = gdp.get(e["iso3"])
        rate = 1.0 if e["currency"] == "USD" else (fx.get(e["iso3"]) or {}).get("value")
        if not g or not rate:
            logging.warning(f"Announcements: missing GDP or FX for {e['iso3']}")
            continue
        per_year_usd = e["amount"] / rate / e["years"]
        rows.append({
            "iso3": e["iso3"],
            "name": _country_label(e["iso3"], countries),
            "value": per_year_usd / g["value"] * 100,
            "detail": f"{e['label']} ({e['period']}): ${per_year_usd / 1e9:,.2f}bn per year"
                      + (f". {e['note']}" if e.get("note") else ""),
            "confidence": e.get("confidence", "medium"),
            "source_url": e["source_url"],
        })
    return {
        "title": "Announced public AI budgets",
        "unit": "% of GDP per year",
        "source": m["source"],
        "source_url": "",
        "as_of": m["updated"],
        "updated": m["updated"],
        "next_review": m["next_review"],
        "auto": False,
        "note": m.get("note", ""),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Staleness — used by the dashboard badge and the Telegram reminder
# ---------------------------------------------------------------------------

def load_indicators() -> dict:
    """Read the generated file; {} if it does not exist yet."""
    try:
        return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def stale_reason(ind: dict, today: date | None = None) -> str:
    """Why an indicator needs attention, or '' if it is up to date."""
    today = today or date.today()
    if ind.get("error"):
        return f"last refresh failed: {ind['error']}"
    if not ind.get("auto") and ind.get("next_review") and today > date.fromisoformat(ind["next_review"]):
        return f"review due since {ind['next_review']}"
    if ind.get("auto") and ind.get("updated"):
        age = (today - date.fromisoformat(ind["updated"])).days
        if age > AUTO_MAX_AGE_DAYS:
            return f"not refreshed for {age} days"
    return ""


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

def refresh() -> dict:
    manual = json.loads(MANUAL_PATH.read_text(encoding="utf-8"))
    previous = load_indicators()

    # GDP entries double as the country directory (names, flags).
    gdp = fetch_world_bank("NY.GDP.MKTP.CD")
    for iso3, o in manual.get("gdp_overrides_usd", {}).items():
        if not iso3.startswith("_"):
            gdp.setdefault(iso3, o)
    countries = gdp
    fx = fetch_world_bank("PA.NUS.FCRF")

    builders = {
        "private_investment": lambda: build_private_investment(manual, gdp, countries),
        "hardware_imports": lambda: build_hardware_imports(gdp, countries),
        "public_compute": lambda: build_public_compute(gdp, countries, previous.get("public_compute")),
        "announcements": lambda: build_announcements(manual, gdp, fx, countries),
    }
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for key, build in builders.items():
        try:
            ind = build()
            ind["rows"].sort(key=lambda r: -r["value"])
            out[key] = ind
            logging.info(f"{key}: {len(ind['rows'])} countries")
        except Exception as e:  # keep the last good data if one source breaks
            logging.error(f"{key}: refresh failed: {e}")
            out[key] = {**previous.get(key, {}), "error": str(e)[:200]}

    OUTPUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = refresh()
    failed = [k for k, v in result.items() if isinstance(v, dict) and v.get("error")]
    sys.exit(1 if failed else 0)
