"""End to end: the daily cake steps against an in-memory Supabase."""

import asyncio
import json
from datetime import date, timedelta
from types import SimpleNamespace

import cake
import cake_dashboard
import cake_market
import country_indicators
import main
from tests.fake_supabase import FakeSupabase

TODAY = date.today()


def d(n):
    return (TODAY - timedelta(days=n)).isoformat()


class FakeGroq:
    """Answers the event extraction prompt with canned events, by title."""

    def __init__(self, events_by_title):
        self.events_by_title = events_by_title
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def with_options(self, **kw):
        return self

    async def _create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        items = json.loads(prompt.split("ARTICLES = ", 1)[1])
        rows = [{"id": it["id"], "event": self.events_by_title.get(it["title"])} for it in items]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"articles": rows})))],
                               usage=SimpleNamespace(total_tokens=1000))


def _article(title, url):
    a = main.Article(title=title, url=url, source="Src", country="🇫🇷", published=TODAY.isoformat())
    a.cross_layer, a.layers = True, ["energie", "modeles"]
    return a


def _offline(monkeypatch):
    monkeypatch.setattr(country_indicators, "_cmc_usd_bn", lambda slug, metric: 10.0)
    monkeypatch.setattr(cake_market, "yf_revenue_usd_bn", lambda t: 5.0)


def test_daily_events_step(monkeypatch):
    _offline(monkeypatch)
    db = FakeSupabase()
    groq = FakeGroq({
        "EDF lance un modèle": {"actor": "EDF", "actor_home_layers": ["energie"], "target_layer": "modeles",
                                "engagement": "annonce", "what": "EDF publie un modèle", "winners": ["GPU"],
                                "relations": [{"from": "EDF", "relation": "utilise", "to": "Nvidia"}]},
        "Acme Power entre dans les puces": {"actor": "Acme Power", "actor_home_layers": ["energie"],
                                            "target_layer": "puces", "engagement": "contrat", "what": "Acme conçoit une puce"},
        "Nvidia sort un GPU": {"actor": "Nvidia", "actor_home_layers": ["puces"], "target_layer": "puces",
                               "engagement": "mise_en_service", "what": "Nouveau GPU"},
    })
    arts = [_article("EDF lance un modèle", "https://a/1"), _article("Acme Power entre dans les puces", "https://a/2"),
            _article("Nvidia sort un GPU", "https://a/3")]
    touched = asyncio.run(cake.process_layer_events(arts, db, groq, "m", retries=0))

    assert {(e["actor_name"], e["target_layer"]) for e in touched} == {("EDF", "modeles"), ("Acme Power", "puces")}
    assert len(db.tables["layer_events"]) == 2
    acme = next(a for a in db.tables["actors"] if a["name"] == "Acme Power")
    assert acme["needs_review"] and acme["home_layers"] == ["energie"]
    assert db.tables["actor_relations"][0]["to_actor"] == "Nvidia"
    assert all(a.get("weight") is not None for a in db.tables["actors"] if a["name"] in ("EDF", "Nvidia"))
    assert any("Acme Power" in line for line in cake.review_reminder_lines(db))

    # Next day, a contract: same event, progressed, notified again once
    edf = next(e for e in touched if e["actor_name"] == "EDF")
    if cake.needs_notification(edf):
        cake.mark_notified(db, [edf])
    groq.events_by_title = {"EDF signe": {"actor": "EDF", "actor_home_layers": ["energie"], "target_layer": "modeles",
                                          "engagement": "contrat", "what": "EDF signe un contrat"}}
    touched2 = asyncio.run(cake.process_layer_events([_article("EDF signe", "https://a/4")], db, groq, "m",
                                                     retries=0, today=(TODAY + timedelta(days=1)).isoformat()))
    assert len(db.tables["layer_events"]) == 2
    row = next(r for r in db.tables["layer_events"] if r["target_layer"] == "modeles")
    assert row["engagement"] == "contrat" and row["article_urls"] == ["https://a/1", "https://a/4"]
    assert touched2[0]["progressed_at"] == (TODAY + timedelta(days=1)).isoformat()


def test_prices_are_stored_once_and_baskets_chain(monkeypatch):
    db = FakeSupabase()
    days = [d(k) for k in range(10, 0, -1)]

    def fake_closes(tickers, period):
        return {t: {day: 100 + i for i, day in enumerate(days)} for t in tickers}

    monkeypatch.setattr(cake_market, "download_closes", fake_closes)
    cake_market.collect_prices(db)
    n = len(db.tables["market_data"])
    cake_market.collect_prices(db)  # same data again: nothing new
    assert len(db.tables["market_data"]) == n
    series = cake_market.load_series(db, ["CAKE_PUCES"], d(30))["CAKE_PUCES"]
    assert series[0][1] == 100.0 and abs(series[-1][1] - 109.0) < 1e-6


def test_scarcity_bottleneck_and_dashboard(monkeypatch):
    db = FakeSupabase()
    # 120 days of energy news at a calm 10 % tension, then 5 tense days
    for k in range(120, -1, -1):
        for i in range(10):
            tense = (k <= 4 and i < 8) or (k > 4 and i == (k % 10))
            db.table("articles").insert({"url": f"https://e/{k}/{i}", "domain": "ia", "published": d(k),
                                         "layers": ["energie"], "scarcity": "tension" if tense else "aucune",
                                         "title": "", "description": "", "topics": []}).execute()
    # energy basket +12 % vs QQQ +2 % over 30 days → confirmed
    for sym, end in (("CAKE_ENERGIE", 112.0), ("QQQ", 102.0)):
        for k, v in ((40, 100.0), (0, end)):
            db.table("market_data").insert({"domain": "ia", "symbol": sym, "value": v,
                                            "collected_at": f"{d(k)}T00:00:00+00:00"}).execute()
    sources = [{"name": "Generic"}, {"name": "Energy", "cake_layer": "energie"}]
    out = cake_market.update_scarcity(db, sources)
    assert out["bottleneck"] == "Énergie (confirmé)"
    today_rows = [r for r in db.tables["layer_daily"] if r["day"] == TODAY.isoformat()]
    assert len(today_rows) == len(cake.layer_codes())
    assert next(r for r in today_rows if r["layer"] == "energie")["bottleneck_status"] == "confirme"

    market = cake_dashboard.load_cake_market(db)
    assert market["bottleneck"] == {"layer": "energie", "status": "confirmé"}
    assert market["layer_extras"][30]["energie"]["gauge"] > 0
    data = cake_dashboard.load_cake_data(db) or {"events": [], "actors": [], "articles": []}
    html = cake_dashboard.cake_section_html(data, [], {}, market=market)
    assert "Goulot : Énergie (confirmé)" in html and "goulot confirmé" in html
    assert len(cake_dashboard.cake_figures(data, market)) == 3  # matrix, value staircase, curves


def test_backfill_after_daily_runs_extends_baskets_backwards(monkeypatch):
    db = FakeSupabase()
    days = [d(k) for k in range(20, 0, -1)]
    closes = {day: 100 + 2 * i for i, day in enumerate(days)}
    monkeypatch.setattr(cake_market, "download_closes",
                        lambda tickers, period: {t: dict(list(closes.items())[-5:] if period == "15d" else closes)
                                                 for t in tickers})
    cake_market.collect_prices(db)                  # a daily run first: index starts 5 days ago
    cake_market.collect_prices(db, backfill=True)   # then the year of history
    series = dict(cake_market.load_series(db, ["CAKE_PUCES"], d(60))["CAKE_PUCES"])
    assert sorted(series) == days                   # every day, no gap, no duplicate
    first_stored = days[-5]
    for day in days:                                # proportional to the closes all the way back
        assert abs(series[day] / series[first_stored] - closes[day] / closes[first_stored]) < 1e-6  # values stored with 6 decimals
