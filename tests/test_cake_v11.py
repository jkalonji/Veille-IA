"""Phase E (v1.1) — directed relations, layer backfill."""

import asyncio
import json
from datetime import date, timedelta
from types import SimpleNamespace

import cake
import cake_market
import country_indicators
import main
from tests.fake_supabase import FakeSupabase
from tests.test_cake_pipeline import FakeGroq

ACTORS = cake.load_manual_actors()


def test_relation_edges_point_to_whom_is_depended_on():
    index = cake.actor_index(ACTORS)
    edges = cake.relation_edges([
        {"from_actor": "OpenAI", "relation": "utilise", "to_actor": "Nvidia"},
        {"from_actor": "TSMC", "relation": "fournit", "to_actor": "Apple"},
        {"from_actor": "Nvidia", "relation": "investit", "to_actor": "CoreWeave"},
        {"from_actor": "EDF", "relation": "partenaire", "to_actor": "Mistral"},
        {"from_actor": "Unknown Co", "relation": "utilise", "to_actor": "Nvidia"},
    ], index)
    assert edges[("OpenAI", "Nvidia")] == 3.0
    assert edges[("Apple", "TSMC")] == 3.0
    assert edges[("CoreWeave", "Nvidia")] == 3.0
    assert edges[("EDF", "Mistral AI")] == edges[("Mistral AI", "EDF")] == 3.0
    assert len(edges) == 5


class BatchGroq(FakeGroq):
    """Answers both the layer tagging prompt and the event extraction prompt."""

    async def _create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "KNOWN_ACTORS" in prompt:
            return await super()._create(**kwargs)
        items = json.loads(prompt.split("ARTICLES = ", 1)[1])
        rows = [{"id": it["id"], "layers": ["energie", "modeles"] if "EDF" in it["title"] else ["puces"],
                 "cross_layer": "EDF" in it["title"], "scarcity": "aucune"} for it in items]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"articles": rows})))],
                               usage=SimpleNamespace(total_tokens=500))


def test_backfill_tags_old_articles_once_and_dates_events_silently(monkeypatch):
    monkeypatch.setattr(country_indicators, "_cmc_usd_bn", lambda slug, metric: 10.0)
    monkeypatch.setattr(cake_market, "yf_revenue_usd_bn", lambda t: 5.0)
    db = FakeSupabase()
    old = (date.today() - timedelta(days=20)).isoformat()
    for i, title in enumerate(["EDF lance un modèle", "TSMC agrandit une usine", "Autre puce"]):
        db.table("articles").insert({"url": f"https://o/{i}", "title": title, "description": "", "source": "S",
                                     "published": old, "domain": "ia", "layers": [], "layers_tagged": False}).execute()
    groq = BatchGroq({"EDF lance un modèle": {"actor": "EDF", "target_layer": "modeles", "engagement": "annonce",
                                              "what": "EDF publie un modèle"}})
    assert asyncio.run(cake.backfill_layers(db, groq, "m", main.Article)) == 3
    assert all(r["layers_tagged"] for r in db.tables["articles"])
    ev = db.tables["layer_events"][0]
    assert ev["first_seen"] == old            # dated by the article, not by the run
    assert ev["notified_at"]                  # history never pings Telegram
    assert not cake.needs_notification(ev)
    assert asyncio.run(cake.backfill_layers(db, groq, "m", main.Article)) == 0   # nothing left
