"""Phase A — article tagging: Groq output validation, relevance filter, prompt."""

import asyncio
import json
from types import SimpleNamespace

import cake
import main


def test_parse_cake_fields_drops_unknown_codes_and_caps_at_three():
    layers, cross, scarcity = cake.parse_cake_fields(
        {"layers": ["energie", "Puces", "spatial", "energie", "modeles", "applications"],
         "cross_layer": True, "scarcity": "tension"}
    )
    assert layers == ["energie", "puces", "modeles"]
    assert cross is True
    assert scarcity == "tension"


def test_parse_cake_fields_defaults():
    assert cake.parse_cake_fields({}) == ([], False, "aucune")
    assert cake.parse_cake_fields({"layers": "energie", "scarcity": "énorme"}) == ([], False, "aucune")
    assert cake.parse_cake_fields({"layers": ["puces"], "cross_layer": "false", "scarcity": "détente"}) == (
        ["puces"], False, "detente")


def test_parse_cake_fields_needs_a_layer_for_a_move():
    assert cake.parse_cake_fields({"layers": [], "cross_layer": True, "scarcity": "tension"}) == ([], False, "aucune")


def test_layer_keywords_match_at_word_start():
    assert cake.matches_layer_keywords("Uranium prices hit a 10-year high", "energie")
    assert cake.matches_layer_keywords("Utilities struggle with new data centers", "energie")
    assert cake.matches_layer_keywords("La file d'attente de raccordement s'allonge", "energie")
    assert not cake.matches_layer_keywords("Solar panels for your balcony", "energie")
    assert not cake.matches_layer_keywords("anything", "modeles")  # no keyword list for this layer


def _article(title, source, description=""):
    return main.Article(title=title, url=f"https://x/{title}", source=source, country="🇺🇸",
                        published="2026-11-07", description=description)


def test_fetch_all_keeps_layer_articles_from_cake_sources(monkeypatch):
    fetched = [
        _article("Uranium prices hit a 10-year high", "Energy Feed"),       # cake source + keyword
        _article("Solar panels for your balcony", "Energy Feed"),           # cake source, no keyword
        _article("Uranium prices hit a 10-year high again", "Generic Feed"),  # keyword, no cake source
        _article("OpenAI ships a new model", "Generic Feed"),               # AI filter
    ]

    async def fake_fetch_rss(session, source):
        return [a for a in fetched if a.source == source["name"]]

    monkeypatch.setattr(main, "fetch_rss", fake_fetch_rss)
    sources = [
        {"name": "Energy Feed", "type": "rss", "url": "u1", "country": "🇺🇸", "cake_layer": "energie"},
        {"name": "Generic Feed", "type": "rss", "url": "u2", "country": "🇺🇸"},
    ]
    kept = [a.title for a in asyncio.run(main.fetch_all(sources))]
    assert kept == ["Uranium prices hit a 10-year high", "OpenAI ships a new model"]


def test_cake_keys_only_in_the_ia_prompt():
    assert '"layers"' in main._build_groq_prompt("ia")
    assert '"layers"' not in main._build_groq_prompt("politique_evenements")


class _FakeGroq:
    def __init__(self, payload):
        self.calls = []
        self.payload = payload
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.payload)))],
            usage=SimpleNamespace(total_tokens=123),
        )


def _classify(articles, payload, monkeypatch):
    groq = _FakeGroq(payload)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GROQ_MODEL", "full")
    monkeypatch.setenv("GROQ_MODEL_FAST", "fast")
    monkeypatch.setattr(main, "AsyncGroq", lambda **kw: groq)
    asyncio.run(main.classify_articles(articles))
    return groq


def test_classify_tags_ia_articles_with_the_full_model(monkeypatch):
    art = _article("EDF publie un modèle de fondation", "Le Monde IA")
    groq = _classify([art], {"articles": [
        {"id": 0, "category": "Business / Industrie", "sentiment": "Neutre", "country": "France",
         "summary": "EDF publie un modèle.", "layers": ["energie", "modeles"], "cross_layer": True,
         "scarcity": "aucune"}]}, monkeypatch)
    assert groq.calls[0]["model"] == "full"
    assert (art.layers, art.cross_layer, art.scarcity) == (["energie", "modeles"], True, "aucune")
    assert (art.category, art.country, art.summary) == ("Business / Industrie", "France", "EDF publie un modèle.")


def test_classify_leaves_other_domains_on_the_fast_model(monkeypatch):
    art = _article("Offensive au Soudan", "AP")
    art.domain = "politique_evenements"
    groq = _classify([art], {"articles": [
        {"id": 0, "category": "Conflits / Guerres", "sentiment": "Negatif", "country": "Soudan",
         "layers": ["energie"], "cross_layer": True}]}, monkeypatch)
    assert groq.calls[0]["model"] == "fast"
    assert (art.layers, art.cross_layer) == ([], False)


def test_classify_sends_20_articles_per_call(monkeypatch):
    arts = [_article(f"OpenAI news {i}", "Src") for i in range(25)]
    groq = _classify(arts, {"articles": [{"id": i, "sentiment": "Positif"} for i in range(20)]}, monkeypatch)
    assert len(groq.calls) == 2
    # The fake answers ids 0-19 each time: the second call's 5 articles are all found
    assert all(a.sentiment == "Positif" for a in arts)


def test_classify_batch_falls_back_for_missing_ids():
    arts = [_article("A", "S"), _article("B", "S")]
    groq = _FakeGroq({"articles": [{"id": 1, "category": "Business / Industrie", "sentiment": "Positif"}]})
    oks = asyncio.run(main._classify_batch(groq, "full", "ia", arts))
    assert oks == [False, True]
    assert (arts[0].sentiment, arts[0].country) == ("Neutre", "Global")
    assert arts[1].category == "Business / Industrie"


def test_cake_counts_html_lists_layers_top_down():
    import cake_dashboard
    html = cake_dashboard.cake_counts_html([{"layers": ["energie", "puces"]}, {"layers": ["energie"]}, {}])
    assert html.index("Applications") < html.index("Énergie")  # energy band at the bottom
    assert cake_dashboard.cake_counts_html([{"layers": []}]) == ""
