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
    assert '"layers"' in main._build_groq_prompt("ia", with_summary=False)
    assert '"layers"' not in main._build_groq_prompt("politique_evenements", with_summary=False)


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


def test_classify_one_tags_ia_articles_with_the_full_model():
    groq = _FakeGroq({"category": "Business / Industrie", "sentiment": "Neutre", "country": "France",
                      "layers": ["energie", "modeles"], "cross_layer": True, "scarcity": "aucune"})
    art = _article("EDF publie un modèle de fondation", "Le Monde IA")
    assert asyncio.run(main._classify_one(groq, "fast", "full", art))
    assert groq.calls[0]["model"] == "full"
    assert groq.calls[0]["max_tokens"] == 350
    assert (art.layers, art.cross_layer, art.scarcity) == (["energie", "modeles"], True, "aucune")


def test_classify_one_leaves_other_domains_on_the_fast_model():
    groq = _FakeGroq({"category": "Conflits / Guerres", "sentiment": "Negatif", "country": "Soudan",
                      "layers": ["energie"], "cross_layer": True})
    art = _article("Offensive au Soudan", "AP")
    art.domain = "politique_evenements"
    asyncio.run(main._classify_one(groq, "fast", "full", art))
    assert groq.calls[0]["model"] == "fast"
    assert (art.layers, art.cross_layer) == ([], False)


def test_cake_counts_html_lists_layers_top_down():
    import cake_dashboard
    html = cake_dashboard.cake_counts_html([{"layers": ["energie", "puces"]}, {"layers": ["energie"]}, {}])
    assert html.index("Applications") < html.index("Énergie")  # energy band at the bottom
    assert cake_dashboard.cake_counts_html([{"layers": []}]) == ""
