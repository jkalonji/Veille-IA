"""Phase B — cross-layer events: spec §4.4 cases, scoring, merge, weights."""

import cake

ACTORS = {a["name"]: {**a, "weight": 0.6} for a in cake.load_manual_actors()}


def _ext(actor, target, engagement="annonce", **kw):
    return cake.clean_extraction({"actor": actor, "target_layer": target, "engagement": engagement,
                                  "what": kw.get("what", ""), "bypassed": kw.get("bypassed", []),
                                  "winners": [], "losers": [], "relations": kw.get("relations", [])})


def _apply(actor_name, target, engagement="annonce", events=None, today="2026-11-07", url="https://a/1"):
    events = [] if events is None else events
    ext = _ext(actor_name, target, engagement)
    actor = cake.resolve_actor(ext["actor"], cake.actor_index(list(ACTORS.values())))
    return cake.apply_extraction(ext, url, actor, events, today)


def test_edf_publishes_a_model_is_a_three_layer_climb():
    status, e = _apply("EDF", "modeles")
    assert status == "new"
    assert (e["origin_layer"], e["target_layer"], e["direction"], e["distance"], e["engagement"]) == (
        "energie", "modeles", "montee", 3, "annonce")


def test_openai_power_plant_is_a_three_layer_descent():
    ext = _ext("OpenAI", "energie", "mise_en_service", bypassed=["énergéticiens"])
    status, e = cake.apply_extraction(ext, "https://a/2", ACTORS["OpenAI"], [], "2026-11-07")
    assert (e["origin_layer"], e["direction"], e["distance"], e["engagement"]) == (
        "modeles", "descente", 3, "mise_en_service")
    assert "énergéticiens" in e["bypassed"]


def test_microsoft_ppa_starts_from_the_closest_home_layer():
    status, e = _apply("Microsoft", "energie", "contrat")
    assert (e["origin_layer"], e["direction"], e["distance"], e["engagement"]) == (
        "infrastructure", "descente", 2, "contrat")


def test_moves_inside_home_layers_are_not_events():
    assert _apply("Nvidia", "puces") == ("not_cross_layer", None)
    assert _apply("OpenAI", "modeles") == ("not_cross_layer", None)


def test_announce_then_contract_40_days_later_is_one_progressed_event():
    events = []
    _apply("EDF", "modeles", "annonce", events, today="2026-09-28", url="https://a/1")
    status, e = _apply("EDF", "modeles", "contrat", events, today="2026-11-07", url="https://a/2")
    assert status == "progressed"
    assert len(events) == 1
    assert e["engagement"] == "contrat" and e["progressed_at"] == "2026-11-07"
    assert e["article_urls"] == ["https://a/1", "https://a/2"]
    assert [h["engagement"] for h in e["engagement_history"]] == ["annonce", "contrat"]


def test_same_move_after_the_merge_window_is_a_new_event():
    events = []
    _apply("EDF", "modeles", events=events, today="2026-01-01")
    status, _ = _apply("EDF", "modeles", events=events, today="2026-11-07")
    assert status == "new" and len(events) == 2


def test_tie_between_two_origins_takes_the_lower_layer():
    assert cake.locate_move(["energie", "infrastructure"], "puces") == ("energie", "montee", 1, True)


def test_score_formula_and_novelty():
    score, bd = cake.event_score(3, "annonce", 0.6, novel=True)
    assert score == min(10, round(10 * (0.35 * 0.85 + 0.30 * 0.40 + 0.35 * 0.6) * 1.2, 1)) == 7.5
    assert bd["N"] == 1.2
    events = []
    _, first = _apply("EDF", "modeles", events=events)
    _, second = _apply("Engie", "modeles", events=events)  # same energie → modeles pair
    assert first["score_breakdown"]["N"] == 1.2 and second["score_breakdown"]["N"] == 1.0


def test_major_event_is_notified_once_then_once_per_progression():
    events = []
    _, e = _apply("EDF", "modeles", events=events, today="2026-11-01")
    assert e["is_major"] and cake.needs_notification(e)
    e["notified_at"] = "2026-11-01T07:10:00+00:00"
    _apply("EDF", "modeles", events=events, today="2026-11-03", url="https://a/9")  # same engagement
    assert not cake.needs_notification(e)
    _apply("EDF", "modeles", "contrat", events=events, today="2026-11-05")
    assert cake.needs_notification(e)
    assert "progression" in cake.major_event_message(e)


def test_major_event_message_format():
    _, e = _apply("EDF", "modeles")
    e["what"], e["losers"], e["winners"] = "EDF publie un modèle de fondation", ["labos américains"], ["fournisseurs de GPU"]
    msg = cake.major_event_message(e, "https://dash.example")
    assert "⬆️ EDF : Énergie → Modèles (3 couches)" in msg
    assert "🔖 Annonce · Score 7,5 (nouveau franchissement)" in msg
    assert "📉 Perdants : labos américains" in msg and "#cake" in msg


def test_home_layer_promoted_30_days_after_going_into_service():
    actor = {"name": "OpenAI", "home_layers": ["modeles", "applications"]}
    event = {"actor_name": "OpenAI", "target_layer": "energie", "engagement": "mise_en_service",
             "engagement_history": [{"engagement": "mise_en_service", "date": "2026-10-01"}]}
    assert cake.promote_home_layers([event], {"OpenAI": actor}, "2026-10-20") == []
    assert cake.promote_home_layers([event], {"OpenAI": actor}, "2026-10-31") == [actor]
    assert actor["home_layers"] == ["modeles", "applications", "energie"]


def test_actor_resolution_uses_aliases_and_accents():
    index = cake.actor_index(list(ACTORS.values()))
    assert cake.resolve_actor("Électricité de France", index)["name"] == "EDF"
    assert cake.resolve_actor("Google DeepMind", index)["name"] == "Alphabet (Google)"
    assert cake.resolve_actor("Anysphere", index)["name"] == "Cursor (Anysphere)"
    assert cake.resolve_actor("Some Startup", index) is None


def test_weight_components():
    assert cake.size_component(500) == 1.0
    assert 0.38 < cake.size_component(10) < 0.39
    assert cake.longevity_component(2024, 2026) == 0.2
    assert cake.longevity_component(2016, 2026) == 10 / 15
    assert cake.longevity_component(2024, 2026, parent_founded=1892) == 0.8
    w, conf = cake.combine_weight({"size": 1.0, "influence": None, "longevity": 0.5, "implantation": None})
    assert w == round((0.30 * 1.0 + 0.15 * 0.5) / 0.45, 4) and conf == "low"
    assert cake.combine_weight({"size": None}) == (None, "low")


def test_pagerank_favours_the_most_cited_actor():
    edges = cake.comention_edges([{"A", "B"}, {"A", "C"}, {"A", "D"}, {"B", "C"}])
    pr = cake.pagerank(["A", "B", "C", "D", "E"], edges)
    assert abs(sum(pr.values()) - 1) < 1e-6
    assert max(pr, key=pr.get) == "A"
    ranks = cake.percentile_ranks(pr)
    assert ranks["A"] == 1.0 and ranks["E"] == 0.0


def test_extraction_validation():
    assert cake.clean_extraction(None) is None
    assert cake.clean_extraction({"actor": "X", "target_layer": "spatial"}) is None
    e = cake.clean_extraction({"actor": "X", "target_layer": "energie", "engagement": "rumeur",
                               "relations": [{"from": "X", "relation": "achete", "to": "Y"},
                                             {"from": "X", "relation": "utilise", "to": "Nvidia"}]})
    assert e["engagement"] == "annonce"
    assert e["relations"] == [{"from": "X", "relation": "utilise", "to": "Nvidia"}]


def test_digest_line():
    assert cake.digest_line([]) == ""
    assert cake.digest_line([{"is_major": True}, {"is_major": False}]) == "🎂 Cake : 2 mouvements (1 majeur)"
