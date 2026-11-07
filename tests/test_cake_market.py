"""Phase C — baskets, abnormal returns, scarcity z-score, bottlenecks."""

from datetime import date, timedelta

import cake_market as cm


def _days(n, start="2026-01-01"):
    return [(date.fromisoformat(start) + timedelta(days=k)).isoformat() for k in range(n)]


def test_basket_is_equal_weight_in_returns_whatever_the_currency():
    closes = {"A": {"2026-01-01": 10, "2026-01-02": 11}, "B": {"2026-01-01": 1000, "2026-01-02": 990}}
    rets = cm.basket_returns(closes, ["A", "B"])
    assert abs(rets["2026-01-02"] - (0.10 - 0.01) / 2) < 1e-12
    index = cm.chain_index(rets, 100.0)
    assert abs(index["2026-01-02"] - 104.5) < 1e-9


def test_chain_index_continues_from_the_last_stored_value():
    rets = {"2026-01-02": 0.1, "2026-01-03": 0.1}
    assert cm.chain_index(rets, 200.0, after="2026-01-02") == {"2026-01-03": 200.0 * 1.1}


def test_abnormal_return_window_j0_j1():
    s = [("2026-01-01", 100), ("2026-01-02", 100), ("2026-01-05", 110), ("2026-01-06", 120)]
    b = [("2026-01-01", 50), ("2026-01-02", 50), ("2026-01-05", 50), ("2026-01-06", 51)]
    # event on Saturday 01-03: last close before = 01-02 (100), second close from it = 01-06 (120)
    ar = cm.abnormal_return(s, b, "2026-01-03")
    assert abs(ar - (0.20 - 0.02)) < 1e-12
    assert cm.abnormal_return(s, b, "2026-01-06") is None  # no J+1 yet


def test_change_pct_uses_the_last_value_on_or_before_each_date():
    s = [("2026-01-01", 100.0), ("2026-01-15", 105.0), ("2026-02-01", 115.0)]
    assert abs(cm.change_pct(s, "2026-02-02", 30) - 15.0) < 1e-9
    assert cm.change_pct([], "2026-02-02", 30) is None


def test_scarcity_counts_detente_separately_from_tension():
    arts = [
        {"published": "2026-01-01", "layers": ["puces"], "scarcity": "tension"},
        {"published": "2026-01-01", "layers": ["puces", "infrastructure"], "scarcity": "detente"},
        {"published": "2026-01-01", "layers": ["puces"], "scarcity": "aucune"},
        {"published": "2026-01-01", "layers": ["energie"], "scarcity": None, "title": "Transformer shortage worsens"},
    ]
    c = cm.daily_layer_counts(arts)
    assert c["puces"]["2026-01-01"] == [3, 1, 1]
    assert c["energie"]["2026-01-01"] == [1, 0.5, 0]  # vocabulary fallback, weak tension


def test_z_score_against_the_layer_own_baseline():
    days = _days(60)
    counts = {d: [10, 2, 0] for d in days[:-1]}        # a steady 20 % tension...
    counts.update({d: [10, 2 + (i % 2), 0] for i, d in enumerate(days[:-1])})  # with a little noise
    counts[days[-1]] = [10, 9, 0]                      # ...then a spike
    s = cm.scarcity_series(counts, days, baseline=90, min_baseline=30)
    assert s[days[10]]["scarcity_z"] is None            # not enough baseline yet
    assert s[days[-1]]["scarcity_z"] > 1.5
    assert cm.gauge(-1) == 0 and cm.gauge(4) == 1 and cm.gauge(1.5) == 0.5


def test_bottleneck_needs_three_days_above_threshold_and_a_stable_baseline():
    z = {"energie": [1.6, 1.7, 2.0], "puces": [0.2, 2.5, 2.6], "modeles": [None, None, 0.1]}
    assert cm.signalled_layer(z, set()) is None              # puces is highest but only 2 days
    z["puces"] = [1.6, 2.5, 2.6]
    assert cm.signalled_layer(z, set()) == "puces"
    assert cm.signalled_layer(z, {"puces"}) == "energie"     # puces' baseline being rebuilt


def test_price_confirmation_rules():
    day = "2026-03-01"
    series = {"CAKE_PUCES": [("2026-01-30", 100.0), (day, 112.0)], "QQQ": [("2026-01-30", 100.0), (day, 104.0)]}
    assert "pts vs QQQ" in cm.price_confirmation("puces", series, day)
    series["CAKE_PUCES"][-1] = (day, 107.0)
    assert cm.price_confirmation("puces", series, day) is None
    elec = {"ELEC_US": [("2026-01-15", 60.0), ("2026-02-15", 70.0)]}
    assert "électricité" in cm.price_confirmation("energie", elec, day)
    tokens = {cm.TOKEN_OUT: [("2026-01-20", 2.0), (day, 2.2)]}
    assert "tokens" in cm.price_confirmation("modeles", tokens, day)
    assert cm.price_confirmation("applications", series, day) is None


def test_migration_needs_seven_days_on_the_new_layer():
    days = _days(12)
    hist = [(d, "puces") for d in days[:4]] + [(days[4], None)] + [(d, "energie") for d in days[5:]]
    assert cm.migration(hist, 7) == ("puces", "energie")
    assert cm.migration(hist[:-1], 7) is None               # only 6 days on energie
    assert cm.migration([(d, "energie") for d in days], 7) is None  # no change


def test_sources_fingerprint_only_moves_for_the_layers_concerned():
    base = [{"name": "Generic"}, {"name": "Energy feed", "cake_layer": "energie"}]
    more = base + [{"name": "DC feed", "cake_layer": "infrastructure"}]
    assert cm.sources_fingerprint(base, "energie") == cm.sources_fingerprint(more, "energie")
    assert cm.sources_fingerprint(base, "infrastructure") != cm.sources_fingerprint(more, "infrastructure")
