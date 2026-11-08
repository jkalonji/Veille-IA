"""Every chart has its explanation, and quoted thresholds match the config."""

import cake
import chart_info
from dashboard import INDICATOR_TABS


def _text(key: str) -> str:
    return " ".join(chart_info.INFO[key]["method"] + chart_info.INFO[key]["reading"])


def test_every_chart_has_method_and_reading():
    keys = {"cake", "cake_counts", "cake_events", "matrix", "value_staircase", "layer_curves",
            "globe", "categories", "story_cloud"} | {k for k, _ in INDICATOR_TABS}
    assert keys <= set(chart_info.INFO)
    for entry in chart_info.INFO.values():
        assert entry["title"] and entry["method"] and entry["reading"]


def test_quoted_thresholds_match_config():
    cfg = cake.load_config()
    score, neck = cfg["event_score"], cfg["bottleneck"]
    assert f"Score ≥ {score['major_threshold']} " in _text("cake_events")
    assert f"Score ≥ {score['strong_signal_threshold']} " in _text("cake_events")
    assert f"×{score['novelty_bonus']}".replace(".", ",") in _text("cake_events")
    assert f"depuis {score['novelty_window_days']} jours" in _text("cake_events")
    z = f"z = {neck['z_threshold']}".replace(".", ",")
    assert z in _text("layer_curves")
    assert f"pendant {neck['min_days']} jours" in _text("layer_curves")


def test_score_coefficients_match_config():
    coeffs = cake.load_config()["actor_weight_coefficients"]
    text = _text("cake_events")
    for label, key in (("taille", "size"), ("influence", "influence"),
                       ("longévité", "longevity"), ("implantation", "implantation")):
        assert f"{label} {coeffs[key] * 100:.0f} %" in text


def test_rendering_escapes_html_and_latex():
    html = chart_info.info_html("hardware_imports")
    assert html.startswith('<details class="chart-info">') and "<h4>Method</h4>" in html
    md = chart_info.info_markdown("hardware_imports")
    assert r"\$50bn" in md
