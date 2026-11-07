"""Phase D — value of a MWh at each floor (spec §6.5)."""

import layer_value


def _inputs(**over):
    base = {
        "electricity_us": {"value": 50, "confidence": "high"},
        "accelerator_price_usd": {"value": 30000, "confidence": "high"},
        "accelerator_tdp_kw": {"value": 0.7, "confidence": "high"},
        "pue": {"value": 1.5, "confidence": "high"},
        "lifetime_years": {"value": 5, "confidence": "high"},
        "utilization": {"value": 0.5, "confidence": "high"},
        "gpu_hour_usd": {"value": 2.1, "confidence": "high"},
        "token_price_usd_per_m": {"value": 1.0, "confidence": "high"},
        "energy_per_m_tokens_mwh": {"value": 0.001, "level": "chip", "confidence": "high"},
        "app_revenue_per_user_month_usd": {"value": 20, "confidence": "high"},
        "energy_per_prompt_wh": {"value": 0.3, "level": "meter", "confidence": "high"},
        "prompts_per_user_month": {"value": 1000, "confidence": "high"},
    }
    for k, v in over.items():
        base[k] = {**base[k], "value": v}
    return base


def _floors(**over):
    return {f["layer"]: f for f in layer_value.compute_floors(_inputs(**over))}


def test_pue_is_applied_to_chips_and_infrastructure():
    f = _floors()
    lifetime_mwh = 0.7 * 1.5 * 5 * 8760 * 0.5 / 1000
    assert abs(f["puces"]["price_usd_mwh"] - round(50 + 30000 / lifetime_mwh, 2)) < 0.01
    assert abs(f["infrastructure"]["price_usd_mwh"] - round(2.1 / (0.7 * 1.5 / 1000), 2)) < 0.01
    # doubling the PUE halves the energy-normalised infrastructure price
    assert abs(_floors(pue=3.0)["infrastructure"]["price_usd_mwh"] * 2 - f["infrastructure"]["price_usd_mwh"]) < 0.02


def test_chip_level_energy_is_brought_to_the_meter():
    f = _floors()
    assert abs(f["modeles"]["price_usd_mwh"] - round(1.0 / (0.001 * 1.5), 2)) < 0.01
    assert abs(f["applications"]["price_usd_mwh"] - round(20 / (0.3 * 1000 / 1e6), 2)) < 0.01  # meter: no PUE


def test_a_floor_selling_below_the_one_underneath_has_negative_value_added():
    f = _floors(token_price_usd_per_m=0.5)   # P4 = 333 < P3 = 2000
    m = f["modeles"]
    assert m["value_added_negative"] is True
    assert m["value_added_usd_mwh"] < 0                       # not clipped to zero
    assert abs(m["value_added_usd_mwh"] - (m["price_usd_mwh"] - f["infrastructure"]["price_usd_mwh"])) < 0.01
    assert not f["infrastructure"]["value_added_negative"]


def test_a_missing_input_removes_the_floor_instead_of_guessing():
    f = _floors(gpu_hour_usd=None)
    assert f["infrastructure"]["price_usd_mwh"] is None
    assert f["modeles"]["value_added_usd_mwh"] is None    # nothing to compare with
    assert f["modeles"]["price_usd_mwh"] is not None


def test_models_and_applications_are_always_low_confidence():
    f = _floors()
    assert f["energie"]["confidence"] == "high"
    assert f["modeles"]["confidence"] == f["applications"]["confidence"] == "low"
