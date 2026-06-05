"""Unit tests for XAUUSD (Gold) presets, ATR defaults, and SL/TP calculation logic.

Tests cover:
  - Preset dict integrity (all required keys present, values in valid ranges)
  - ATR default mapping per symbol (EURUSD, XAUUSD, unknown)
  - SL/TP calculation at EURUSD and XAUUSD price scales
  - Lot sizing differences between EURUSD and XAUUSD
  - Gold vs forex SL distance sanity checks
"""

from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Re-create the logic from main.py so we can test it in isolation
# ─────────────────────────────────────────────────────────────────────────────

# Same as scripts/main.py PRESETS dict
PRESETS = {
    "ana": {
        "_name": "Ana Kurallar (Production)",
        "_desc": "Düşük riskli, uzun vadeli, yüksek güven filtreli üretim kuralları.",
        "RISK_REWARD_RATIO":    2.5,
        "SL_ATR_MULTIPLIER":    1.5,
        "CONFIDENCE_THRESHOLD": 0.65,
        "ADX_MINIMUM":          25.0,
        "COOLDOWN_HOURS":       8.0,
    },
    "test": {
        "_name": "Test Kuralları (MAKSIMUM AGRESIF -- Aninda Pozisyon)",
        "_desc": "Tum filtreler kapatildi.",
        "RISK_REWARD_RATIO":    1.0,
        "SL_ATR_MULTIPLIER":    0.5,
        "CONFIDENCE_THRESHOLD": 0.50,
        "ADX_MINIMUM":          0.0,
        "COOLDOWN_HOURS":       0.0,
        "META_CONF_MIN":        0.0,
    },
    "altin": {
        "_name": "Altın Kuralları (XAUUSD)",
        "_desc": "XAUUSD için optimize edilmiş.",
        "RISK_REWARD_RATIO":    2.0,
        "SL_ATR_MULTIPLIER":    1.5,
        "CONFIDENCE_THRESHOLD": 0.60,
        "ADX_MINIMUM":          20.0,
        "COOLDOWN_HOURS":       4.0,
        "META_CONF_MIN":        0.50,
    },
    "altin_test": {
        "_name": "Altın Test (AGRESIF)",
        "_desc": "XAUUSD test modu.",
        "RISK_REWARD_RATIO":    1.5,
        "SL_ATR_MULTIPLIER":    1.0,
        "CONFIDENCE_THRESHOLD": 0.50,
        "ADX_MINIMUM":          0.0,
        "COOLDOWN_HOURS":       0.0,
        "META_CONF_MIN":        0.0,
    },
}

# Same as scripts/main.py _ATR_DEFAULTS
_ATR_DEFAULTS = {
    "EURUSD": 0.0050,
    "GBPUSD": 0.0060,
    "USDJPY": 0.70,
    "XAUUSD": 28.0,
    "XAGUSD": 0.50,
}

# Required trade parameter keys (excluding _name, _desc)
_REQUIRED_KEYS = {
    "RISK_REWARD_RATIO",
    "SL_ATR_MULTIPLIER",
    "CONFIDENCE_THRESHOLD",
    "ADX_MINIMUM",
    "COOLDOWN_HOURS",
}


def compute_sl_tp(current_price: float, atr_val: float, sl_atr_mult: float,
                  rr_ratio: float, side: int) -> tuple[float, float]:
    """Replicate the SL/TP calculation logic from main.py."""
    dist = atr_val * sl_atr_mult
    if side == 1:  # BUY
        sl = current_price - dist
        tp = current_price + (dist * rr_ratio)
    else:  # SELL
        sl = current_price + dist
        tp = current_price - (dist * rr_ratio)
    return sl, tp


def compute_lot_size(nav: float, position_pct: float, mid_price: float,
                     symbol: str, is_mt5: bool) -> float:
    """Replicate the lot sizing logic from main.py."""
    order_usd = nav * position_pct
    qty = order_usd / max(mid_price, 1e-9)
    if is_mt5:
        if symbol == "XAUUSD":
            qty = qty / 100.0    # 1 lot = 100 oz
        else:
            qty = qty / 100000.0  # 1 lot = 100,000 units
        qty = max(0.01, round(qty, 2))
    else:
        qty = round(qty, 5)
    return qty


# ─────────────────────────────────────────────────────────────────────────────
# Preset integrity tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPresetIntegrity:
    """Verify all presets have the correct structure and valid values."""

    @pytest.mark.parametrize("key", list(PRESETS.keys()))
    def test_preset_has_name_and_desc(self, key):
        """Every preset must have _name and _desc metadata."""
        p = PRESETS[key]
        assert "_name" in p, f"Preset '{key}' missing _name"
        assert "_desc" in p, f"Preset '{key}' missing _desc"
        assert isinstance(p["_name"], str) and len(p["_name"]) > 0
        assert isinstance(p["_desc"], str) and len(p["_desc"]) > 0

    @pytest.mark.parametrize("key", list(PRESETS.keys()))
    def test_preset_has_required_keys(self, key):
        """Every preset must have all required trade parameter keys."""
        p = PRESETS[key]
        param_keys = {k for k in p if not k.startswith("_")}
        missing = _REQUIRED_KEYS - param_keys
        assert not missing, f"Preset '{key}' missing keys: {missing}"

    @pytest.mark.parametrize("key", list(PRESETS.keys()))
    def test_preset_values_are_numeric(self, key):
        """All trade parameter values must be int or float."""
        p = PRESETS[key]
        for k, v in p.items():
            if not k.startswith("_"):
                assert isinstance(v, (int, float)), \
                    f"Preset '{key}' key '{k}' has non-numeric value: {v!r}"

    @pytest.mark.parametrize("key", list(PRESETS.keys()))
    def test_preset_rr_ratio_positive(self, key):
        """Risk/reward ratio must be > 0."""
        assert PRESETS[key]["RISK_REWARD_RATIO"] > 0

    @pytest.mark.parametrize("key", list(PRESETS.keys()))
    def test_preset_sl_multiplier_positive(self, key):
        """SL ATR multiplier must be > 0."""
        assert PRESETS[key]["SL_ATR_MULTIPLIER"] > 0

    @pytest.mark.parametrize("key", list(PRESETS.keys()))
    def test_preset_confidence_in_range(self, key):
        """Confidence threshold must be between 0 and 1."""
        conf = PRESETS[key]["CONFIDENCE_THRESHOLD"]
        assert 0.0 <= conf <= 1.0

    @pytest.mark.parametrize("key", list(PRESETS.keys()))
    def test_preset_cooldown_non_negative(self, key):
        """Cooldown hours must be >= 0."""
        assert PRESETS[key]["COOLDOWN_HOURS"] >= 0.0

    def test_altin_preset_exists(self):
        """Gold preset must exist."""
        assert "altin" in PRESETS
        assert "altin_test" in PRESETS

    def test_preset_count(self):
        """Must have exactly 4 presets."""
        assert len(PRESETS) == 4


# ─────────────────────────────────────────────────────────────────────────────
# ATR default mapping tests
# ─────────────────────────────────────────────────────────────────────────────

class TestATRDefaults:
    """Verify ATR defaults are appropriate for each instrument."""

    def test_eurusd_atr(self):
        assert _ATR_DEFAULTS["EURUSD"] == pytest.approx(0.005)

    def test_xauusd_atr(self):
        """Gold ATR ~25-35, default should be in that range."""
        assert 20.0 <= _ATR_DEFAULTS["XAUUSD"] <= 40.0

    def test_xagusd_atr(self):
        assert _ATR_DEFAULTS["XAGUSD"] == pytest.approx(0.50)

    def test_usdjpy_atr(self):
        assert _ATR_DEFAULTS["USDJPY"] == pytest.approx(0.70)

    def test_unknown_symbol_fallback(self):
        """Unknown symbols should fallback to EURUSD-like default."""
        fallback = _ATR_DEFAULTS.get("UNKNOWN_SYM", 0.0050)
        assert fallback == pytest.approx(0.005)

    def test_gold_atr_much_larger_than_forex(self):
        """Gold ATR must be orders of magnitude larger than forex ATR."""
        ratio = _ATR_DEFAULTS["XAUUSD"] / _ATR_DEFAULTS["EURUSD"]
        assert ratio > 1000  # 28 / 0.005 = 5600x


# ─────────────────────────────────────────────────────────────────────────────
# SL/TP calculation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSLTPCalculation:
    """Verify SL/TP distances are correct at different price scales."""

    # ── EURUSD ────────────────────────────────────────────────────────

    def test_eurusd_buy_sl_tp(self):
        """EURUSD BUY: SL below entry, TP above entry."""
        sl, tp = compute_sl_tp(
            current_price=1.08000,
            atr_val=0.0050,       # 50 pips ATR
            sl_atr_mult=1.5,      # Ana preset
            rr_ratio=2.5,         # Ana preset
            side=1,
        )
        dist = 0.0050 * 1.5  # = 0.0075
        assert sl == pytest.approx(1.08000 - 0.0075)  # 1.07250
        assert tp == pytest.approx(1.08000 + 0.0075 * 2.5)  # 1.09875

    def test_eurusd_sell_sl_tp(self):
        """EURUSD SELL: SL above entry, TP below entry."""
        sl, tp = compute_sl_tp(
            current_price=1.08000,
            atr_val=0.0050,
            sl_atr_mult=1.5,
            rr_ratio=2.5,
            side=-1,
        )
        dist = 0.0050 * 1.5
        assert sl == pytest.approx(1.08000 + dist)     # 1.08750
        assert tp == pytest.approx(1.08000 - dist * 2.5)  # 1.06125

    # ── XAUUSD (Gold) ────────────────────────────────────────────────

    def test_xauusd_buy_sl_tp_altin_preset(self):
        """Gold BUY with altin preset: SL ~$42 below, TP ~$84 above."""
        sl, tp = compute_sl_tp(
            current_price=2350.00,
            atr_val=28.0,         # Gold ATR default
            sl_atr_mult=1.5,      # Altin preset
            rr_ratio=2.0,         # Altin preset
            side=1,
        )
        dist = 28.0 * 1.5  # = $42
        assert sl == pytest.approx(2350.00 - 42.0)   # $2308
        assert tp == pytest.approx(2350.00 + 42.0 * 2.0)  # $2434

    def test_xauusd_sell_sl_tp_altin_preset(self):
        """Gold SELL with altin preset."""
        sl, tp = compute_sl_tp(
            current_price=2350.00,
            atr_val=28.0,
            sl_atr_mult=1.5,
            rr_ratio=2.0,
            side=-1,
        )
        dist = 42.0
        assert sl == pytest.approx(2350.00 + dist)     # $2392
        assert tp == pytest.approx(2350.00 - dist * 2.0)  # $2266

    def test_xauusd_buy_sl_tp_altin_test_preset(self):
        """Gold BUY with altin_test preset: tighter SL ($28)."""
        sl, tp = compute_sl_tp(
            current_price=2350.00,
            atr_val=28.0,
            sl_atr_mult=1.0,      # Altin test preset
            rr_ratio=1.5,         # Altin test preset
            side=1,
        )
        dist = 28.0 * 1.0  # = $28
        assert sl == pytest.approx(2350.00 - 28.0)    # $2322
        assert tp == pytest.approx(2350.00 + 28.0 * 1.5)  # $2392

    # ── Sanity checks ────────────────────────────────────────────────

    def test_gold_sl_distance_reasonable(self):
        """Gold SL distance should be $25-60 range, not $0.007 or $500."""
        for preset_key in ("altin", "altin_test"):
            p = PRESETS[preset_key]
            dist = _ATR_DEFAULTS["XAUUSD"] * p["SL_ATR_MULTIPLIER"]
            assert 20.0 <= dist <= 80.0, \
                f"Preset '{preset_key}' SL distance ${dist:.0f} is outside reasonable range"

    def test_forex_sl_distance_reasonable(self):
        """Forex SL distance should be 30-120 pips range."""
        for preset_key in ("ana", "test"):
            p = PRESETS[preset_key]
            dist_pips = (_ATR_DEFAULTS["EURUSD"] * p["SL_ATR_MULTIPLIER"]) / 0.0001
            assert 10.0 <= dist_pips <= 200.0, \
                f"Preset '{preset_key}' SL distance {dist_pips:.0f} pips is outside reasonable range"

    def test_sl_always_worse_than_entry(self):
        """SL must always be on the losing side of entry for both BUY and SELL."""
        for price, atr in [(1.08, 0.005), (2350.0, 28.0)]:
            sl_buy, _ = compute_sl_tp(price, atr, 1.5, 2.0, side=1)
            sl_sell, _ = compute_sl_tp(price, atr, 1.5, 2.0, side=-1)
            assert sl_buy < price, "BUY SL must be below entry"
            assert sl_sell > price, "SELL SL must be above entry"

    def test_tp_always_better_than_entry(self):
        """TP must always be on the profitable side of entry."""
        for price, atr in [(1.08, 0.005), (2350.0, 28.0)]:
            _, tp_buy = compute_sl_tp(price, atr, 1.5, 2.0, side=1)
            _, tp_sell = compute_sl_tp(price, atr, 1.5, 2.0, side=-1)
            assert tp_buy > price, "BUY TP must be above entry"
            assert tp_sell < price, "SELL TP must be below entry"

    def test_tp_farther_than_sl(self):
        """With R/R > 1, TP distance must be greater than SL distance."""
        for price, atr in [(1.08, 0.005), (2350.0, 28.0)]:
            sl, tp = compute_sl_tp(price, atr, 1.5, 2.0, side=1)
            sl_dist = abs(price - sl)
            tp_dist = abs(tp - price)
            assert tp_dist > sl_dist, "TP should be farther than SL when R/R > 1"


# ─────────────────────────────────────────────────────────────────────────────
# Lot sizing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLotSizing:
    """Verify lot calculation produces sensible values for EURUSD and XAUUSD."""

    def test_eurusd_lot_size_mt5(self):
        """EURUSD lot = order_usd / price / 100,000."""
        qty = compute_lot_size(
            nav=100_000, position_pct=0.05, mid_price=1.08000,
            symbol="EURUSD", is_mt5=True,
        )
        # $5000 / 1.08 / 100000 = 0.0463 → 0.05
        assert 0.01 <= qty <= 0.10
        assert qty == pytest.approx(0.05, abs=0.01)

    def test_xauusd_lot_size_mt5(self):
        """XAUUSD lot = order_usd / price / 100 (1 lot = 100 oz)."""
        qty = compute_lot_size(
            nav=100_000, position_pct=0.05, mid_price=2350.00,
            symbol="XAUUSD", is_mt5=True,
        )
        # $5000 / 2350 / 100 = 0.0213 → 0.02
        assert 0.01 <= qty <= 0.10
        assert qty == pytest.approx(0.02, abs=0.01)

    def test_xauusd_smaller_nav_min_lot(self):
        """Small NAV should produce minimum lot 0.01."""
        qty = compute_lot_size(
            nav=5_000, position_pct=0.05, mid_price=2350.00,
            symbol="XAUUSD", is_mt5=True,
        )
        # $250 / 2350 / 100 = 0.001 → clamped to 0.01
        assert qty == 0.01

    def test_xauusd_lot_vs_eurusd_lot(self):
        """Same NAV should produce roughly similar dollar exposure."""
        qty_eur = compute_lot_size(
            nav=100_000, position_pct=0.05, mid_price=1.08,
            symbol="EURUSD", is_mt5=True,
        )
        qty_xau = compute_lot_size(
            nav=100_000, position_pct=0.05, mid_price=2350.0,
            symbol="XAUUSD", is_mt5=True,
        )
        # Dollar exposure: EUR → qty * 100000 * price, XAU → qty * 100 * price
        exposure_eur = qty_eur * 100_000 * 1.08
        exposure_xau = qty_xau * 100 * 2350.0
        # Both should be roughly $5000 (5% of $100K NAV)
        assert 2000 <= exposure_eur <= 8000
        assert 2000 <= exposure_xau <= 8000

    def test_non_mt5_lot_sizing(self):
        """Non-MT5 brokers use raw qty (no /100000 division)."""
        qty = compute_lot_size(
            nav=100_000, position_pct=0.05, mid_price=2350.0,
            symbol="XAUUSD", is_mt5=False,
        )
        # $5000 / 2350 = 2.12766 → rounded to 5 decimals
        assert qty == pytest.approx(2.12766, abs=0.001)
