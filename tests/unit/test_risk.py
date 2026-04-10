"""Unit tests for Risk Engine."""

import pytest
from borsabot.risk.engine import RiskEngine, RiskLimits


@pytest.fixture
def engine():
    limits = RiskLimits(
        max_position_usd=10_000,
        max_portfolio_pct=0.10,
        max_drawdown_pct=0.05,
        vol_target_annual=0.15,
        kelly_fraction=0.25,
    )
    return RiskEngine(limits=limits, nav=100_000)


def test_order_allowed_within_limits(engine):
    ok, msg = engine.check_new_order("BTCUSDT", 5_000)
    assert ok
    assert msg == "OK"


def test_order_blocked_by_position_limit(engine):
    ok, msg = engine.check_new_order("BTCUSDT", 15_000)
    assert not ok
    assert "Position limit" in msg


def test_order_blocked_by_nav_pct():
    # Use a high max_position_usd so only NAV% fires
    limits = RiskLimits(
        max_position_usd=100_000,
        max_portfolio_pct=0.10,
    )
    e = RiskEngine(limits=limits, nav=100_000)
    # 10_001 / 100_000 = 10.001% > 10% limit
    ok, msg = e.check_new_order("BTCUSDT", 10_001)
    assert not ok
    assert "NAV%" in msg


def test_drawdown_halt_triggers(engine):
    # Lose 5% of NAV
    engine.update_pnl(-5_001)
    ok, msg = engine.check_new_order("BTCUSDT", 100)
    assert not ok
    assert "halted" in msg.lower()


def test_daily_reset_clears_halt(engine):
    engine.update_pnl(-5_001)
    engine.reset_daily()
    ok, _ = engine.check_new_order("BTCUSDT", 1_000)
    assert ok


def test_vol_adjusted_size_decreases_with_vol(engine):
    # Low vol: 0.005/day → annualized ≈ 0.079 → position = 100k * 0.15 / 0.079 ≈ 189k but capped at 10k
    # We need a case where capping doesn't happen:
    # Use a 10M nav so that neither case hits the 10k cap
    limits = RiskLimits(max_position_usd=200_000)
    big_engine = RiskEngine(limits=limits, nav=1_000_000)
    size_low  = big_engine.vol_adjusted_size(0.005)   # low vol → bigger position
    size_high = big_engine.vol_adjusted_size(0.05)    # high vol → smaller position
    assert size_low > size_high


def test_vol_adjusted_size_capped(engine):
    # Even at very low vol, size should be capped at max_position_usd
    size = engine.vol_adjusted_size(0.0001)
    assert size <= engine.limits.max_position_usd + 1  # allow tiny fp error


def test_kelly_size_zero_when_no_edge(engine):
    assert engine.kelly_size(win_rate=0.0, avg_win=1.0, avg_loss=1.0) == 0.0


def test_kelly_size_positive_edge(engine):
    size = engine.kelly_size(win_rate=0.6, avg_win=1.0, avg_loss=1.0)
    assert size > 0


def test_kelly_size_capped_at_max(engine):
    # Huge win rate should still be capped
    size = engine.kelly_size(win_rate=0.99, avg_win=10.0, avg_loss=1.0)
    assert size <= engine.limits.max_position_usd


def test_portfolio_summary_structure(engine):
    engine.update_position("BTCUSDT", 5_000)
    summary = engine.portfolio_summary()
    assert "nav" in summary
    assert "total_exposure" in summary
    assert "daily_pnl" in summary
    assert summary["total_exposure"] == pytest.approx(5_000)
