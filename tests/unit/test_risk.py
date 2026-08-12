"""Unit tests for Risk Engine."""

from concurrent.futures import ThreadPoolExecutor

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


def test_reducing_order_not_blocked(engine):
    """An order that reduces an existing position must never hit the size limit."""
    # Open a long at the max position size.
    engine.update_position("BTCUSDT", 10_000, price=100.0)
    # Selling 5k reduces net to 5k — allowed even though abs(10k)+abs(5k) > limit.
    ok, msg = engine.check_new_order("BTCUSDT", -5_000)
    assert ok, msg


def test_increasing_order_still_blocked(engine):
    """Adding to a position past the limit is still blocked."""
    engine.update_position("BTCUSDT", 8_000, price=100.0)
    ok, msg = engine.check_new_order("BTCUSDT", 5_000)   # net 13k > 10k
    assert not ok
    assert "Position limit" in msg


def test_mark_to_market_trips_drawdown_halt(engine):
    """Unrealized loss on an open position must trip the daily drawdown halt."""
    engine.update_position("BTCUSDT", 50_000, price=100.0)
    # -11% move on a 50k long = -5_500 unrealized > 5% of 100k NAV.
    delta, halted = engine.mark_to_market("BTCUSDT", 89.0)
    assert delta == pytest.approx(-5_500.0)
    assert halted
    ok, msg = engine.check_new_order("BTCUSDT", 100)
    assert not ok and "halted" in msg.lower()


def test_mark_to_market_feeds_only_incremental_delta(engine):
    """Re-marking at the same price must not double-count the PnL."""
    engine.update_position("BTCUSDT", 50_000, price=100.0)
    d1, _ = engine.mark_to_market("BTCUSDT", 98.0)
    d2, _ = engine.mark_to_market("BTCUSDT", 98.0)
    assert d1 == pytest.approx(-1_000.0)
    assert d2 == pytest.approx(0.0)


def test_mark_to_market_gain_does_not_halt(engine):
    engine.update_position("BTCUSDT", 50_000, price=100.0)
    delta, halted = engine.mark_to_market("BTCUSDT", 110.0)
    assert delta == pytest.approx(5_000.0)
    assert not halted


def test_check_and_reserve_updates_position(engine):
    ok, msg, token = engine.check_and_reserve("BTCUSDT", 5_000, price=100.0)
    assert ok and token is not None
    assert engine._positions["BTCUSDT"] == pytest.approx(5_000)


def test_check_and_reserve_blocked_returns_no_token(engine):
    ok, msg, token = engine.check_and_reserve("BTCUSDT", 15_000, price=100.0)
    assert not ok
    assert token is None
    assert engine._positions.get("BTCUSDT", 0.0) == 0.0   # nothing reserved


def test_release_rolls_back_reservation(engine):
    engine.update_position("BTCUSDT", 3_000, price=100.0)
    ok, _, token = engine.check_and_reserve("BTCUSDT", 4_000, price=110.0)
    assert ok
    assert engine._positions["BTCUSDT"] == pytest.approx(7_000)
    engine.release(token)
    # Position and entry basis restored exactly to the pre-reservation state.
    assert engine._positions["BTCUSDT"] == pytest.approx(3_000)


def test_concurrent_reservations_never_exceed_limit():
    """Atomic reserve must hold the position limit under concurrent threads."""
    limits = RiskLimits(max_position_usd=10_000, max_portfolio_pct=1.0)
    e = RiskEngine(limits=limits, nav=1_000_000)

    def reserve():
        ok, _, _ = e.check_and_reserve("BTCUSDT", 2_000, price=100.0)
        return ok

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(lambda _: reserve(), range(20)))

    # Exactly five 2k reservations fit in the 10k limit — never more.
    assert sum(results) == 5
    assert e._positions["BTCUSDT"] == pytest.approx(10_000)


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
