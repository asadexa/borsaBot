"""Unit tests for the R-based StopManager and ActiveTrade state.

Tests cover:
  - R value calculation for BUY and SELL
  - Breakeven trigger (+1R)
  - Partial close trigger (+1.5R)
  - Trail trigger (+2R)
  - Milestone ordering (milestones never fire out of order)
  - Zero-risk guard
  - Broker failures (modify_sl returns False)
"""

from __future__ import annotations

import asyncio
import pytest

from borsabot.core.events import OrderSide
from borsabot.execution.trade_state import ActiveTrade, RMilestone
from borsabot.execution.stop_manager import StopManager, StopConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class MockBroker:
    """Minimal broker stub that records calls and returns configurable results."""

    name = "mock"

    def __init__(self, modify_ok: bool = True, partial_ok: bool = True) -> None:
        self.modify_ok  = modify_ok
        self.partial_ok = partial_ok
        self.modify_calls:  list[tuple[int, float]] = []
        self.partial_calls: list[tuple[int, float]] = []

    async def modify_sl(self, ticket: int, new_sl: float) -> bool:
        self.modify_calls.append((ticket, new_sl))
        return self.modify_ok

    async def partial_close(self, ticket: int, volume: float) -> bool:
        self.partial_calls.append((ticket, volume))
        return self.partial_ok


def make_buy_trade(
    entry: float = 1.10000,
    sl: float    = 1.09000,  # 100 pips risk → 1 R = 100 pips
    volume: float = 1.0,
    ticket: int   = 1001,
) -> ActiveTrade:
    return ActiveTrade(
        symbol="EURUSD",
        side=OrderSide.BUY,
        entry_price=entry,
        initial_sl=sl,
        volume=volume,
        ticket=ticket,
    )


def make_sell_trade(
    entry: float = 1.10000,
    sl: float    = 1.11000,  # 100 pips risk
    volume: float = 1.0,
    ticket: int   = 1002,
) -> ActiveTrade:
    return ActiveTrade(
        symbol="EURUSD",
        side=OrderSide.SELL,
        entry_price=entry,
        initial_sl=sl,
        volume=volume,
        ticket=ticket,
    )


def run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
# ActiveTrade: R value
# ─────────────────────────────────────────────────────────────────────────────

class TestActiveTrade:

    def test_buy_r_at_entry(self):
        t = make_buy_trade(entry=1.10000, sl=1.09000)
        assert t.r_value(1.10000) == pytest.approx(0.0)

    def test_buy_r_plus_1(self):
        t = make_buy_trade(entry=1.10000, sl=1.09000)
        # +1R = entry + risk = 1.10000 + 0.01000 = 1.11000
        assert t.r_value(1.11000) == pytest.approx(1.0)

    def test_buy_r_minus(self):
        t = make_buy_trade(entry=1.10000, sl=1.09000)
        # At SL: -1R
        assert t.r_value(1.09000) == pytest.approx(-1.0)

    def test_sell_r_plus_1(self):
        t = make_sell_trade(entry=1.10000, sl=1.11000)
        # SELL profits when price falls: +1R = entry - risk = 1.09000
        assert t.r_value(1.09000) == pytest.approx(1.0)

    def test_sell_r_minus(self):
        t = make_sell_trade(entry=1.10000, sl=1.11000)
        assert t.r_value(1.11000) == pytest.approx(-1.0)

    def test_breakeven_price_buy(self):
        t = make_buy_trade(entry=1.10000, sl=1.09000)
        assert t.breakeven_price == pytest.approx(1.10000)

    def test_trail_1r_sl_buy(self):
        t = make_buy_trade(entry=1.10000, sl=1.09000)
        assert t.trail_1r_sl == pytest.approx(1.11000)

    def test_trail_1r_sl_sell(self):
        t = make_sell_trade(entry=1.10000, sl=1.11000)
        assert t.trail_1r_sl == pytest.approx(1.09000)

    def test_risk_points(self):
        t = make_buy_trade(entry=1.10000, sl=1.09000)
        assert t.risk_points == pytest.approx(0.01000)


# ─────────────────────────────────────────────────────────────────────────────
# StopManager: BUY milestones
# ─────────────────────────────────────────────────────────────────────────────

class TestStopManagerBuy:

    def setup_method(self):
        self.broker = MockBroker()
        self.cfg    = StopConfig(breakeven_r=1.0, partial_r=1.5, partial_pct=0.5, trail_r=2.0)
        self.mgr    = StopManager(self.broker, self.cfg)
        self.trade  = make_buy_trade(entry=1.10000, sl=1.09000, volume=2.0, ticket=1001)
        self.mgr.register(self.trade)

    def _tick(self, bid: float, ask: float | None = None):
        run(self.mgr.on_price("EURUSD", bid=bid, ask=ask or bid + 0.00010))

    def test_no_action_below_1r(self):
        self._tick(bid=1.10950)  # +0.95R
        assert self.trade.milestone == RMilestone.NONE
        assert not self.broker.modify_calls

    def test_breakeven_at_1r(self):
        self._tick(bid=1.11000)  # exactly +1R
        assert self.trade.milestone == RMilestone.BREAKEVEN
        assert len(self.broker.modify_calls) == 1
        ticket, new_sl = self.broker.modify_calls[0]
        assert ticket  == 1001
        assert new_sl  == pytest.approx(1.10000)
        assert self.trade.current_sl == pytest.approx(1.10000)

    def test_partial_close_at_1_5r(self):
        self._tick(bid=1.11000)   # tick 1: breakeven fires  (NONE → BREAKEVEN)
        self._tick(bid=1.11500)   # tick 2: at +1.5R but milestone=BREAKEVEN → PARTIAL fires
        assert self.trade.milestone == RMilestone.PARTIAL_HALF
        assert len(self.broker.partial_calls) == 1
        _, vol = self.broker.partial_calls[0]
        assert vol == pytest.approx(1.0)         # 50 % of 2.0
        assert self.trade.remaining_volume == pytest.approx(1.0)

    def test_trail_at_2r(self):
        self._tick(bid=1.11000)   # tick 1: breakeven fires  (NONE → BREAKEVEN)
        self._tick(bid=1.11500)   # tick 2: partial fires     (BREAKEVEN → PARTIAL_HALF)
        # tick 3: price already at +1.5R; need a tick at +2R to fire trail
        self._tick(bid=1.12000)   # +2R — but milestone is PARTIAL_HALF so trail fires here
        assert self.trade.milestone == RMilestone.TRAIL_1R
        # Last modify call: new SL = entry + 1R = 1.11000
        ticket, new_sl = self.broker.modify_calls[-1]
        assert new_sl == pytest.approx(1.11000)

    def test_milestones_fire_once_each(self):
        """Repeated ticks at the same R level must not re-fire milestones."""
        for _ in range(5):
            self._tick(bid=1.11000)  # +1R
        assert len(self.broker.modify_calls) == 1  # only once

    def test_milestones_not_skipped_forward(self):
        """Jump straight to +2R — all milestones must still fire in order."""
        self._tick(bid=1.12000)  # +2R in one jump
        # Breakeven fires first
        assert self.trade.milestone == RMilestone.BREAKEVEN
        # Partial and trail each require a separate tick to cascade
        self._tick(bid=1.12000)
        assert self.trade.milestone == RMilestone.PARTIAL_HALF
        self._tick(bid=1.12000)
        assert self.trade.milestone == RMilestone.TRAIL_1R


# ─────────────────────────────────────────────────────────────────────────────
# StopManager: SELL milestones
# ─────────────────────────────────────────────────────────────────────────────

class TestStopManagerSell:

    def setup_method(self):
        self.broker = MockBroker()
        self.mgr    = StopManager(self.broker, StopConfig())
        self.trade  = make_sell_trade(entry=1.10000, sl=1.11000, volume=1.0, ticket=2001)
        self.mgr.register(self.trade)

    def _tick(self, bid: float, ask: float | None = None):
        run(self.mgr.on_price("EURUSD", bid=bid, ask=ask or bid + 0.00010))

    def test_sell_breakeven(self):
        # SELL +1R: ask must be <= entry - risk = 1.09000
        # spread 0.00010 → ask=1.08990 means r = (1.10000 - 1.08990)/0.01 = 1.01 ✓
        self._tick(bid=1.08980, ask=1.08990)
        assert self.trade.milestone == RMilestone.BREAKEVEN
        _, new_sl = self.broker.modify_calls[0]
        assert new_sl == pytest.approx(1.10000)

    def test_sell_no_action_above_entry(self):
        self._tick(bid=1.10500)  # SELL is losing when price goes up
        assert self.trade.milestone == RMilestone.NONE


# ─────────────────────────────────────────────────────────────────────────────
# Error / edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestStopManagerEdgeCases:

    def test_zero_risk_trade_skipped(self, caplog):
        """Trade where entry == SL should be registered but ignored."""
        broker = MockBroker()
        mgr    = StopManager(broker)
        trade  = ActiveTrade(
            symbol="EURUSD", side=OrderSide.BUY,
            entry_price=1.10000, initial_sl=1.10000,
            volume=1.0, ticket=9999,
        )
        mgr.register(trade)
        assert mgr.active_count == 0  # rejected because risk_points ≈ 0

    def test_broker_modify_failure_no_milestone(self):
        """If broker.modify_sl returns False, milestone must NOT advance."""
        broker = MockBroker(modify_ok=False)
        mgr    = StopManager(broker)
        trade  = make_buy_trade(ticket=3001)
        mgr.register(trade)
        run(mgr.on_price("EURUSD", bid=1.11000, ask=1.11010))
        assert trade.milestone == RMilestone.NONE  # failed, so no advance

    def test_unknown_symbol_ignored(self):
        """Tick for a different symbol must not affect trades."""
        broker = MockBroker()
        mgr    = StopManager(broker)
        trade  = make_buy_trade(ticket=4001)
        mgr.register(trade)
        run(mgr.on_price("USDJPY", bid=155.00, ask=155.01))
        assert trade.milestone == RMilestone.NONE

    def test_remove_trade(self):
        broker = MockBroker()
        mgr    = StopManager(broker)
        trade  = make_buy_trade(ticket=5001)
        mgr.register(trade)
        assert mgr.active_count == 1
        mgr.remove(5001)
        assert mgr.active_count == 0

    def test_status_snapshot(self):
        broker = MockBroker()
        mgr    = StopManager(broker)
        trade  = make_buy_trade(ticket=6001)
        mgr.register(trade)
        snap = mgr.status()
        assert len(snap) == 1
        assert snap[0]["ticket"]    == 6001
        assert snap[0]["milestone"] == "none"
