"""Unit tests for the R-based StopManager and ActiveTrade state.

Tests cover:
  - R value calculation for BUY and SELL
  - Breakeven trigger (+1R)
  - Partial close trigger (+1.5R)
  - Trail trigger (+2R)
  - Milestone ordering (milestones never fire out of order)
  - Zero-risk guard
  - Broker failures (modify_sl returns False)
  - Retry exhaustion (3 failed attempts → skip milestone)
  - Broker exceptions (raise instead of returning False)
  - Broker without modify_sl/partial_close (hasattr check)
  - SELL full cascade (breakeven → partial → trail)
  - Multiple trades on the same symbol
  - _round_lots edge cases (float precision, tiny volumes)
  - Position sync (orphaned trade cleanup)
  - Ticket auto-increment for ticket=0 collision
  - Extreme R jumps (+5R single tick)
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
        self._positions: dict[str, float] = {}

    async def modify_sl(self, ticket: int, new_sl: float) -> bool:
        self.modify_calls.append((ticket, new_sl))
        return self.modify_ok

    async def partial_close(self, ticket: int, volume: float) -> bool:
        self.partial_calls.append((ticket, volume))
        return self.partial_ok

    async def get_positions(self) -> dict[str, float]:
        return dict(self._positions)


class ExplodingBroker:
    """Broker stub that raises exceptions on every call."""

    name = "exploding"

    async def modify_sl(self, ticket: int, new_sl: float) -> bool:
        raise ConnectionError("MT5 connection lost")

    async def partial_close(self, ticket: int, volume: float) -> bool:
        raise ConnectionError("MT5 connection lost")

    async def get_positions(self) -> dict[str, float]:
        raise ConnectionError("MT5 connection lost")


class NakedBroker:
    """Broker stub with NO modify_sl or partial_close methods at all."""

    name = "naked"

    async def get_positions(self) -> dict[str, float]:
        return {}


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

    # ── XAUUSD (Gold) price scale tests ──────────────────────────────

    def test_xauusd_buy_r_value(self):
        """Gold at $2350 entry, $2325 SL → $25 risk = 1R."""
        t = ActiveTrade(
            symbol="XAUUSD", side=OrderSide.BUY,
            entry_price=2350.00, initial_sl=2325.00,
            volume=0.10, ticket=100,
        )
        assert t.risk_points == pytest.approx(25.0)
        assert t.r_value(2375.00) == pytest.approx(1.0)   # +1R
        assert t.r_value(2387.50) == pytest.approx(1.5)   # +1.5R
        assert t.r_value(2400.00) == pytest.approx(2.0)   # +2R
        assert t.r_value(2325.00) == pytest.approx(-1.0)  # -1R (SL hit)

    def test_xauusd_sell_r_value(self):
        """Gold SELL at $2350 entry, $2375 SL → $25 risk."""
        t = ActiveTrade(
            symbol="XAUUSD", side=OrderSide.SELL,
            entry_price=2350.00, initial_sl=2375.00,
            volume=0.10, ticket=101,
        )
        assert t.r_value(2325.00) == pytest.approx(1.0)   # +1R (price fell)
        assert t.r_value(2300.00) == pytest.approx(2.0)   # +2R
        assert t.r_value(2375.00) == pytest.approx(-1.0)  # -1R (SL hit)

    def test_xauusd_trail_sl_levels(self):
        """Gold trail SL should be entry ± risk_points."""
        buy = ActiveTrade(
            symbol="XAUUSD", side=OrderSide.BUY,
            entry_price=2350.00, initial_sl=2325.00,
            volume=0.10, ticket=102,
        )
        assert buy.breakeven_price == pytest.approx(2350.00)
        assert buy.trail_1r_sl == pytest.approx(2375.00)  # entry + 25

        sell = ActiveTrade(
            symbol="XAUUSD", side=OrderSide.SELL,
            entry_price=2350.00, initial_sl=2375.00,
            volume=0.10, ticket=103,
        )
        assert sell.trail_1r_sl == pytest.approx(2325.00)  # entry - 25


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

    def test_sell_full_cascade(self):
        """SELL direction: complete breakeven → partial → trail cycle."""
        # Risk = 0.01 (entry 1.10 - SL 1.11 = 0.01)
        # +1R  at ask = 1.09000
        # +1.5R at ask = 1.08500
        # +2R  at ask = 1.08000
        self._tick(bid=1.08980, ask=1.09000)  # +1R → breakeven
        assert self.trade.milestone == RMilestone.BREAKEVEN

        self._tick(bid=1.08490, ask=1.08500)  # +1.5R → partial
        assert self.trade.milestone == RMilestone.PARTIAL_HALF
        assert len(self.broker.partial_calls) == 1
        assert self.trade.remaining_volume == pytest.approx(0.50)

        self._tick(bid=1.07990, ask=1.08000)  # +2R → trail
        assert self.trade.milestone == RMilestone.TRAIL_1R
        _, trail_sl = self.broker.modify_calls[-1]
        # SELL trail SL = entry - 1R = 1.10 - 0.01 = 1.09
        assert trail_sl == pytest.approx(1.09000)

# ─────────────────────────────────────────────────────────────────────────────
# StopManager: XAUUSD (Gold) full cascade
# ─────────────────────────────────────────────────────────────────────────────

class TestStopManagerGold:
    """Full milestone cascade at XAUUSD price scale ($2350, $25 risk)."""

    def setup_method(self):
        self.broker = MockBroker()
        self.cfg    = StopConfig(breakeven_r=1.0, partial_r=1.5, partial_pct=0.5,
                                 trail_r=2.0, min_lot_step=0.01)
        self.mgr    = StopManager(self.broker, self.cfg)
        # Gold BUY: entry $2350, SL $2325 → risk=$25, 0.10 lot
        self.trade  = ActiveTrade(
            symbol="XAUUSD", side=OrderSide.BUY,
            entry_price=2350.00, initial_sl=2325.00,
            volume=0.10, ticket=20001,
        )
        self.mgr.register(self.trade)

    def _tick(self, bid: float, ask: float | None = None):
        # Gold spread ~$0.30
        run(self.mgr.on_price("XAUUSD", bid=bid, ask=ask or bid + 0.30))

    def test_gold_no_action_below_1r(self):
        self._tick(bid=2374.00)  # +24/25 = +0.96R
        assert self.trade.milestone == RMilestone.NONE

    def test_gold_breakeven_at_1r(self):
        self._tick(bid=2375.00)  # +$25 = +1R
        assert self.trade.milestone == RMilestone.BREAKEVEN
        _, new_sl = self.broker.modify_calls[0]
        assert new_sl == pytest.approx(2350.00)

    def test_gold_full_cascade(self):
        """Gold: breakeven($2375) → partial($2387.50) → trail($2400)."""
        # +1R → breakeven
        self._tick(bid=2375.00)
        assert self.trade.milestone == RMilestone.BREAKEVEN
        assert self.trade.current_sl == pytest.approx(2350.00)

        # +1.5R → partial close (0.05 of 0.10 lot)
        self._tick(bid=2387.50)
        assert self.trade.milestone == RMilestone.PARTIAL_HALF
        assert len(self.broker.partial_calls) == 1
        _, vol = self.broker.partial_calls[0]
        assert vol == pytest.approx(0.05)
        assert self.trade.remaining_volume == pytest.approx(0.05)

        # +2R → trail SL to entry + $25 = $2375
        self._tick(bid=2400.00)
        assert self.trade.milestone == RMilestone.TRAIL_1R
        _, trail_sl = self.broker.modify_calls[-1]
        assert trail_sl == pytest.approx(2375.00)

    def test_gold_sell_cascade(self):
        """Gold SELL: breakeven→partial→trail with falling prices."""
        sell = ActiveTrade(
            symbol="XAUUSD", side=OrderSide.SELL,
            entry_price=2350.00, initial_sl=2375.00,
            volume=0.10, ticket=20002,
        )
        mgr = StopManager(self.broker, self.cfg)
        mgr.register(sell)

        # +1R: ask ≤ $2325
        run(mgr.on_price("XAUUSD", bid=2324.50, ask=2325.00))
        assert sell.milestone == RMilestone.BREAKEVEN

        # +1.5R: ask ≤ $2312.50
        run(mgr.on_price("XAUUSD", bid=2312.00, ask=2312.50))
        assert sell.milestone == RMilestone.PARTIAL_HALF

        # +2R: ask ≤ $2300
        run(mgr.on_price("XAUUSD", bid=2299.50, ask=2300.00))
        assert sell.milestone == RMilestone.TRAIL_1R
        # Trail SL = entry - $25 = $2325
        _, trail_sl = self.broker.modify_calls[-1]
        assert trail_sl == pytest.approx(2325.00)


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
        """If broker.modify_sl returns False, milestone must NOT advance (first attempt)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Retry exhaustion tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRetryExhaustion:

    def test_breakeven_skipped_after_max_retries(self):
        """After MAX_RETRIES failed modify_sl, breakeven should force-advance."""
        broker = MockBroker(modify_ok=False)
        mgr    = StopManager(broker)
        trade  = make_buy_trade(ticket=7001)
        mgr.register(trade)

        # Send MAX_RETRIES ticks at +1R
        for _ in range(StopManager.MAX_RETRIES):
            run(mgr.on_price("EURUSD", bid=1.11000, ask=1.11010))

        # After 3 failed attempts, milestone should force-advance
        assert trade.milestone == RMilestone.BREAKEVEN
        assert len(broker.modify_calls) == StopManager.MAX_RETRIES

    def test_partial_close_skipped_after_max_retries(self):
        """After MAX_RETRIES failed partial_close, partial should force-advance."""
        broker = MockBroker(modify_ok=True, partial_ok=False)
        mgr    = StopManager(broker)
        trade  = make_buy_trade(volume=2.0, ticket=7002)
        mgr.register(trade)

        # First: breakeven (succeeds)
        run(mgr.on_price("EURUSD", bid=1.11000, ask=1.11010))
        assert trade.milestone == RMilestone.BREAKEVEN

        # Then: 3 failed partial close attempts at +1.5R
        for _ in range(StopManager.MAX_RETRIES):
            run(mgr.on_price("EURUSD", bid=1.11500, ask=1.11510))

        # Should force-advance past PARTIAL_HALF
        assert trade.milestone == RMilestone.PARTIAL_HALF
        assert len(broker.partial_calls) == StopManager.MAX_RETRIES
        # Volume should NOT have changed since partial_close failed
        assert trade.remaining_volume == pytest.approx(2.0)

    def test_trail_skipped_after_max_retries(self):
        """After MAX_RETRIES failed trail modify_sl, trail should force-advance."""
        broker = MockBroker(modify_ok=True, partial_ok=True)
        mgr    = StopManager(broker)
        trade  = make_buy_trade(volume=2.0, ticket=7003)
        mgr.register(trade)

        # Breakeven + partial (both succeed)
        run(mgr.on_price("EURUSD", bid=1.11000, ask=1.11010))
        run(mgr.on_price("EURUSD", bid=1.11500, ask=1.11510))
        assert trade.milestone == RMilestone.PARTIAL_HALF

        # Now make modify_sl fail for trail
        broker.modify_ok = False
        for _ in range(StopManager.MAX_RETRIES):
            run(mgr.on_price("EURUSD", bid=1.12000, ask=1.12010))

        assert trade.milestone == RMilestone.TRAIL_1R

    def test_retry_counter_resets_after_success(self):
        """Retry counter should reset to 0 when a milestone succeeds."""
        broker = MockBroker(modify_ok=True)
        mgr    = StopManager(broker)
        trade  = make_buy_trade(ticket=7004)
        mgr.register(trade)

        run(mgr.on_price("EURUSD", bid=1.11000, ask=1.11010))
        assert trade.milestone == RMilestone.BREAKEVEN
        assert mgr._retry_counts[7004] == 0  # reset after success


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Broker exception handling
# ─────────────────────────────────────────────────────────────────────────────

class TestBrokerExceptions:

    def test_modify_sl_exception_no_crash(self):
        """Broker raising exception should not crash StopManager."""
        broker = ExplodingBroker()
        mgr    = StopManager(broker)
        trade  = make_buy_trade(ticket=8001)
        mgr.register(trade)

        # Should not raise
        run(mgr.on_price("EURUSD", bid=1.11000, ask=1.11010))
        assert trade.milestone == RMilestone.NONE  # exception = failed

    def test_partial_close_exception_no_crash(self):
        """Broker partial_close exception should not crash."""
        broker = ExplodingBroker()
        mgr    = StopManager(broker)
        trade  = make_buy_trade(volume=2.0, ticket=8002)
        mgr.register(trade)

        # Force breakeven via mock first, then switch to exploding
        # Use a two-phase approach
        mock_broker = MockBroker()
        mgr2 = StopManager(mock_broker)
        trade2 = make_buy_trade(volume=2.0, ticket=8003)
        mgr2.register(trade2)

        run(mgr2.on_price("EURUSD", bid=1.11000, ask=1.11010))
        assert trade2.milestone == RMilestone.BREAKEVEN

        # Now replace broker with exploding one
        mgr2.broker = ExplodingBroker()
        run(mgr2.on_price("EURUSD", bid=1.11500, ask=1.11510))
        assert trade2.milestone == RMilestone.BREAKEVEN  # failed, stays


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Broker without modify_sl/partial_close
# ─────────────────────────────────────────────────────────────────────────────

class TestNakedBroker:

    def test_no_modify_sl_graceful_skip(self):
        """Broker without modify_sl should gracefully return False."""
        broker = NakedBroker()
        mgr    = StopManager(broker)
        trade  = make_buy_trade(ticket=9001)
        mgr.register(trade)

        run(mgr.on_price("EURUSD", bid=1.11000, ask=1.11010))
        assert trade.milestone == RMilestone.NONE  # skipped

    def test_no_partial_close_graceful_skip(self):
        """Broker without partial_close should gracefully skip."""
        # Use a mixed broker: has modify_sl but not partial_close
        class HalfBroker:
            name = "half"
            async def modify_sl(self, ticket, new_sl):
                return True
        broker = HalfBroker()
        mgr    = StopManager(broker)
        trade  = make_buy_trade(volume=2.0, ticket=9002)
        mgr.register(trade)

        run(mgr.on_price("EURUSD", bid=1.11000, ask=1.11010))
        assert trade.milestone == RMilestone.BREAKEVEN  # modify_sl works

        run(mgr.on_price("EURUSD", bid=1.11500, ask=1.11510))
        # partial_close missing → returns False → stays at BREAKEVEN
        assert trade.milestone == RMilestone.BREAKEVEN


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Multiple trades same symbol
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleTrades:

    def test_two_trades_same_symbol_independent(self):
        """Two trades on EURUSD should be evaluated independently."""
        broker = MockBroker()
        mgr    = StopManager(broker)

        trade1 = make_buy_trade(entry=1.10000, sl=1.09000, ticket=10001)
        trade2 = make_buy_trade(entry=1.10500, sl=1.09500, ticket=10002)
        mgr.register(trade1)
        mgr.register(trade2)
        assert mgr.active_count == 2

        # At bid=1.11000: trade1 is at +1R, trade2 is at +0.5R
        run(mgr.on_price("EURUSD", bid=1.11000, ask=1.11010))
        assert trade1.milestone == RMilestone.BREAKEVEN
        assert trade2.milestone == RMilestone.NONE  # not yet at +1R

    def test_different_symbols_dont_interfere(self):
        """Trades on different symbols should not be evaluated by wrong ticks."""
        broker = MockBroker()
        mgr    = StopManager(broker)

        eurusd = make_buy_trade(entry=1.10000, sl=1.09000, ticket=10003)
        mgr.register(eurusd)

        gbpusd = ActiveTrade(
            symbol="GBPUSD", side=OrderSide.BUY,
            entry_price=1.30000, initial_sl=1.29000,
            volume=1.0, ticket=10004,
        )
        mgr.register(gbpusd)

        # EURUSD tick at +1R, GBPUSD should not be affected
        run(mgr.on_price("EURUSD", bid=1.11000, ask=1.11010))
        assert eurusd.milestone == RMilestone.BREAKEVEN
        assert gbpusd.milestone == RMilestone.NONE


# ─────────────────────────────────────────────────────────────────────────────
# NEW: _round_lots edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestRoundLots:

    def setup_method(self):
        self.mgr = StopManager(MockBroker())

    def test_exact_multiple(self):
        # 1.0 / 0.01 = 100 → exactly 1.0
        assert self.mgr._round_lots(1.0) == pytest.approx(1.0)

    def test_rounds_down(self):
        # 0.035 / 0.01 = 3.5 → floor → 3 * 0.01 = 0.03
        assert self.mgr._round_lots(0.035) == pytest.approx(0.03)

    def test_very_small_below_step(self):
        # 0.005 < min_lot_step (0.01) → rounds to 0
        assert self.mgr._round_lots(0.005) == pytest.approx(0.0)

    def test_ieee_754_precision(self):
        # 0.1 + 0.2 ≠ 0.3 in float but should round correctly
        assert self.mgr._round_lots(0.1 + 0.2) == pytest.approx(0.3)

    def test_custom_step(self):
        mgr = StopManager(MockBroker(), StopConfig(min_lot_step=0.1))
        assert mgr._round_lots(0.55) == pytest.approx(0.5)

    def test_zero_volume(self):
        assert self.mgr._round_lots(0.0) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Position sync (orphaned trade cleanup)
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionSync:

    def test_sync_removes_orphaned_trades(self):
        """Trades whose symbol is no longer in broker positions are removed."""
        broker = MockBroker()
        broker._positions = {"EURUSD": 1.0}  # Position exists at start

        mgr   = StopManager(broker)
        trade = make_buy_trade(ticket=11001)
        mgr.register(trade)
        assert mgr.active_count == 1

        # Position still open → sync should not remove
        removed = run(mgr.sync_positions())
        assert removed == 0
        assert mgr.active_count == 1

        # Position closed → sync should remove
        broker._positions = {}
        removed = run(mgr.sync_positions())
        assert removed == 1
        assert mgr.active_count == 0

    def test_sync_keeps_active_positions(self):
        """Trades whose symbol still has a position should be kept."""
        broker = MockBroker()
        broker._positions = {"EURUSD": 1.0, "GBPUSD": 0.5}

        mgr = StopManager(broker)
        t1 = make_buy_trade(ticket=11002)
        t2 = ActiveTrade(
            symbol="GBPUSD", side=OrderSide.BUY,
            entry_price=1.30, initial_sl=1.29,
            volume=0.5, ticket=11003,
        )
        mgr.register(t1)
        mgr.register(t2)

        # Remove only GBPUSD position
        broker._positions = {"EURUSD": 1.0}
        removed = run(mgr.sync_positions())
        assert removed == 1
        assert mgr.active_count == 1
        assert 11002 in mgr._trades
        assert 11003 not in mgr._trades

    def test_sync_broker_error_no_crash(self):
        """If broker.get_positions() raises, sync should return 0."""
        broker = ExplodingBroker()
        mgr    = StopManager(broker)
        trade  = make_buy_trade(ticket=11004)
        mgr.register(trade)

        removed = run(mgr.sync_positions())
        assert removed == 0
        assert mgr.active_count == 1  # trade should remain

    def test_sync_empty_no_call(self):
        """If no trades are managed, sync should not call broker."""
        broker = MockBroker()
        mgr    = StopManager(broker)

        removed = run(mgr.sync_positions())
        assert removed == 0


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Ticket auto-increment
# ─────────────────────────────────────────────────────────────────────────────

class TestTicketAutoIncrement:

    def test_ticket_zero_gets_synthetic(self):
        """Trades with ticket=0 should get unique negative ticket IDs."""
        broker = MockBroker()
        mgr    = StopManager(broker)

        t1 = make_buy_trade(ticket=0)
        t2 = make_buy_trade(ticket=0)
        mgr.register(t1)
        mgr.register(t2)

        assert mgr.active_count == 2
        assert t1.ticket != t2.ticket
        assert t1.ticket < 0
        assert t2.ticket < 0

    def test_normal_ticket_unchanged(self):
        """Trades with a real ticket should keep their ticket."""
        broker = MockBroker()
        mgr    = StopManager(broker)

        trade = make_buy_trade(ticket=5555)
        mgr.register(trade)
        assert trade.ticket == 5555


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Extreme R jumps
# ─────────────────────────────────────────────────────────────────────────────

class TestExtremeJumps:

    def test_5r_jump_cascades_correctly(self):
        """A +5R jump should cascade through all milestones, one per tick."""
        broker = MockBroker()
        mgr    = StopManager(broker)
        trade  = make_buy_trade(volume=2.0, ticket=12001)
        mgr.register(trade)

        # +5R in first tick → only breakeven fires
        run(mgr.on_price("EURUSD", bid=1.15000, ask=1.15010))
        assert trade.milestone == RMilestone.BREAKEVEN

        # second tick → partial
        run(mgr.on_price("EURUSD", bid=1.15000, ask=1.15010))
        assert trade.milestone == RMilestone.PARTIAL_HALF

        # third tick → trail
        run(mgr.on_price("EURUSD", bid=1.15000, ask=1.15010))
        assert trade.milestone == RMilestone.TRAIL_1R

        # fourth tick → nothing more happens
        run(mgr.on_price("EURUSD", bid=1.15000, ask=1.15010))
        assert trade.milestone == RMilestone.TRAIL_1R

    def test_price_retreats_after_milestone(self):
        """Once a milestone fires, price retreating should not undo it."""
        broker = MockBroker()
        mgr    = StopManager(broker)
        trade  = make_buy_trade(ticket=12002)
        mgr.register(trade)

        # +1R → breakeven
        run(mgr.on_price("EURUSD", bid=1.11000, ask=1.11010))
        assert trade.milestone == RMilestone.BREAKEVEN

        # Price retreats to +0.5R
        run(mgr.on_price("EURUSD", bid=1.10500, ask=1.10510))
        assert trade.milestone == RMilestone.BREAKEVEN  # stays
        assert trade.current_sl == pytest.approx(1.10000)  # SL still at entry
