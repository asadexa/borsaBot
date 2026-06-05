"""StopManager — R-based stop management engine.

Monitors active trades tick-by-tick and applies three rules automatically:

  ┌─────────────────────────────────────────────────────────────┐
  │  Milestone    │  Trigger  │  Action                         │
  ├───────────────┼───────────┼─────────────────────────────────┤
  │  BREAKEVEN    │  +1R      │  Move SL → entry price          │
  │  PARTIAL_HALF │  +1.5R    │  Close 50 %, SL stays at entry  │
  │  TRAIL_1R     │  +2R      │  Move SL → entry + 1R           │
  └─────────────────────────────────────────────────────────────┘

Each milestone fires exactly once per trade.  The StopManager is
broker-agnostic: it calls two methods on BrokerGateway subclasses:
  • modify_sl(ticket, new_sl)   — adjust the stop loss price
  • partial_close(ticket, vol)  — close a fraction of the position

Both methods are declared on BrokerGateway (optional / async).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from borsabot.execution.trade_state import ActiveTrade, RMilestone

if TYPE_CHECKING:
    from borsabot.brokers.base import BrokerGateway

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

class StopConfig:
    """
    Thresholds (in R multiples) that govern the three milestones.

    Override any to customise behaviour — e.g. move breakeven earlier
    at +0.8R for volatile assets, or skip partial close entirely.
    """

    def __init__(
        self,
        breakeven_r:     float = 1.0,   # R level that triggers breakeven SL
        partial_r:       float = 1.5,   # R level that triggers 50 % partial close
        partial_pct:     float = 0.50,  # fraction to close (0.50 = 50 %)
        trail_r:         float = 2.0,   # R level that triggers trail +1R SL
        min_lot_step:    float = 0.01,  # minimum broker lot increment
    ) -> None:
        self.breakeven_r  = breakeven_r
        self.partial_r    = partial_r
        self.partial_pct  = partial_pct
        self.trail_r      = trail_r
        self.min_lot_step = min_lot_step


# ──────────────────────────────────────────────────────────────────────────────
# Core engine
# ──────────────────────────────────────────────────────────────────────────────

class StopManager:
    """
    R-based trade lifecycle manager.

    Usage::

        cfg     = StopConfig()
        manager = StopManager(broker, cfg)

        # Register a trade immediately after it fills
        manager.register(trade)

        # On every price tick (called by market-data loop)
        await manager.on_price(symbol="EURUSD", bid=1.08540, ask=1.08560)

        # When a position closes externally
        manager.remove(ticket)
    """

    # Max broker call retries per milestone before skipping
    MAX_RETRIES = 3

    def __init__(
        self,
        broker: "BrokerGateway",
        config: StopConfig | None = None,
    ) -> None:
        self.broker = broker
        self.cfg    = config or StopConfig()
        # ticket → ActiveTrade
        self._trades: dict[int, ActiveTrade] = {}
        # ticket → retry count for the current milestone action
        self._retry_counts: dict[int, int] = {}
        # Auto-increment for ticket=0 fallback
        self._next_synthetic_ticket = -1

    # ── Trade registry ────────────────────────────────────────────────────

    def register(self, trade: ActiveTrade) -> None:
        """Add a freshly-filled trade to the watchlist."""
        if trade.risk_points < 1e-10:
            log.warning(
                "StopManager.register: trade %s has zero risk_points "
                "(entry=SL) — R management disabled for this trade",
                trade.ticket,
            )
            return
        # Prevent ticket=0 collisions (e.g. Binance UUID order IDs)
        if trade.ticket == 0:
            trade.ticket = self._next_synthetic_ticket
            self._next_synthetic_ticket -= 1
            log.info(
                "StopManager: assigned synthetic ticket=%d for %s",
                trade.ticket, trade.symbol,
            )
        self._trades[trade.ticket] = trade
        self._retry_counts[trade.ticket] = 0
        log.info(
            "StopManager: registered ticket=%d %s %s "
            "entry=%.5f  sl=%.5f  risk=%.5fpip  vol=%.2f",
            trade.ticket, trade.symbol, trade.side.value,
            trade.entry_price, trade.initial_sl,
            trade.risk_points, trade.volume,
        )

    def remove(self, ticket: int) -> None:
        """Remove a trade (called when the position closes)."""
        self._trades.pop(ticket, None)
        self._retry_counts.pop(ticket, None)
        log.info("StopManager: removed ticket=%d", ticket)

    @property
    def active_count(self) -> int:
        return len(self._trades)

    # ── Tick handler ──────────────────────────────────────────────────────

    async def on_price(self, symbol: str, bid: float, ask: float) -> None:
        """
        Evaluate every active trade for this symbol on each tick.

        Args:
            symbol: e.g. "EURUSD"
            bid:    current bid price
            ask:    current ask price
        """
        tasks = [
            self._evaluate(trade, bid, ask)
            for trade in list(self._trades.values())
            if trade.symbol == symbol
        ]
        if tasks:
            await asyncio.gather(*tasks)

    # ── Per-trade evaluation ──────────────────────────────────────────────

    async def _evaluate(self, trade: ActiveTrade, bid: float, ask: float) -> None:
        """Run the milestone cascade for a single trade."""
        # Use the price that matters for profit calculation:
        #   BUY profits at bid (what we can sell at right now)
        #   SELL profits at ask (what we buy back at)
        from borsabot.core.events import OrderSide
        current_price = bid if trade.side == OrderSide.BUY else ask

        r = trade.r_value(current_price)

        # ── Milestone BREAKEVEN (+1R) ──────────────────────────────────
        if (
            trade.milestone == RMilestone.NONE
            and r >= self.cfg.breakeven_r
        ):
            await self._do_breakeven(trade)

        # ── Milestone PARTIAL_HALF (+1.5R) ────────────────────────────
        elif (
            trade.milestone == RMilestone.BREAKEVEN
            and r >= self.cfg.partial_r
        ):
            await self._do_partial_close(trade)

        # ── Milestone TRAIL_1R (+2R) ──────────────────────────────────
        elif (
            trade.milestone == RMilestone.PARTIAL_HALF
            and r >= self.cfg.trail_r
        ):
            await self._do_trail_1r(trade)

    # ── Retry management ──────────────────────────────────────────────

    def _check_retry(self, trade: ActiveTrade, action_name: str) -> bool:
        """Increment retry count; return True if under MAX_RETRIES, else skip milestone."""
        count = self._retry_counts.get(trade.ticket, 0) + 1
        self._retry_counts[trade.ticket] = count
        if count >= self.MAX_RETRIES:
            log.warning(
                "StopManager [%s] ticket=%d: %d retries exhausted — skipping milestone",
                action_name, trade.ticket, count,
            )
            return False  # caller should force-advance milestone
        return True  # still has retries left

    # ── Actions ───────────────────────────────────────────────────────────

    async def _do_breakeven(self, trade: ActiveTrade) -> None:
        """Move SL to entry price — risk eliminated."""
        new_sl = trade.breakeven_price
        log.info(
            "StopManager [BREAKEVEN] ticket=%d %s  SL: %.5f → %.5f (entry)",
            trade.ticket, trade.symbol, trade.current_sl, new_sl,
        )
        ok = await self._modify_sl(trade, new_sl)
        if ok:
            trade.current_sl = new_sl
            trade.milestone  = RMilestone.BREAKEVEN
            self._retry_counts[trade.ticket] = 0  # reset for next milestone
        elif not self._check_retry(trade, "BREAKEVEN"):
            # Retries exhausted — force-advance to prevent infinite loop
            trade.current_sl = new_sl
            trade.milestone  = RMilestone.BREAKEVEN
            self._retry_counts[trade.ticket] = 0

    async def _do_partial_close(self, trade: ActiveTrade) -> None:
        """Close 50 % of remaining volume, SL stays at entry."""
        close_vol = self._round_lots(trade.remaining_volume * self.cfg.partial_pct)
        if close_vol <= 0:
            log.warning(
                "StopManager [PARTIAL] ticket=%d: close_vol=0, skipping (min_lot_step=%.2f)",
                trade.ticket, self.cfg.min_lot_step,
            )
            trade.milestone = RMilestone.PARTIAL_HALF  # advance anyway
            self._retry_counts[trade.ticket] = 0
            return

        log.info(
            "StopManager [PARTIAL CLOSE] ticket=%d %s  closing %.2f lots (50 %% of %.2f)",
            trade.ticket, trade.symbol, close_vol, trade.remaining_volume,
        )
        ok = await self._partial_close_position(trade, close_vol)
        if ok:
            trade.remaining_volume -= close_vol
            trade.milestone         = RMilestone.PARTIAL_HALF
            self._retry_counts[trade.ticket] = 0  # reset for next milestone
            log.info(
                "StopManager [PARTIAL CLOSE] ok — remaining vol=%.2f  sl stays at %.5f",
                trade.remaining_volume, trade.current_sl,
            )
        elif not self._check_retry(trade, "PARTIAL_CLOSE"):
            # Retries exhausted — advance milestone to prevent infinite loop
            trade.milestone = RMilestone.PARTIAL_HALF
            self._retry_counts[trade.ticket] = 0

    async def _do_trail_1r(self, trade: ActiveTrade) -> None:
        """Move SL to entry + 1R — locking in 1R minimum profit."""
        new_sl = trade.trail_1r_sl
        log.info(
            "StopManager [TRAIL +1R] ticket=%d %s  SL: %.5f → %.5f",
            trade.ticket, trade.symbol, trade.current_sl, new_sl,
        )
        ok = await self._modify_sl(trade, new_sl)
        if ok:
            trade.current_sl = new_sl
            trade.milestone  = RMilestone.TRAIL_1R
            self._retry_counts[trade.ticket] = 0
        elif not self._check_retry(trade, "TRAIL_1R"):
            trade.current_sl = new_sl
            trade.milestone  = RMilestone.TRAIL_1R
            self._retry_counts[trade.ticket] = 0

    # ── Broker call helpers ───────────────────────────────────────────────

    async def _modify_sl(self, trade: ActiveTrade, new_sl: float) -> bool:
        """Try broker.modify_sl — returns True on success."""
        try:
            if hasattr(self.broker, "modify_sl"):
                return await self.broker.modify_sl(trade.ticket, new_sl)
            else:
                log.warning(
                    "StopManager: broker %r does not implement modify_sl — "
                    "SL update skipped",
                    self.broker.name,
                )
                return False
        except Exception as exc:
            log.error(
                "StopManager._modify_sl failed ticket=%d: %s",
                trade.ticket, exc,
            )
            return False

    async def _partial_close_position(self, trade: ActiveTrade, volume: float) -> bool:
        """Try broker.partial_close — returns True on success."""
        try:
            if hasattr(self.broker, "partial_close"):
                return await self.broker.partial_close(trade.ticket, volume)
            else:
                log.warning(
                    "StopManager: broker %r does not implement partial_close — "
                    "partial close skipped",
                    self.broker.name,
                )
                return False
        except Exception as exc:
            log.error(
                "StopManager._partial_close_position failed ticket=%d: %s",
                trade.ticket, exc,
            )
            return False

    # ── Utilities ─────────────────────────────────────────────────────────

    def _round_lots(self, volume: float) -> float:
        """Round down to nearest lot step (float-safe)."""
        import math
        step = self.cfg.min_lot_step
        # Add tiny epsilon before floor to avoid 1.0/0.01 == 99.9999... issue
        return round(math.floor(volume / step + 1e-9) * step, 10)

    # ── Position sync (BUG #1 fix) ────────────────────────────────────────

    async def sync_positions(self) -> int:
        """Remove trades whose positions no longer exist at the broker.

        Call periodically (e.g. every 30s) to clean up orphaned trades
        that closed externally (TP hit, manual close, SL hit).

        Returns:
            Number of orphaned trades removed.
        """
        if not self._trades:
            return 0

        try:
            live_positions = await self.broker.get_positions()
        except Exception as exc:
            log.warning("StopManager.sync_positions: broker error: %s", exc)
            return 0

        # Collect tickets that are in StopManager but NOT in broker positions
        # For MT5, get_positions returns {symbol: volume}.  We need to cross-
        # reference by symbol because tickets aren't in the dict.
        # Use a conservative approach: if the symbol has zero position, remove
        # ALL trades for that symbol.
        live_symbols = {sym for sym, qty in live_positions.items() if abs(qty) > 1e-10}

        orphaned = [
            ticket
            for ticket, trade in list(self._trades.items())
            if trade.symbol not in live_symbols
        ]

        for ticket in orphaned:
            trade = self._trades.get(ticket)
            if trade:
                log.info(
                    "StopManager [SYNC] removing orphaned trade: "
                    "ticket=%d %s %s (position no longer open)",
                    ticket, trade.symbol, trade.side.value,
                )
            self.remove(ticket)

        return len(orphaned)

    # ── Debug helpers ─────────────────────────────────────────────────────

    def status(self) -> list[dict]:
        """Return a snapshot of all managed trades for the dashboard."""
        return [
            {
                "ticket":    t.ticket,
                "symbol":    t.symbol,
                "side":      t.side.value,
                "entry":     t.entry_price,
                "sl":        t.current_sl,
                "milestone": t.milestone.value,
                "volume":    t.remaining_volume,
            }
            for t in self._trades.values()
        ]
