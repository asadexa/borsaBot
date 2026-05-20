"""ActiveTrade — tracks a single open position's R-level state.

Stores the entry price, initial stop loss, and which R milestones have
already been acted on. A new instance is created when an order fills
and destroyed when the position closes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from borsabot.core.events import OrderSide


class RMilestone(str, Enum):
    """R-level checkpoints that trigger stop adjustments."""
    NONE         = "none"
    BREAKEVEN    = "breakeven"      # +1R  → stop → entry
    PARTIAL_HALF = "partial_half"   # +1.5R → close 50 %, stop stays at entry
    TRAIL_1R     = "trail_1r"       # +2R  → stop → entry + 1R


@dataclass
class ActiveTrade:
    """
    Represents a live position being managed by the StopManager.

    Args:
        symbol:       Trading symbol (e.g. "EURUSD")
        side:         BUY or SELL
        entry_price:  Actual fill price
        initial_sl:   Stop loss at time of entry (absolute price)
        volume:       Total position size in lots / contracts
        ticket:       MT5 position ticket (broker ID)
        strategy_id:  Optional tag to identify which strategy opened this
    """

    symbol:       str
    side:         OrderSide
    entry_price:  float
    initial_sl:   float
    volume:       float
    ticket:       int = 0
    strategy_id:  str = ""
    opened_at_ns: int = field(default_factory=time.time_ns)

    # ── Derived / managed state ───────────────────────────────────────────

    current_sl:   float = field(init=False)
    milestone:    RMilestone = field(default=RMilestone.NONE, init=False)
    # volume remaining after any partial close
    remaining_volume: float = field(init=False)

    def __post_init__(self) -> None:
        self.current_sl       = self.initial_sl
        self.remaining_volume = self.volume

    # ── R helpers ────────────────────────────────────────────────────────

    @property
    def risk_points(self) -> float:
        """Distance in price between entry and initial SL (always positive)."""
        return abs(self.entry_price - self.initial_sl)

    def r_value(self, current_price: float) -> float:
        """
        Current floating R multiple.

        Positive = in profit, Negative = in loss.
        Rounded to 8 dp to eliminate IEEE-754 drift at exact thresholds.
        """
        if self.risk_points < 1e-10:
            return 0.0
        if self.side == OrderSide.BUY:
            raw = (current_price - self.entry_price) / self.risk_points
        else:
            raw = (self.entry_price - current_price) / self.risk_points
        return round(raw, 8)

    # ── Milestone helpers ─────────────────────────────────────────────────

    @property
    def breakeven_price(self) -> float:
        """Entry price (the SL target at +1R)."""
        return self.entry_price

    @property
    def trail_1r_sl(self) -> float:
        """SL target at the +2R milestone — entry + 1R in trade direction."""
        if self.side == OrderSide.BUY:
            return self.entry_price + self.risk_points
        else:
            return self.entry_price - self.risk_points

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ActiveTrade {self.symbol} {self.side.value} "
            f"entry={self.entry_price:.5f} sl={self.current_sl:.5f} "
            f"vol={self.remaining_volume:.2f} milestone={self.milestone.value}>"
        )
