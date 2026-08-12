"""Real-time Risk Engine.

Enforces position limits, NAV exposure caps, volatility-adjusted sizing,
Kelly criterion allocation, and daily drawdown halt guard.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)


class RiskLimits:
    """Configuration object for risk limits."""

    def __init__(
        self,
        max_position_usd: float = 50_000.0,
        max_portfolio_pct: float = 0.10,       # max single-asset % of NAV
        max_drawdown_pct: float = 0.05,        # daily halt at 5% NAV loss
        vol_target_annual: float = 0.15,       # 15% annualized vol target
        kelly_fraction: float = 0.25,          # quarter-Kelly (conservative)
    ) -> None:
        self.max_position_usd  = max_position_usd
        self.max_portfolio_pct = max_portfolio_pct
        self.max_drawdown_pct  = max_drawdown_pct
        self.vol_target_annual = vol_target_annual
        self.kelly_fraction    = kelly_fraction


class RiskEngine:
    """
    Real-time risk management layer.

    Sits between the Signal layer and the Execution Engine.
    All signals must pass through check_new_order() before execution.
    """

    def __init__(self, limits: RiskLimits, nav: float) -> None:
        self.limits    = limits
        self.nav       = nav
        self._positions: dict[str, float] = {}    # symbol → signed USD value
        self._entry_price: dict[str, float] = {}  # symbol → entry price basis
        self._mark_pnl: dict[str, float] = {}     # symbol → last marked unrealized PnL
        self._daily_pnl: float = 0.0
        self._halted:    bool  = False
        # Reentrant lock — guards all shared mutable state so concurrent callers
        # (and the atomic check_and_reserve below) never interleave a check with
        # another caller's position update. RLock because mark_to_market() calls
        # update_pnl() while already holding it.
        self._lock = threading.RLock()

    # ── Position & Exposure Checks ────────────────────────────────────────

    def check_new_order(
        self,
        symbol: str,
        order_usd: float,
    ) -> tuple[bool, str]:
        """
        Validate a new order against all risk limits.

        `order_usd` is signed: + increases a long (or covers a short), - the
        opposite. Limits are checked against the *resulting* net exposure, so an
        order that reduces or closes a position is never blocked by the size
        limit (closing risk should always be allowed).

        Returns (allowed: bool, reason: str)
        """
        with self._lock:
            if self._halted:
                return False, "Trading halted: daily drawdown limit reached"

            current_signed = self._positions.get(symbol, 0.0)
            new_signed     = current_signed + order_usd
            new_pos_usd    = abs(new_signed)

            # Reducing or closing exposure is always allowed.
            if new_pos_usd <= abs(current_signed):
                return True, "OK"

            # ── Single position size limit ─────────────────────────────
            if new_pos_usd > self.limits.max_position_usd:
                return False, (
                    f"Position limit exceeded: "
                    f"${new_pos_usd:,.0f} > ${self.limits.max_position_usd:,.0f}"
                )

            # ── Single asset NAV% limit ────────────────────────────────
            nav_pct = new_pos_usd / (self.nav + 1e-9)
            if nav_pct > self.limits.max_portfolio_pct:
                return False, (
                    f"NAV% limit exceeded: "
                    f"{nav_pct*100:.1f}% > {self.limits.max_portfolio_pct*100:.0f}%"
                )

            return True, "OK"

    def check_and_reserve(
        self,
        symbol: str,
        order_usd: float,
        price: float | None = None,
    ) -> tuple[bool, str, tuple | None]:
        """Atomically check limits AND reserve the exposure under one lock.

        This closes the check-then-act race: the caller awaits execution between
        deciding and confirming a fill, and two concurrent signals could both
        pass an isolated check_new_order() before either updated the position.
        Reserving here means the second signal sees the first's exposure.

        Returns (allowed, reason, token). On success the position is already
        updated; pass `token` to release() to roll the reservation back if the
        order is never actually sent (e.g. execution returns no fills).
        """
        with self._lock:
            ok, msg = self.check_new_order(symbol, order_usd)
            if not ok:
                return ok, msg, None
            token = (
                symbol,
                self._positions.get(symbol, 0.0),
                self._entry_price.get(symbol),
                self._mark_pnl.get(symbol, 0.0),
            )
            self.update_position(symbol, order_usd, price=price)
            return ok, msg, token

    def release(self, token: tuple | None) -> None:
        """Roll back a reservation made by check_and_reserve() (exact restore)."""
        if token is None:
            return
        symbol, prev_pos, prev_entry, prev_mark = token
        with self._lock:
            self._positions[symbol] = prev_pos
            if prev_entry is None:
                self._entry_price.pop(symbol, None)
            else:
                self._entry_price[symbol] = prev_entry
            self._mark_pnl[symbol] = prev_mark

    # ── Position tracking ─────────────────────────────────────────────────

    def update_position(
        self,
        symbol: str,
        delta_usd: float,
        price: float | None = None,
    ) -> None:
        """Update position after a fill (delta_usd is signed: + = long, - = short).

        When `price` is supplied, the entry-price basis is (re)set on a fresh
        open or a direction flip so that mark_to_market() can compute unrealized
        PnL. `price` is optional to keep the simple notional-only call site valid.
        """
        with self._lock:
            prev = self._positions.get(symbol, 0.0)
            new  = prev + delta_usd
            if price is not None:
                opened_or_flipped = prev == 0.0 or (prev > 0) != (new > 0)
                if opened_or_flipped:
                    self._entry_price[symbol] = price
                self._mark_pnl[symbol] = 0.0
            self._positions[symbol] = new

    def mark_to_market(self, symbol: str, price: float) -> tuple[float, bool]:
        """Mark an open position to market and feed the PnL change to the halt.

        Returns (pnl_delta_usd, halted). The incremental unrealized PnL since the
        last mark is added to the daily PnL, so the drawdown circuit breaker
        reacts to open positions moving against us — not just realized trades.
        """
        with self._lock:
            notional = self._positions.get(symbol, 0.0)
            entry    = self._entry_price.get(symbol)
            if notional == 0.0 or not entry:
                return 0.0, self._halted
            unrealized = notional * (price / entry - 1.0)
            delta = unrealized - self._mark_pnl.get(symbol, 0.0)
            self._mark_pnl[symbol] = unrealized
            halted = self.update_pnl(delta)
            return delta, halted

    def update_pnl(self, pnl_usd: float) -> bool:
        """
        Record PnL. Returns True if trading should halt (drawdown exceeded).
        """
        with self._lock:
            self._daily_pnl += pnl_usd
            if self._daily_pnl < -abs(self.nav * self.limits.max_drawdown_pct):
                if not self._halted:        # log/critical only on the transition
                    self._halted = True
                    log.critical(
                        "TRADING HALTED: daily PnL $%.0f exceeded drawdown limit %.0f%%",
                        self._daily_pnl,
                        self.limits.max_drawdown_pct * 100,
                    )
                return True
            return False

    def reset_daily(self) -> None:
        """Reset daily PnL counter (call at start of each trading day)."""
        with self._lock:
            self._daily_pnl = 0.0
            self._halted    = False
        log.info("Risk engine: daily PnL reset")

    # ── Volatility-Adjusted Position Sizing ───────────────────────────────

    def vol_adjusted_size(
        self,
        recent_vol_daily: float,
    ) -> float:
        """
        Compute position size in USD such that:
            position_size × annual_vol ≈ nav × vol_target

        Args:
            recent_vol_daily: Recent daily return standard deviation

        Returns:
            Position size in USD (capped at max_position_usd)
        """
        annual_vol = recent_vol_daily * np.sqrt(252)
        target_usd = self.nav * self.limits.vol_target_annual
        raw_size   = target_usd / (annual_vol + 1e-9)
        return float(min(raw_size, self.limits.max_position_usd))

    # ── Kelly Criterion ───────────────────────────────────────────────────

    def kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Fractional Kelly position sizing in USD.

        Full Kelly: f* = (p × b − q) / b
            p = win_rate
            q = 1 − win_rate
            b = avg_win / avg_loss

        Fractional Kelly = f* × kelly_fraction (default 0.25 = quarter-Kelly)

        Returns:
            Position size in USD
        """
        if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0:
            return 0.0

        b     = avg_win / avg_loss
        q     = 1.0 - win_rate
        f_star = max(0.0, (win_rate * b - q) / b)

        return float(min(
            self.nav * f_star * self.limits.kelly_fraction,
            self.limits.max_position_usd,
        ))

    # ── Portfolio summary ─────────────────────────────────────────────────

    def portfolio_summary(self) -> dict:
        with self._lock:
            total_exposure = sum(abs(v) for v in self._positions.values())
            return {
                "nav":             self.nav,
                "total_exposure":  total_exposure,
                "exposure_pct":    total_exposure / (self.nav + 1e-9) * 100,
                "daily_pnl":       self._daily_pnl,
                "halted":          self._halted,
                "positions":       dict(self._positions),
            }
