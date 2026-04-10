"""Real-time Risk Engine.

Enforces position limits, NAV exposure caps, volatility-adjusted sizing,
Kelly criterion allocation, and daily drawdown halt guard.
"""

from __future__ import annotations

import logging

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
        self._positions: dict[str, float] = {}    # symbol → USD value
        self._daily_pnl: float = 0.0
        self._halted:    bool  = False

    # ── Position & Exposure Checks ────────────────────────────────────────

    def check_new_order(
        self,
        symbol: str,
        order_usd: float,
    ) -> tuple[bool, str]:
        """
        Validate a new order against all risk limits.

        Returns (allowed: bool, reason: str)
        """
        if self._halted:
            return False, "Trading halted: daily drawdown limit reached"

        current_pos_usd = abs(self._positions.get(symbol, 0.0))
        new_pos_usd     = current_pos_usd + abs(order_usd)

        # ── Single position size limit ─────────────────────────────────
        if new_pos_usd > self.limits.max_position_usd:
            return False, (
                f"Position limit exceeded: "
                f"${new_pos_usd:,.0f} > ${self.limits.max_position_usd:,.0f}"
            )

        # ── Single asset NAV% limit ────────────────────────────────────
        nav_pct = new_pos_usd / (self.nav + 1e-9)
        if nav_pct > self.limits.max_portfolio_pct:
            return False, (
                f"NAV% limit exceeded: "
                f"{nav_pct*100:.1f}% > {self.limits.max_portfolio_pct*100:.0f}%"
            )

        return True, "OK"

    # ── Position tracking ─────────────────────────────────────────────────

    def update_position(self, symbol: str, delta_usd: float) -> None:
        """Update position after a fill (delta_usd is signed: + = long, - = short)."""
        self._positions[symbol] = self._positions.get(symbol, 0.0) + delta_usd

    def update_pnl(self, pnl_usd: float) -> bool:
        """
        Record PnL. Returns True if trading should halt (drawdown exceeded).
        """
        self._daily_pnl += pnl_usd
        if self._daily_pnl < -abs(self.nav * self.limits.max_drawdown_pct):
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
        total_exposure = sum(abs(v) for v in self._positions.values())
        return {
            "nav":             self.nav,
            "total_exposure":  total_exposure,
            "exposure_pct":    total_exposure / (self.nav + 1e-9) * 100,
            "daily_pnl":       self._daily_pnl,
            "halted":          self._halted,
            "positions":       dict(self._positions),
        }
