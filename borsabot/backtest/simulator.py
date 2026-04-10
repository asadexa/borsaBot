"""Backtesting simulator with realistic slippage and fee modeling."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from borsabot.backtest.metrics import full_report
from borsabot.core.events import OrderSide

log = logging.getLogger(__name__)


class BacktestSimulator:
    """
    Event-driven backtesting simulator.

    Simulates order execution with:
      - Per-trade transaction costs (fee + slippage)
      - Mark-to-market PnL on each bar
      - No look-ahead bias (signals use only data up to current bar)

    Usage:
        sim    = BacktestSimulator(fee_bps=5, slippage_bps=3)
        result = sim.run(signals=signals_series, prices=close_series)
        print(result["metrics"])
    """

    def __init__(
        self,
        fee_bps: float = 5.0,
        slippage_bps: float = 3.0,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.fee_bps        = fee_bps
        self.slippage_bps   = slippage_bps
        self.initial_capital = initial_capital
        self._total_cost_bps = fee_bps + slippage_bps

    def fill_price(self, mid: float, side: OrderSide) -> float:
        """Simulate execution price with fees and slippage."""
        direction = 1 if side == OrderSide.BUY else -1
        return mid * (1.0 + direction * self._total_cost_bps / 10_000)

    def run(
        self,
        signals: pd.Series,    # index=timestamp, values ∈ {-1, 0, +1}
        prices: pd.Series,     # index=timestamp, close prices
    ) -> dict:
        """
        Run the backtest.

        Args:
            signals: Signal series aligned to prices.
            prices:  Close price series.

        Returns:
            Dict with keys: equity, returns, trades, metrics
        """
        equity_vals = [self.initial_capital]
        capital     = self.initial_capital
        position    = 0         # -1, 0, +1
        entry_price = 0.0
        trades: list[dict] = []

        returns_list: list[float] = []

        for i, (ts, sig) in enumerate(signals.items()):
            if ts not in prices.index:
                continue
            price = float(prices.loc[ts])

            # ── Mark-to-market PnL on current position ────────────────
            if i > 0 and position != 0:
                prev_price = float(prices.iloc[prices.index.get_loc(ts) - 1])
                bar_ret = (price - prev_price) / (prev_price + 1e-9) * position
                capital *= (1.0 + bar_ret)

            # ── Position change ────────────────────────────────────────
            if sig != 0 and sig != position:
                # Exit old position (if any)
                if position != 0:
                    exit_price = self.fill_price(
                        price, OrderSide.SELL if position > 0 else OrderSide.BUY
                    )
                    pnl = (exit_price - entry_price) * position
                    trades.append({
                        "exit_ts":    ts,
                        "exit_price": exit_price,
                        "pnl":        pnl,
                        "side":       "long" if position > 0 else "short",
                    })

                # Enter new position
                entry_price = self.fill_price(
                    price, OrderSide.BUY if sig > 0 else OrderSide.SELL
                )
                position = int(sig)
                if trades:
                    trades[-1].update({"entry_ts": ts, "entry_price": entry_price})

            elif sig == 0 and position != 0:
                # Close position
                exit_price = self.fill_price(
                    price, OrderSide.SELL if position > 0 else OrderSide.BUY
                )
                pnl = (exit_price - entry_price) * position
                trades.append({
                    "exit_ts": ts, "exit_price": exit_price,
                    "pnl": pnl, "side": "long" if position > 0 else "short",
                })
                position = 0

            returns_list.append(capital / equity_vals[-1] - 1)
            equity_vals.append(capital)

        equity   = pd.Series(equity_vals, name="equity")
        returns  = pd.Series(returns_list, name="returns")
        trades_df = pd.DataFrame(trades)
        metrics  = full_report(returns, equity)

        log.info(
            "Backtest complete: Sharpe=%.2f MDD=%.1f%% Trades=%d",
            metrics["sharpe"], metrics["max_drawdown"] * 100, len(trades_df),
        )

        return {
            "equity":   equity,
            "returns":  returns,
            "trades":   trades_df,
            "metrics":  metrics,
        }
