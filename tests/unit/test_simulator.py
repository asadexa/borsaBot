"""Unit tests for the backtest simulator — transaction-cost accounting."""

from __future__ import annotations

import pandas as pd

from borsabot.backtest.simulator import BacktestSimulator


def _flat_market():
    """Flat prices so mark-to-market PnL is zero; only costs move equity."""
    idx = pd.date_range("2024-01-01", periods=3, freq="1h")
    prices  = pd.Series([100.0, 100.0, 100.0], index=idx)
    signals = pd.Series([1, 1, 0], index=idx)   # open, hold, close
    return signals, prices


def test_costs_drag_equity_on_flat_market():
    """A round-trip on a flat market must lose exactly the entry+exit cost."""
    signals, prices = _flat_market()
    sim = BacktestSimulator(fee_bps=5.0, slippage_bps=3.0, initial_capital=100_000.0)
    result = sim.run(signals, prices)

    final_equity = float(result["equity"].iloc[-1])
    cost_rate = (5.0 + 3.0) / 10_000.0
    expected = 100_000.0 * (1.0 - cost_rate) ** 2   # one entry + one exit

    assert final_equity == __import__("pytest").approx(expected, rel=1e-9)
    assert final_equity < 100_000.0          # costs actually reduced equity


def test_zero_costs_preserve_equity_on_flat_market():
    """With no fees/slippage, a flat-market round-trip leaves equity unchanged."""
    signals, prices = _flat_market()
    sim = BacktestSimulator(fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0)
    result = sim.run(signals, prices)

    assert float(result["equity"].iloc[-1]) == __import__("pytest").approx(100_000.0)


def test_higher_costs_reduce_sharpe():
    """Identical signals/prices but higher costs must not improve performance."""
    idx = pd.date_range("2024-01-01", periods=20, freq="1h")
    prices  = pd.Series([100.0 + i * 0.1 for i in range(20)], index=idx)  # gentle uptrend
    signals = pd.Series([1, 0] * 10, index=idx)                           # churn every bar

    cheap = BacktestSimulator(fee_bps=1.0, slippage_bps=0.0).run(signals, prices)
    pricey = BacktestSimulator(fee_bps=50.0, slippage_bps=20.0).run(signals, prices)

    assert float(pricey["equity"].iloc[-1]) < float(cheap["equity"].iloc[-1])
