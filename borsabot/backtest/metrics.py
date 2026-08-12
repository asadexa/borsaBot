"""Performance metrics for backtesting and live trading evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe Ratio. Returns 0 if std = 0."""
    if returns.empty or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Sortino Ratio: penalizes only downside volatility.

    Denominator is the downside deviation = sqrt(mean(min(0, r)**2)) computed
    over ALL observations (not the sample std of the negative subset, which
    understates risk and uses the wrong N).
    """
    if returns.empty:
        return 0.0
    downside_sq = np.minimum(returns, 0.0) ** 2
    downside_dev = float(np.sqrt(downside_sq.mean()))
    if downside_dev == 0:
        return 0.0
    return float(returns.mean() / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative fraction, e.g. -0.15 = 15%)."""
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / (roll_max + 1e-9)
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, equity: pd.Series, periods: int = 252) -> float:
    """Annualized return / |Max Drawdown|."""
    annual_ret = returns.mean() * periods
    mdd        = abs(max_drawdown(equity))
    return float(annual_ret / (mdd + 1e-9))


def hit_rate(returns: pd.Series) -> float:
    """Fraction of positive return periods."""
    if returns.empty:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def profit_factor(returns: pd.Series) -> float:
    """Gross profits / Gross losses."""
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return float(gains / (losses + 1e-9))


def backtest_vs_live_drift(bt_sharpe: float, live_sharpe: float) -> dict:
    """Compare backtest and live Sharpe — detect performance decay."""
    if abs(bt_sharpe) < 1e-6:
        decay_pct = 0.0
    else:
        decay_pct = (bt_sharpe - live_sharpe) / abs(bt_sharpe) * 100

    return {
        "backtest_sharpe": bt_sharpe,
        "live_sharpe":     live_sharpe,
        "sharpe_decay_pct": decay_pct,
        "flag_retrain":    decay_pct > 30.0,   # retrain if > 30% decay
    }


def full_report(returns: pd.Series, equity: pd.Series) -> dict:
    """Compute full performance report dict."""
    return {
        "sharpe":       sharpe_ratio(returns),
        "sortino":      sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "calmar":       calmar_ratio(returns, equity),
        "hit_rate":     hit_rate(returns),
        "profit_factor": profit_factor(returns),
        "total_return": (
            float((equity.iloc[-1] / equity.iloc[0]) - 1)
            if not equity.empty and equity.iloc[0] != 0
            else 0.0
        ),
        "n_trades":     int((returns != 0).sum()),
    }
