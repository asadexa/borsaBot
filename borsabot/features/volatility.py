"""Volatility normalization for financial returns.

Financial returns are heteroskedastic — volatility clusters over time.
Normalizing by rolling volatility makes features and returns more stationary
and improves model stability significantly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_scaled(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Divide returns by their rolling standard deviation.

    Result has unit volatility over the rolling window.
    Eliminates volatility regime differences between calm and crisis periods.
    """
    vol = returns.rolling(window, min_periods=1).std()
    return returns / (vol + 1e-9)


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Standard rolling z-score normalization: (x - μ) / σ

    Result has rolling mean≈0 and std≈1 over the window.
    Useful for mean-reversion signals and cross-asset comparisons.
    """
    mu  = series.rolling(window, min_periods=1).mean()
    sig = series.rolling(window, min_periods=1).std()
    return (series - mu) / (sig + 1e-9)


def parkinson_vol(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
    trading_periods: int = 252,
) -> pd.Series:
    """
    Parkinson (1980) high-low volatility estimator.

    More efficient than close-to-close when intraday (H/L) data is available.
    Annualized, suitable for position sizing.
    """
    hl_log = np.log(high / low) ** 2
    factor = 1 / (4 * np.log(2))
    daily_var = factor * hl_log.rolling(window, min_periods=1).mean()
    return np.sqrt(daily_var * trading_periods)


def realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Annualized realized volatility (std of returns × √252)."""
    return returns.rolling(window, min_periods=1).std() * np.sqrt(252)


def vol_regime(returns: pd.Series, window: int = 20, threshold: float = 0.20) -> pd.Series:
    """
    Simple binary volatility regime: 1 = high-vol, 0 = low-vol.
    Threshold is annualized vol (e.g. 0.20 = 20%).
    """
    rv = realized_volatility(returns, window)
    return (rv > threshold).astype(int)
