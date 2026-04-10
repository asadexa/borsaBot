"""Fractional differentiation for financial time series stationarity.

Based on Marcos López de Prado — Advances in Financial Machine Learning, Ch.5.

Key idea: standard differencing (d=1) destroys memory.
Fractional differencing (e.g. d=0.4) achieves stationarity while
preserving maximum signal from the historical price path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_weights(d: float, size: int) -> np.ndarray:
    """
    Compute fractional differentiation weights for order d.

    The weight series decays geometrically, capturing long-memory
    while still converging to zero (unlike integer differencing).
    """
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1])   # oldest to newest


def frac_diff(
    series: pd.Series,
    d: float,
    thresh: float = 1e-5,
) -> pd.Series:
    """
    Apply fractional differentiation of order d to a price series.

    Args:
        series: Raw price/log-price series (should be log-prices for best results)
        d:      Differentiation order ∈ (0, 1). Typical: 0.3–0.5
        thresh: Drop weights below this threshold (truncates very old observations)

    Returns:
        Fractionally differentiated series (same index, NaN for initial window)
    """
    weights = get_weights(d, len(series))
    # Truncate negligible weights
    weights = weights[np.abs(weights) > thresh]
    width = len(weights)

    if width > len(series):
        return pd.Series(dtype=float, index=series.index)

    out: dict[pd.Timestamp, float] = {}
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1: i + 1].values
        out[series.index[i]] = float(np.dot(weights, window))

    return pd.Series(out, name=f"fracdiff_{d}")


def find_min_d(
    series: pd.Series,
    d_range: tuple[float, float] = (0.0, 1.0),
    step: float = 0.05,
    p_threshold: float = 0.05,
) -> float:
    """
    Find the minimum d that achieves stationarity (ADF p-value < p_threshold).

    This preserves as much memory as possible while satisfying stationarity.
    """
    from statsmodels.tsa.stattools import adfuller

    d = d_range[0]
    while d <= d_range[1]:
        fd = frac_diff(series, d).dropna()
        if len(fd) < 20:
            d += step
            continue
        result = adfuller(fd, maxlag=1, regression="c", autolag=None)
        pvalue = result[1]
        if pvalue < p_threshold:
            return round(d, 3)
        d += step

    return d_range[1]   # fallback to full differencing
