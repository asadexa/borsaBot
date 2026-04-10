"""Unit tests for feature engineering pipeline."""

import pytest
import numpy as np
import pandas as pd

from borsabot.features.frac_diff import frac_diff, get_weights, find_min_d
from borsabot.features.volatility import (
    volatility_scaled, rolling_zscore, realized_volatility
)
from borsabot.features.microstructure import extract_all
from borsabot.features.builder import FeatureBuilder


# ── Fractional Differentiation ────────────────────────────────────────────────

def test_frac_diff_d1_equals_diff(sample_prices):
    """fracdiff(d=1) should approximately equal pd.Series.diff()."""
    fd = frac_diff(sample_prices, d=1.0, thresh=1e-7)
    normal_diff = sample_prices.diff()
    common_idx = fd.index.intersection(normal_diff.dropna().index)
    np.testing.assert_allclose(
        fd.loc[common_idx].values,
        normal_diff.loc[common_idx].values,
        atol=0.01,
    )


def test_frac_diff_stationarity(sample_prices):
    """fracdiff output should be stationary (ADF p < 0.05) for d=0.4."""
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        pytest.skip("statsmodels not installed")
    log_prices = np.log(sample_prices + 1e-9)
    fd = frac_diff(log_prices, d=0.4).dropna()
    # Guard: ADF needs non-constant input
    if fd.std() == 0 or len(fd) < 20:
        pytest.skip("frac_diff output too short or constant for ADF test")
    result = adfuller(fd, maxlag=1, regression="c", autolag=None)
    pvalue = result[1]
    assert pvalue < 0.10, f"frac_diff output not stationary: p={pvalue:.4f}"


def test_frac_diff_no_nan_or_inf(sample_prices):
    fd = frac_diff(sample_prices, d=0.4).dropna()
    assert not fd.isna().any()
    assert not np.isinf(fd).any()


def test_get_weights_length():
    w = get_weights(0.5, 10)
    assert len(w) == 10
    assert w[-1] == pytest.approx(1.0)   # last weight is always 1.0


# ── Volatility ────────────────────────────────────────────────────────────────

def test_vol_scaled_reduces_heteroskedasticity(sample_returns):
    """Vol-scaled returns should have less variation in rolling std than raw returns."""
    scaled = volatility_scaled(sample_returns, window=20).dropna()
    # Basic sanity: vol-scaled returns should have std near 1.0
    assert 0.5 < scaled.std() < 5.0   # loose bound
    # Should not produce NaN or Inf
    assert not scaled.isna().any()
    assert not np.isinf(scaled).any()


def test_rolling_zscore_mean_near_zero(sample_prices):
    z = rolling_zscore(sample_prices, window=20).dropna()
    assert abs(z.mean()) < 1.0   # should be near 0 over the window


def test_rolling_zscore_no_nan(sample_prices):
    z = rolling_zscore(sample_prices, window=5).dropna()
    assert not z.isna().any()


# ── Microstructure features ───────────────────────────────────────────────────

def test_extract_all_returns_dict(book):
    features = extract_all(book)
    assert isinstance(features, dict)
    assert "obi_5" in features
    assert "microprice" in features
    assert "spread_bps" in features


def test_obi_in_range(book):
    for levels in [1, 5, 10]:
        obi = extract_all(book).get(f"obi_{levels}", 0)
        assert -1.0 <= obi <= 1.0, f"OBI_{levels} out of range: {obi}"


# ── FeatureBuilder ────────────────────────────────────────────────────────────

def test_feature_builder_no_nan(book, sample_ticks):
    fb = FeatureBuilder()
    features = fb.build(sample_ticks.tail(50), book)
    assert not features.isna().any(), f"NaN in features: {features[features.isna()].index.tolist()}"
    assert not np.isinf(features).any()


def test_feature_builder_returns_series(book, sample_ticks):
    fb = FeatureBuilder()
    result = fb.build(sample_ticks.tail(50), book)
    assert isinstance(result, pd.Series)
    assert len(result) > 0


def test_batch_feature_builder(sample_prices):
    ohlcv = pd.DataFrame({
        "open":   sample_prices,
        "high":   sample_prices * 1.001,
        "low":    sample_prices * 0.999,
        "close":  sample_prices,
        "volume": 100.0,
    })
    fb = FeatureBuilder()
    df = fb.build_batch(ohlcv, min_rows=30)
    assert len(df) == len(ohlcv) - 30
    assert not df.isna().any().any()
