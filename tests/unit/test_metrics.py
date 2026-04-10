"""Unit tests for backtest metrics."""

import pytest
import numpy as np
import pandas as pd

from borsabot.backtest.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown,
    hit_rate, profit_factor, backtest_vs_live_drift,
)


@pytest.fixture
def positive_returns():
    # Mean 1%, std 0.1% → strong positive Sharpe
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(0.01, 0.001, 252))


@pytest.fixture
def negative_returns():
    # Mean -1%, std 0.1% → strong negative Sharpe
    rng = np.random.default_rng(2)
    return pd.Series(rng.normal(-0.01, 0.001, 252))


@pytest.fixture
def mixed_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.001, 0.02, 252))


def test_sharpe_positive_for_positive_returns(positive_returns):
    assert sharpe_ratio(positive_returns) > 0


def test_sharpe_negative_for_negative_returns(negative_returns):
    assert sharpe_ratio(negative_returns) < 0


def test_sharpe_zero_for_zero_std():
    flat = pd.Series([0.0] * 100)
    assert sharpe_ratio(flat) == 0.0


def test_max_drawdown_total_loss():
    equity = pd.Series([1.0, 0.5, 0.01])
    mdd = max_drawdown(equity)
    assert mdd < -0.95


def test_max_drawdown_no_loss(positive_returns):
    equity = (1 + positive_returns).cumprod()
    assert max_drawdown(equity) == pytest.approx(0.0, abs=1e-5)


def test_hit_rate_all_positive(positive_returns):
    assert hit_rate(positive_returns) == pytest.approx(1.0)


def test_hit_rate_all_negative(negative_returns):
    assert hit_rate(negative_returns) == pytest.approx(0.0)


def test_profit_factor_above_one_for_profitable(mixed_returns):
    profitable = pd.Series([0.02, 0.01, -0.005, 0.03, -0.01])
    pf = profit_factor(profitable)
    assert pf > 1.0


def test_backtest_vs_live_no_drift():
    result = backtest_vs_live_drift(1.0, 1.0)
    assert result["sharpe_decay_pct"] == pytest.approx(0.0)
    assert not result["flag_retrain"]


def test_backtest_vs_live_flags_retrain():
    result = backtest_vs_live_drift(2.0, 0.5)
    assert result["sharpe_decay_pct"] == pytest.approx(75.0)
    assert result["flag_retrain"]
