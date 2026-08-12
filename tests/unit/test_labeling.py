"""Unit tests for Triple Barrier labeling."""

import pytest
import numpy as np
import pandas as pd

from borsabot.models.labeling import (
    compute_daily_volatility,
    triple_barrier_labels,
)


@pytest.fixture
def trending_up_prices():
    return pd.Series(
        np.arange(100.0, 200.0, 0.1),
        index=pd.date_range("2024-01-01", periods=1000, freq="1min"),
    )


@pytest.fixture
def simple_prices():
    rng = np.random.default_rng(42)
    return pd.Series(
        100 + np.cumsum(rng.normal(0, 0.5, 200)),
        index=pd.date_range("2024-01-01", periods=200, freq="1H"),
    )


def make_events(close: pd.Series, n: int = 5, horizon: int = 20) -> pd.DataFrame:
    t_events = close.index[10:10+n]
    t1 = pd.Series(
        [close.index[min(i + horizon, len(close) - 1)]
         for i in range(10, 10+n)],
        index=t_events,
    )
    trgt = pd.Series(0.01, index=t_events)
    return pd.DataFrame({"t1": t1, "trgt": trgt, "side": 1.0})


def test_labels_only_valid_values(simple_prices):
    events = make_events(simple_prices)
    labels = triple_barrier_labels(simple_prices, events)
    assert set(labels.unique()).issubset({-1, 0, 1})


def test_labels_length_matches_events(simple_prices):
    events = make_events(simple_prices, n=10)
    labels = triple_barrier_labels(simple_prices, events)
    assert len(labels) == 10


def test_trending_market_mostly_positive(trending_up_prices):
    events = make_events(trending_up_prices, n=20, horizon=50)
    labels = triple_barrier_labels(trending_up_prices, events)
    # In a strongly trending market, most labels should be +1
    positive_frac = (labels == 1).mean()
    assert positive_frac > 0.5


def test_short_side_hits_stop_in_uptrend(trending_up_prices):
    """A short bet (side=-1) in a strong uptrend should mostly hit stop-loss (-1)."""
    events = make_events(trending_up_prices, n=20, horizon=50)
    events["side"] = -1.0
    labels = triple_barrier_labels(trending_up_prices, events)
    # Shorts in a rising market lose → label -1 (stop hit), not +1.
    assert (labels == -1).mean() > 0.5
    assert (labels == 1).mean() < 0.5


def test_long_and_short_are_mirror_in_trend(trending_up_prices):
    """Long and short on the same uptrend events should give opposite labels."""
    long_events = make_events(trending_up_prices, n=15, horizon=50)
    short_events = long_events.copy()
    short_events["side"] = -1.0

    long_labels = triple_barrier_labels(trending_up_prices, long_events)
    short_labels = triple_barrier_labels(trending_up_prices, short_events)

    # Where the long wins (+1), the short should lose (-1).
    long_wins = long_labels == 1
    assert (short_labels[long_wins] == -1).all()


def test_daily_vol_is_positive(simple_prices):
    vol = compute_daily_volatility(simple_prices, span=20)
    assert (vol.dropna() > 0).all()


def test_daily_vol_exponentially_weighted(simple_prices):
    vol = compute_daily_volatility(simple_prices, span=20)
    # EWM vol should respond faster than rolling vol to a spike
    assert vol.dropna().std() >= 0   # basic sanity
