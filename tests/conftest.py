"""Shared fixtures for all tests."""

from __future__ import annotations

import asyncio
import sys
import pytest
import numpy as np
import pandas as pd

from borsabot.market_data.order_book import OrderBook
from borsabot.brokers.base import MockBrokerGateway

# ── Windows ZeroMQ Fix ────────────────────────────────────────────────────────
# Windows uses ProactorEventLoop by default, which does NOT support the
# add_reader() method required by zmq.asyncio. We switch to SelectorEventLoop
# to make ZeroMQ work correctly in tests.
#
# The module-level set_event_loop_policy() is not enough because pytest-asyncio
# manages its own event loop. We use the official `event_loop_policy` fixture
# so pytest-asyncio picks up our policy before each test.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture
def book() -> OrderBook:
    """A valid L2 order book with 5 price levels each side."""
    b = OrderBook(symbol="BTCUSDT", depth=20)
    b.apply_snapshot({
        "lastUpdateId": 100,
        "bids": [
            [50000.0, 1.0],
            [49990.0, 2.0],
            [49980.0, 3.0],
            [49970.0, 1.5],
            [49960.0, 0.5],
        ],
        "asks": [
            [50010.0, 1.0],
            [50020.0, 2.0],
            [50030.0, 3.0],
            [50040.0, 1.5],
            [50050.0, 0.5],
        ],
    })
    return b


@pytest.fixture
def mock_broker() -> MockBrokerGateway:
    return MockBrokerGateway()


@pytest.fixture
def sample_prices() -> pd.Series:
    rng = np.random.default_rng(42)
    prices = pd.Series(
        100 + np.cumsum(rng.normal(0, 1, 500)),
        index=pd.date_range("2024-01-01", periods=500, freq="1min"),
        name="price",
    )
    return prices


@pytest.fixture
def sample_returns(sample_prices) -> pd.Series:
    return sample_prices.pct_change().dropna()


@pytest.fixture
def sample_ticks(sample_prices) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "time":  sample_prices.index,
        "price": sample_prices.values,
        "qty":   rng.uniform(0.01, 1.0, len(sample_prices)),
        "side":  rng.choice(["B", "S"], len(sample_prices)),
    }).set_index("time")
