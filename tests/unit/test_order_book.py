"""Unit tests for OrderBook engine."""

import pytest
import numpy as np
from borsabot.market_data.order_book import OrderBook


def test_snapshot_loads_correctly(book):
    assert book.best_bid() == pytest.approx(50000.0)
    assert book.best_ask() == pytest.approx(50010.0)
    assert book.last_update_id == 100


def test_mid_price(book):
    assert book.mid_price() == pytest.approx(50005.0)


def test_spread(book):
    assert book.spread() == pytest.approx(10.0)


def test_spread_bps(book):
    # spread / mid * 10000
    expected = 10.0 / 50005.0 * 10_000
    assert book.spread_bps() == pytest.approx(expected, rel=1e-4)


def test_microprice_between_bid_and_ask(book):
    mp = book.microprice()
    assert book.best_bid() < mp < book.best_ask()


def test_obi_in_valid_range(book):
    for levels in [1, 5, 10]:
        obi = book.order_book_imbalance(levels)
        assert -1.0 <= obi <= 1.0


def test_obi_balanced_book():
    """Equal bid and ask depth → OBI = 0."""
    b = OrderBook("BTCUSDT")
    b.apply_snapshot({
        "lastUpdateId": 1,
        "bids": [[100.0, 5.0], [99.0, 5.0]],
        "asks": [[101.0, 5.0], [102.0, 5.0]],
    })
    assert b.order_book_imbalance(2) == pytest.approx(0.0, abs=1e-6)


def test_stale_delta_ignored(book):
    """Delta with u <= last_update_id should be discarded."""
    stale = {"U": 50, "u": 99, "b": [[50000.0, 999.0]], "a": []}
    result = book.apply_delta(stale)
    assert result is False
    assert book.bids[50000.0] == 1.0   # unchanged


def test_valid_delta_applied(book):
    delta = {
        "U": 101, "u": 102,
        "b": [[50000.0, 5.0]],  # update best bid qty
        "a": [],
    }
    result = book.apply_delta(delta)
    assert result is True
    assert book.bids[50000.0] == pytest.approx(5.0)


def test_delta_removes_level(book):
    delta = {"U": 101, "u": 102, "b": [[50000.0, 0.0]], "a": []}
    book.apply_delta(delta)
    assert 50000.0 not in book.bids


def test_book_validity(book):
    assert book.is_valid()


def test_crossed_book_is_invalid():
    b = OrderBook("TEST")
    b.apply_snapshot({
        "lastUpdateId": 1,
        "bids": [[101.0, 1.0]],   # bid > ask = crossed
        "asks": [[100.0, 1.0]],
    })
    assert not b.is_valid()


def test_depth_trim_respected():
    b = OrderBook("TEST", depth=3)
    b.apply_snapshot({
        "lastUpdateId": 1,
        "bids": [[100.0, 1], [99.0, 1], [98.0, 1], [97.0, 1], [96.0, 1]],
        "asks": [[101.0, 1], [102.0, 1], [103.0, 1], [104.0, 1], [105.0, 1]],
    })
    assert len(b.bids) == 3
    assert len(b.asks) == 3


def test_serialization_round_trip(book):
    d = book.to_dict()
    restored = OrderBook.from_dict(d)
    assert restored.best_bid() == pytest.approx(book.best_bid())
    assert restored.best_ask() == pytest.approx(book.best_ask())
    assert restored.last_update_id == book.last_update_id
