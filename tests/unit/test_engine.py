"""Unit tests for the ExecutionEngine signal→order path."""

from __future__ import annotations

import pytest

from borsabot.brokers.base import MockBrokerGateway
from borsabot.core.events import OrderSide, Signal
from borsabot.execution.engine import ExecutionEngine
from borsabot.market_data.order_book import OrderBook


def _engine(**kw) -> ExecutionEngine:
    # duration 0 → TWAP slice interval 0 → no real waiting in tests.
    return ExecutionEngine(
        broker=MockBrokerGateway(),
        default_twap_duration=0,
        default_twap_slices=4,
        **kw,
    )


def _signal() -> Signal:
    return Signal(symbol="BTCUSDT", side=OrderSide.BUY, confidence=0.9)


@pytest.mark.asyncio
async def test_no_book_path_does_not_raise_and_executes():
    """An invalid (empty) book must fail open without raising NameError."""
    engine = _engine()
    empty_book = OrderBook("BTCUSDT")          # no levels → is_valid() False
    orders = await engine.on_signal(_signal(), empty_book, quantity=1.0)
    assert orders, "fail-open path should still place orders"


@pytest.mark.asyncio
async def test_no_book_blocked_when_disallowed():
    engine = _engine(allow_no_book=False)
    empty_book = OrderBook("BTCUSDT")
    orders = await engine.on_signal(_signal(), empty_book, quantity=1.0)
    assert orders == []


@pytest.mark.asyncio
async def test_slice_quantities_sum_to_total_not_n_times():
    """Each tracked slice records its own qty — the sum equals the order size."""
    engine = _engine()
    empty_book = OrderBook("BTCUSDT")
    quantity = 1.0
    orders = await engine.on_signal(_signal(), empty_book, quantity=quantity)
    total_tracked = sum(o.request.quantity for o in orders)
    assert total_tracked == pytest.approx(quantity)
    # Sanity: with the old bug each of the N orders carried the full quantity.
    assert total_tracked < quantity * len(orders)
