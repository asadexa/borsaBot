"""Integration tests for broker adapters (uses MockBrokerGateway)."""

import pytest
import asyncio
from borsabot.brokers.base import MockBrokerGateway, BrokerGateway
from borsabot.core.events import OrderRequest, OrderSide, OrderStatus, OrderType


ADAPTERS = [MockBrokerGateway]


@pytest.fixture(params=ADAPTERS)
def broker(request):
    return request.param()


# ── Contract tests — all adapters must satisfy these ─────────────────────────

@pytest.mark.asyncio
async def test_connect_disconnect(broker):
    await broker.connect()
    assert broker.is_connected
    await broker.disconnect()
    assert not broker.is_connected


@pytest.mark.asyncio
async def test_submit_market_order_returns_response(broker):
    await broker.connect()
    req = OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.001,
        client_order_id="test-int-001",
    )
    resp = await broker.submit_order(req)
    assert resp.status in (
        OrderStatus.FILLED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.PENDING,
    )
    assert resp.broker_order_id != ""
    await broker.disconnect()


@pytest.mark.asyncio
async def test_get_balance_returns_dict(broker):
    await broker.connect()
    balance = await broker.get_balance()
    assert isinstance(balance, dict)
    await broker.disconnect()


@pytest.mark.asyncio
async def test_get_positions_returns_dict(broker):
    await broker.connect()
    positions = await broker.get_positions()
    assert isinstance(positions, dict)
    await broker.disconnect()


@pytest.mark.asyncio
async def test_cancel_order(broker):
    await broker.connect()
    req = OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.001,
    )
    resp = await broker.submit_order(req)
    # cancel may return True or False depending on adapter/status
    result = await broker.cancel_order(resp.broker_order_id)
    assert isinstance(result, bool)
    await broker.disconnect()


@pytest.mark.asyncio
async def test_interface_completeness(broker):
    """All abstract methods of BrokerGateway must be implemented."""
    abstract_methods = {
        "connect", "disconnect", "is_connected",
        "submit_order", "cancel_order", "get_order_status",
        "get_positions", "get_balance", "stream_market_data",
    }
    for method in abstract_methods:
        assert hasattr(broker, method), f"Missing method: {method}"
