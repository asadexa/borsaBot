"""Unit tests for Order FSM and OrderManager."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from borsabot.execution.order_fsm import Order, OrderManager
from borsabot.core.events import OrderRequest, OrderResponse, OrderSide, OrderStatus, OrderType


@pytest.fixture
def sample_request():
    return OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.01,
        client_order_id="test-001",
    )


@pytest.fixture
def manager():
    return OrderManager()


def make_response(status: OrderStatus, filled_qty: float = 0.0, price: float = 0.0) -> OrderResponse:
    return OrderResponse(
        broker_order_id="broker-999",
        client_order_id="test-001",
        status=status,
        filled_qty=filled_qty,
        avg_price=price,
    )


# ── State transitions ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_order_created_state(manager, sample_request):
    order = manager.create(sample_request)
    assert order.state == OrderStatus.CREATED.value


@pytest.mark.asyncio
async def test_order_full_fill_flow(manager, sample_request):
    order = manager.create(sample_request)
    await manager.handle_response(order, make_response(OrderStatus.SENT_TO_BROKER))
    await manager.handle_response(order, make_response(OrderStatus.PENDING))
    await manager.handle_response(order, make_response(OrderStatus.FILLED, filled_qty=0.01, price=50000))
    assert order.state == OrderStatus.FILLED.value
    assert order.filled_qty == pytest.approx(0.01)
    assert order.avg_price == pytest.approx(50000.0)


@pytest.mark.asyncio
async def test_partial_fill_accumulates(manager, sample_request):
    order = manager.create(sample_request)
    await manager.handle_response(order, make_response(OrderStatus.SENT_TO_BROKER))
    await manager.handle_response(order, make_response(OrderStatus.PENDING))
    await manager.handle_response(order, make_response(OrderStatus.PARTIALLY_FILLED, 0.005, 50000))
    await manager.handle_response(order, make_response(OrderStatus.FILLED, 0.01, 50100))
    assert order.filled_qty == pytest.approx(0.01)
    # Second fill: qty 0.01 - 0.005 = 0.005 at 50100
    expected_avg = (0.005 * 50000 + 0.005 * 50100) / 0.01
    assert order.avg_price == pytest.approx(expected_avg, rel=0.01)


@pytest.mark.asyncio
async def test_order_cancellation(manager, sample_request):
    order = manager.create(sample_request)
    await manager.handle_response(order, make_response(OrderStatus.SENT_TO_BROKER))
    await manager.handle_response(order, make_response(OrderStatus.PENDING))
    await manager.handle_response(order, make_response(OrderStatus.CANCELLED))
    assert order.state == OrderStatus.CANCELLED.value
    assert order.is_terminal


@pytest.mark.asyncio
async def test_open_orders_excludes_terminal(manager, sample_request):
    o1 = manager.create(sample_request)
    o2 = manager.create(sample_request)
    # Take o1 through full lifecycle
    await manager.handle_response(o1, make_response(OrderStatus.SENT_TO_BROKER))
    await manager.handle_response(o1, make_response(OrderStatus.PENDING))
    await manager.handle_response(o1, make_response(OrderStatus.FILLED, 0.01, 50000))
    assert o2 in manager.open_orders()
    assert o1 not in manager.open_orders()


# ── Failure handling ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_requote_retries_up_to_max(manager, sample_request):
    order = manager.create(sample_request)
    mock_broker = AsyncMock()
    mock_broker.submit_order.return_value = make_response(OrderStatus.REJECTED)

    for _ in range(OrderManager.MAX_RESUBMIT + 1):
        await manager.handle_reject(order, reason="requote", broker=mock_broker)

    # Should not exceed MAX_RESUBMIT resubmit calls
    assert order.resubmit_count <= OrderManager.MAX_RESUBMIT


@pytest.mark.asyncio
async def test_rejection_sets_rejected_state(manager, sample_request):
    order = manager.create(sample_request)
    await manager.handle_reject(order, reason="insufficient_margin")
    assert order.state == OrderStatus.REJECTED.value
    assert order.is_terminal


def test_order_latency_ms_is_non_negative(sample_request):
    order = Order(request=sample_request)
    assert order.latency_ms >= 0.0
