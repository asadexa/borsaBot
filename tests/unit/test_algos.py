"""Unit tests for execution algorithms: TWAP, VWAP, POV."""

import asyncio
import pytest
import numpy as np
from unittest.mock import AsyncMock

from borsabot.execution.algos import TWAP, VWAP, POV, select_algo
from borsabot.core.events import OrderSide, OrderStatus, OrderResponse


def make_fill_response(qty: float) -> OrderResponse:
    return OrderResponse(
        broker_order_id="b-999",
        client_order_id="",
        status=OrderStatus.FILLED,
        filled_qty=qty,
        avg_price=50_000.0,
    )


@pytest.fixture
def mock_broker():
    broker = AsyncMock()
    broker.name = "mock"
    return broker


# ── TWAP ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_twap_submits_exact_slices(mock_broker):
    mock_broker.submit_order.return_value = make_fill_response(0.001)
    twap = TWAP(duration_sec=1, slices=5)
    # Patch asyncio.sleep to avoid waiting
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        responses = await twap.execute(mock_broker, "BTCUSDT", OrderSide.BUY, 0.005)
    assert mock_broker.submit_order.call_count == 5
    assert len(responses) == 5


@pytest.mark.asyncio
async def test_twap_slice_qty_correct(mock_broker):
    mock_broker.submit_order.return_value = make_fill_response(0.001)
    twap = TWAP(duration_sec=1, slices=4)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        await twap.execute(mock_broker, "BTCUSDT", OrderSide.BUY, 0.004)
    # Each call should request 0.001
    for call in mock_broker.submit_order.call_args_list:
        req = call.args[0]
        assert req.quantity == pytest.approx(0.001, rel=1e-4)


# ── VWAP ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_vwap_total_qty_matches(mock_broker):
    mock_broker.submit_order.return_value = make_fill_response(1.0)
    profile = np.array([0.2, 0.3, 0.1, 0.4])
    vwap = VWAP(volume_profile=profile, interval_sec=0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        await vwap.execute(mock_broker, "BTCUSDT", OrderSide.BUY, total_qty=10.0)
    submitted = sum(c.args[0].quantity for c in mock_broker.submit_order.call_args_list)
    assert submitted == pytest.approx(10.0, rel=1e-4)


def test_vwap_profile_normalizes():
    profile = np.array([1, 2, 3, 4])
    vwap = VWAP(volume_profile=profile)
    assert vwap.volume_profile.sum() == pytest.approx(1.0)


# ── POV ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pov_completes_total_qty(mock_broker):
    mock_broker.submit_order.return_value = make_fill_response(0.5)
    pov = POV(pov_rate=0.5)
    await pov.execute(mock_broker, "BTCUSDT", OrderSide.BUY, 1.0)

    # Simulate market trades until filled
    submitted = 0.0
    while not pov.is_complete:
        resp = await pov.on_trade(1.0, mock_broker, "BTCUSDT", OrderSide.BUY)
        if resp:
            submitted += resp.filled_qty

    assert pov.remaining <= 0


@pytest.mark.asyncio
async def test_pov_never_exceeds_remaining():
    pov = POV(pov_rate=0.1)
    pov._remaining = 0.05
    # Market trade of 10 — at 10% rate → would want 1.0, but capped at 0.05
    broker = AsyncMock()
    broker.submit_order = AsyncMock(return_value=make_fill_response(0.05))

    resp = await pov.on_trade(10.0, broker, "BTCUSDT", OrderSide.BUY)
    assert broker.submit_order.call_args.args[0].quantity == pytest.approx(0.05)


# ── Algorithm selector ────────────────────────────────────────────────────────

def test_select_algo_high_urgency():
    assert select_algo("high", 0.01) == "twap"


def test_select_algo_low_urgency_no_profile():
    assert select_algo("low", 0.001) == "pov"


def test_select_algo_vwap_when_profile_and_size():
    assert select_algo("medium", 0.02, has_volume_profile=True) == "vwap"
