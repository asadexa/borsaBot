"""Broker Abstraction Layer — Abstract Base Class.

All broker-specific logic is hidden behind this interface.
Strategy and execution code never import broker SDKs directly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator

from borsabot.core.events import OrderRequest, OrderResponse

log = logging.getLogger(__name__)


class BrokerGateway(ABC):
    """Abstract broker interface. All adapters must implement this contract."""

    name: str = "base"

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the broker / exchange."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleanly disconnect from the broker / exchange."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the broker connection is live."""

    # ── Order Management ──────────────────────────────────────────────────

    @abstractmethod
    async def submit_order(self, req: OrderRequest) -> OrderResponse:
        """Submit a new order. Returns the broker's response."""

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""

    @abstractmethod
    async def get_order_status(self, broker_order_id: str) -> OrderResponse:
        """Fetch current status of an existing order."""

    # ── Account ───────────────────────────────────────────────────────────

    @abstractmethod
    async def get_positions(self) -> dict[str, float]:
        """Return dict of {symbol: quantity} for open positions."""

    @abstractmethod
    async def get_balance(self) -> dict[str, float]:
        """Return dict of {asset: amount} for account balances."""

    # ── Market Data ───────────────────────────────────────────────────────

    @abstractmethod
    async def stream_market_data(self, symbols: list[str]) -> AsyncIterator[dict]:
        """Yield raw market data dicts from broker's websocket feed."""

    @abstractmethod
    async def get_historical_klines(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> "pd.DataFrame":
        """Fetch historical klines/candles. Returns DataFrame with DatetimeIndex and [open, high, low, close, volume]."""

class MockBrokerGateway(BrokerGateway):
    """In-memory mock adapter used in unit tests — no network required."""

    name = "mock"
    _connected = False
    _orders: dict[str, OrderResponse] = {}

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def submit_order(self, req: OrderRequest) -> OrderResponse:
        import time, uuid
        oid = str(uuid.uuid4())
        resp = OrderResponse(
            broker_order_id=oid,
            client_order_id=req.client_order_id,
            status="filled",  # type: ignore[arg-type]
            filled_qty=req.quantity,
            avg_price=req.price or 50_000.0,
            timestamp_ns=time.time_ns(),
        )
        self._orders[oid] = resp
        return resp

    async def cancel_order(self, broker_order_id: str) -> bool:
        return broker_order_id in self._orders

    async def get_order_status(self, broker_order_id: str) -> OrderResponse:
        return self._orders[broker_order_id]

    async def get_positions(self) -> dict[str, float]:
        return {}

    async def get_balance(self) -> dict[str, float]:
        return {"USDT": 100_000.0}

    async def get_historical_klines(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> "pd.DataFrame":
        import pandas as pd
        import numpy as np
        dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq="D")
        return pd.DataFrame({
            "open": np.random.randn(limit) + 1.05,
            "high": np.random.randn(limit) + 1.06,
            "low": np.random.randn(limit) + 1.04,
            "close": np.random.randn(limit) + 1.05,
            "volume": np.random.rand(limit) * 1000,
        }, index=dates)

    async def stream_market_data(
        self,
        symbols: list[str],
        on_tick=None,
    ) -> None:
        """Coroutine — sürekli sentetik tick üretir, on_tick callback'i çağırır."""
        import asyncio, time, random
        base_prices = {"BTCUSDT": 65_000.0, "ETHUSDT": 3_500.0,
                       "BTC-USD": 65_000.0, "ETH-USD": 3_500.0}
        prices = {s: base_prices.get(s, 50_000.0) for s in symbols}
        seq: dict[str, int] = {s: 0 for s in symbols}
        log.info("MockBroker: tick stream basladi (%s)", symbols)
        while True:
            for sym in symbols:
                prices[sym] *= (1 + random.gauss(0, 0.0005))
                seq[sym] += 1
                tick = {
                    "symbol": sym,
                    "s":      sym,
                    "price":  round(prices[sym], 2),
                    "qty":    round(random.uniform(0.001, 0.1), 6),
                    "side":   random.choice(["b", "s"]),
                    "ts":     time.time_ns(),
                    "_seq":   seq[sym],
                }
                if on_tick is not None:
                    await on_tick(tick)
            await asyncio.sleep(1.0)   # 1 tick/saniye
