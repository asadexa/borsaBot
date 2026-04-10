"""Binance adapter — WebSocket market data + REST order management."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import AsyncIterator

from borsabot.brokers.base import BrokerGateway
from borsabot.core.events import OrderRequest, OrderResponse, OrderSide, OrderStatus, OrderType

log = logging.getLogger(__name__)

try:
    from binance import AsyncClient, BinanceSocketManager
    from binance.exceptions import BinanceAPIException
    _BINANCE_AVAILABLE = True
except ImportError:
    _BINANCE_AVAILABLE = False
    log.warning("python-binance not installed — BinanceAdapter will not function.")


class BinanceAdapter(BrokerGateway):
    """
    Binance broker adapter.

    Market data: WebSocket depth + aggTrade streams (via BinanceSocketManager)
    Order management: Binance REST API (via AsyncClient)

    Testnet support: set BINANCE_TESTNET=true in .env
    """

    name = "binance"

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._client: "AsyncClient | None" = None
        self._bm: "BinanceSocketManager | None" = None
        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self) -> None:
        if not _BINANCE_AVAILABLE:
            raise RuntimeError("python-binance is not installed.")
        self._client = await AsyncClient.create(
            api_key=self._api_key,
            api_secret=self._api_secret,
            testnet=self._testnet,
        )
        self._bm = BinanceSocketManager(self._client)
        self._connected = True
        log.info("BinanceAdapter connected (testnet=%s)", self._testnet)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close_connection()
        self._connected = False
        log.info("BinanceAdapter disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Order Management ──────────────────────────────────────────────────

    async def submit_order(self, req: OrderRequest) -> OrderResponse:
        assert self._client, "Not connected"
        try:
            kwargs: dict = {
                "symbol": req.symbol,
                "side": req.side.value.upper(),
                "type": req.order_type.value.upper(),
                "quantity": str(req.quantity),
                "newClientOrderId": req.client_order_id or str(uuid.uuid4()),
            }
            if req.order_type == OrderType.LIMIT and req.price:
                kwargs["price"] = str(req.price)
                kwargs["timeInForce"] = "GTC"

            raw = await self._client.create_order(**kwargs)
            return self._map_order(raw, req.client_order_id)

        except BinanceAPIException as exc:
            log.error("Binance order error: %s", exc)
            return OrderResponse(
                broker_order_id="",
                client_order_id=req.client_order_id,
                status=OrderStatus.REJECTED,
                filled_qty=0.0,
                avg_price=0.0,
            )

    async def cancel_order(self, broker_order_id: str) -> bool:
        assert self._client, "Not connected"
        try:
            await self._client.cancel_order(orderId=int(broker_order_id))
            return True
        except BinanceAPIException as exc:
            log.error("Cancel failed: %s", exc)
            return False

    async def get_order_status(self, broker_order_id: str) -> OrderResponse:
        assert self._client, "Not connected"
        raw = await self._client.get_order(orderId=int(broker_order_id))
        return self._map_order(raw, "")

    async def get_positions(self) -> dict[str, float]:
        """For spot: return non-zero balances as pseudo-positions."""
        info = await self._client.get_account()
        return {
            b["asset"]: float(b["free"]) + float(b["locked"])
            for b in info["balances"]
            if float(b["free"]) + float(b["locked"]) > 0
        }

    async def get_balance(self) -> dict[str, float]:
        info = await self._client.get_account()
        return {b["asset"]: float(b["free"]) for b in info["balances"] if float(b["free"]) > 0}

    # ── Market Data ───────────────────────────────────────────────────────

    async def stream_market_data(self, symbols: list[str]) -> AsyncIterator[dict]:
        assert self._bm, "Not connected"
        streams = [f"{s.lower()}@depth20@100ms" for s in symbols]
        socket = self._bm.multiplex_socket(streams)
        async with socket as stream:
            while True:
                msg = await stream.recv()
                if msg:
                    # Multiplex mesajlari {stream: "...", data: {...}} seklinde gelir.
                    yield msg.get("data", msg)

    async def stream_agg_trades(self, symbols: list[str]) -> AsyncIterator[dict]:
        assert self._bm, "Not connected"
        streams = [f"{s.lower()}@aggTrade" for s in symbols]
        socket = self._bm.multiplex_socket(streams)
        async with socket as stream:
            while True:
                msg = await stream.recv()
                if msg:
                    yield msg.get("data", msg)

    async def get_historical_klines(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> "pd.DataFrame":
        import pandas as pd
        if not self._client:
            return pd.DataFrame()
        res = await self._client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
        if not res:
            return pd.DataFrame()
        df = pd.DataFrame(res, columns=["time", "open", "high", "low", "close", "volume", 
                                        "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _map_order(raw: dict, client_id: str) -> OrderResponse:
        status_map = {
            "NEW": OrderStatus.PENDING,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.CANCELLED,
        }
        filled_qty = float(raw.get("executedQty", 0))
        fills = raw.get("fills", [])
        avg_price = (
            sum(float(f["price"]) * float(f["qty"]) for f in fills) / (filled_qty or 1)
            if fills else float(raw.get("price", 0))
        )
        return OrderResponse(
            broker_order_id=str(raw.get("orderId", "")),
            client_order_id=raw.get("clientOrderId", client_id),
            status=status_map.get(raw.get("status", ""), OrderStatus.PENDING),
            filled_qty=filled_qty,
            avg_price=avg_price,
            timestamp_ns=int(raw.get("transactTime", time.time_ns() // 1_000_000)) * 1_000_000,
        )
