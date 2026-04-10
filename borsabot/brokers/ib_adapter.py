"""Interactive Brokers adapter using ib_insync."""

from __future__ import annotations

import logging
import time
import uuid
from typing import AsyncIterator

from borsabot.brokers.base import BrokerGateway
from borsabot.core.events import OrderRequest, OrderResponse, OrderSide, OrderStatus, OrderType

log = logging.getLogger(__name__)

try:
    import ib_insync as ibi
    _IB_AVAILABLE = True
except ImportError:
    _IB_AVAILABLE = False
    log.warning("ib_insync not installed — IBAdapter will not function.")


class IBAdapter(BrokerGateway):
    """
    Interactive Brokers adapter via ib_insync.

    Supports stocks, futures, forex, and options through IB TWS / Gateway.
    Requires IB TWS or IB Gateway running locally.
    """

    name = "interactive_brokers"

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib: "ibi.IB | None" = None
        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self) -> None:
        if not _IB_AVAILABLE:
            raise RuntimeError("ib_insync is not installed.")
        self._ib = ibi.IB()
        await self._ib.connectAsync(self._host, self._port, clientId=self._client_id)
        self._connected = True
        log.info("IBAdapter connected (host=%s port=%d)", self._host, self._port)

    async def disconnect(self) -> None:
        if self._ib:
            self._ib.disconnect()
        self._connected = False
        log.info("IBAdapter disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected and (self._ib.isConnected() if self._ib else False)

    # ── Order Management ──────────────────────────────────────────────────

    def _make_contract(self, symbol: str) -> "ibi.Contract":
        """Resolve symbol string to an IB contract. Extend as needed."""
        # Default: US stock on SMART exchange
        return ibi.Stock(symbol, "SMART", "USD")

    async def submit_order(self, req: OrderRequest) -> OrderResponse:
        assert self._ib, "Not connected"
        contract = self._make_contract(req.symbol)

        if req.order_type == OrderType.MARKET:
            order = ibi.MarketOrder(req.side.value.upper(), req.quantity)
        elif req.order_type == OrderType.LIMIT and req.price:
            order = ibi.LimitOrder(req.side.value.upper(), req.quantity, req.price)
        else:
            raise ValueError(f"Unsupported order type: {req.order_type}")

        order.orderRef = req.client_order_id or str(uuid.uuid4())
        trade = self._ib.placeOrder(contract, order)

        # Wait briefly for immediate fill acknowledgment
        await self._ib.sleep(0.5)
        return self._map_trade(trade, req.client_order_id)

    async def cancel_order(self, broker_order_id: str) -> bool:
        assert self._ib, "Not connected"
        try:
            order = next(
                (t.order for t in self._ib.openTrades() if str(t.order.orderId) == broker_order_id),
                None,
            )
            if order:
                self._ib.cancelOrder(order)
                return True
            return False
        except Exception as exc:
            log.error("IB cancel error: %s", exc)
            return False

    async def get_order_status(self, broker_order_id: str) -> OrderResponse:
        assert self._ib, "Not connected"
        trade = next(
            (t for t in self._ib.trades() if str(t.order.orderId) == broker_order_id),
            None,
        )
        if trade is None:
            raise ValueError(f"Order {broker_order_id} not found")
        return self._map_trade(trade, "")

    async def get_positions(self) -> dict[str, float]:
        assert self._ib, "Not connected"
        await self._ib.reqPositionsAsync()
        return {
            p.contract.symbol: p.position
            for p in self._ib.positions()
            if p.position != 0
        }

    async def get_balance(self) -> dict[str, float]:
        assert self._ib, "Not connected"
        account_values = await self._ib.accountSummaryAsync()
        return {
            v.tag: float(v.value)
            for v in account_values
            if v.tag in ("NetLiquidation", "TotalCashValue", "AvailableFunds")
        }

    async def stream_market_data(self, symbols: list[str]) -> AsyncIterator[dict]:
        assert self._ib, "Not connected"
        contracts = [self._make_contract(s) for s in symbols]
        tickers = [self._ib.reqMktData(c) for c in contracts]
        while True:
            await self._ib.sleep(0.1)
            for t in tickers:
                if t.last:
                    yield {
                        "symbol": t.contract.symbol,
                        "price": t.last,
                        "bid": t.bid,
                        "ask": t.ask,
                        "ts": time.time_ns(),
                    }

    async def get_historical_klines(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> "pd.DataFrame":
        import pandas as pd
        if not self._ib:
            return pd.DataFrame()
        contract = self._make_contract(symbol)
        tf_map = {"1m": "1 min", "5m": "5 mins", "15m": "15 mins", "1h": "1 hour", "4h": "4 hours", "1d": "1 day"}
        duration = f"{limit} D" if timeframe == "1d" else f"{limit * 3} H"
        bars = await self._ib.reqHistoricalDataAsync(
            contract, endDateTime='', durationStr=duration,
            barSizeSetting=tf_map.get(timeframe, "1 day"), whatToShow='MIDPOINT', useRTH=True
        )
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["date"])
        df.set_index("time", inplace=True)
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _map_trade(trade: "ibi.Trade", client_id: str) -> OrderResponse:
        status_map = {
            "PreSubmitted": OrderStatus.SENT_TO_BROKER,
            "Submitted": OrderStatus.PENDING,
            "Filled": OrderStatus.FILLED,
            "PartialFill": OrderStatus.PARTIALLY_FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Inactive": OrderStatus.REJECTED,
        }
        fills = trade.fills
        avg_price = (
            sum(f.execution.price * f.execution.shares for f in fills) /
            sum(f.execution.shares for f in fills)
            if fills else 0.0
        )
        return OrderResponse(
            broker_order_id=str(trade.order.orderId),
            client_order_id=trade.order.orderRef or client_id,
            status=status_map.get(trade.orderStatus.status, OrderStatus.PENDING),
            filled_qty=trade.orderStatus.filled,
            avg_price=avg_price,
            timestamp_ns=time.time_ns(),
        )
