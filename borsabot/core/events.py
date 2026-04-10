"""Shared event primitives used across all platform components."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    CREATED = "created"
    SENT_TO_BROKER = "sent_to_broker"
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    # Market data
    TICK = "market.tick"
    BOOK_UPDATE = "market.book"
    OHLCV = "market.ohlcv"
    TRADE_STREAM = "market.trade"
    # Feature layer
    FEATURES = "features"
    # Signal layer
    SIGNAL = "signal"
    # Risk
    RISK_BREACH = "risk.breach"
    FEED_GAP = "risk.feed_gap"
    # Order lifecycle
    ORDER_CREATED = "order.created"
    ORDER_SENT = "order.sent"
    ORDER_FILLED = "order.filled"
    ORDER_REJECTED = "order.rejected"
    ORDER_CANCELLED = "order.cancelled"


# ─────────────────────────────────────────────────────────────────────────────
# Core Event envelope
# ─────────────────────────────────────────────────────────────────────────────

_seq_counter: int = 0


def _next_seq() -> int:
    global _seq_counter
    _seq_counter += 1
    return _seq_counter


@dataclass
class Event:
    """Canonical event envelope transmitted on the ZeroMQ message bus."""

    event_type: str                          # One of EventType values
    symbol: str
    payload: dict
    source: str = ""
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    sequence_id: int = field(default_factory=_next_seq)

    @property
    def topic(self) -> str:
        """ZeroMQ PUB/SUB topic string: '<event_type>.<symbol>'."""
        evt_type = self.event_type.value if hasattr(self.event_type, "value") else self.event_type
        return f"{evt_type}.{self.symbol}"


# ─────────────────────────────────────────────────────────────────────────────
# Order primitives
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    sl: float | None = None
    tp: float | None = None
    client_order_id: str = ""
    strategy_id: str = ""


@dataclass
class OrderResponse:
    broker_order_id: str
    client_order_id: str
    status: OrderStatus
    filled_qty: float
    avg_price: float
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())


# ─────────────────────────────────────────────────────────────────────────────
# Trading signal
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    symbol: str
    side: OrderSide
    confidence: float       # meta-model output ∈ [0, 1]
    strategy_id: str = ""
    sl: float | None = None
    tp: float | None = None
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    features: dict = field(default_factory=dict)
