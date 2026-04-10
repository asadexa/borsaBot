"""Order Lifecycle Finite State Machine.

Manages the complete lifecycle of an order from creation to terminal state.
Uses the `transitions` library for clean FSM definition.

States:
  CREATED → SENT_TO_BROKER → PENDING → PARTIALLY_FILLED → FILLED
                                     ↓                          ↓
                                  CANCELLED                  CANCELLED
                 (any state) → REJECTED

Failure handling:
  - Re-quotes: automatic resubmit up to MAX_RESUBMIT times
  - Latency spikes: verify order status before acting to avoid duplicate fills
  - Exchange failures: cancel + publish risk alert
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from borsabot.core.events import OrderRequest, OrderResponse, OrderStatus

log = logging.getLogger(__name__)

try:
    from transitions import Machine
    _TRANSITIONS_AVAILABLE = True
except ImportError:
    _TRANSITIONS_AVAILABLE = False
    log.warning("transitions not installed — OrderFSM will use a simple state dict.")


# ─────────────────────────────────────────────────────────────────────────────
# Order FSM
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Order:
    """Represents a single order with full state management."""

    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request: OrderRequest | None = None
    broker_order_id: str = ""
    filled_qty: float = 0.0
    avg_price: float = 0.0
    resubmit_count: int = 0
    created_at_ns: int = field(default_factory=time.time_ns)
    updated_at_ns: int = field(default_factory=time.time_ns)

    # FSM state is managed by transitions Machine if available
    state: str = OrderStatus.CREATED.value

    # ── FSM transitions ───────────────────────────────────────────────────

    STATES = [s.value for s in OrderStatus]

    TRANSITIONS = [
        {"trigger": "submit",       "source": "created",          "dest": "sent_to_broker"},
        {"trigger": "acknowledge",  "source": "sent_to_broker",   "dest": "pending"},
        {"trigger": "partial_fill", "source": ["pending", "partially_filled"], "dest": "partially_filled"},
        {"trigger": "full_fill",    "source": ["pending", "partially_filled"], "dest": "filled"},
        {"trigger": "reject",       "source": "*",                "dest": "rejected"},
        {"trigger": "cancel",       "source": ["pending", "partially_filled", "sent_to_broker"],
                                                                   "dest": "cancelled"},
    ]

    def __post_init__(self) -> None:
        if _TRANSITIONS_AVAILABLE:
            self._machine = Machine(
                model=self,
                states=self.STATES,
                transitions=self.TRANSITIONS,
                initial=self.state,
                ignore_invalid_triggers=False,
            )

    def apply_fill(self, fill_qty: float, fill_price: float) -> None:
        """Update fill quantities and weighted average price."""
        prev_filled   = self.filled_qty
        self.filled_qty  += fill_qty
        # Weighted average price
        self.avg_price = (
            (prev_filled * self.avg_price + fill_qty * fill_price) /
            (self.filled_qty + 1e-9)
        )
        self.updated_at_ns = time.time_ns()

    @property
    def is_terminal(self) -> bool:
        return self.state in {
            OrderStatus.FILLED.value,
            OrderStatus.REJECTED.value,
            OrderStatus.CANCELLED.value,
        }

    @property
    def latency_ms(self) -> float:
        """Round-trip time from creation to last update."""
        return (self.updated_at_ns - self.created_at_ns) / 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Order Manager — handles failures, retries, and re-quotes
# ─────────────────────────────────────────────────────────────────────────────

class OrderManager:
    """
    Manages a collection of orders and handles failure scenarios:
      - Re-quotes: broker rejects with "requote" → retry up to MAX_RESUBMIT
      - Latency spike: verify order status before re-acting (prevent duplicate fills)
      - Exchange failure: cancel + log critical alert
    """

    MAX_RESUBMIT = 3
    LATENCY_SPIKE_MS = 500.0    # above this, verify before re-sending

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}

    def create(self, request: OrderRequest) -> Order:
        order = Order(request=request)
        self._orders[order.order_id] = order
        return order

    def get(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def open_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if not o.is_terminal]

    async def handle_response(
        self,
        order: Order,
        response: OrderResponse,
        broker=None,
    ) -> None:
        """Process a broker response and advance the FSM state."""
        status = response.status

        if status == OrderStatus.SENT_TO_BROKER:
            order.broker_order_id = response.broker_order_id
            if _TRANSITIONS_AVAILABLE:
                order.submit()
            else:
                order.state = OrderStatus.SENT_TO_BROKER.value

        elif status == OrderStatus.PENDING:
            if _TRANSITIONS_AVAILABLE:
                order.acknowledge()
            else:
                order.state = OrderStatus.PENDING.value

        elif status == OrderStatus.PARTIALLY_FILLED:
            order.apply_fill(response.filled_qty - order.filled_qty, response.avg_price)
            if _TRANSITIONS_AVAILABLE:
                order.partial_fill()
            order.state = OrderStatus.PARTIALLY_FILLED.value

        elif status == OrderStatus.FILLED:
            order.apply_fill(response.filled_qty - order.filled_qty, response.avg_price)
            if _TRANSITIONS_AVAILABLE:
                order.full_fill()
            else:
                order.state = OrderStatus.FILLED.value
            log.info(
                "Order FILLED: id=%s qty=%.4f avg_price=%.4f latency=%.1fms",
                order.order_id, order.filled_qty, order.avg_price, order.latency_ms,
            )

        elif status == OrderStatus.REJECTED:
            await self.handle_reject(order, reason="broker_rejection", broker=broker)

        elif status == OrderStatus.CANCELLED:
            if _TRANSITIONS_AVAILABLE:
                order.cancel()
            else:
                order.state = OrderStatus.CANCELLED.value

    async def handle_reject(
        self,
        order: Order,
        reason: str = "unknown",
        broker=None,
    ) -> None:
        if reason == "requote" and order.resubmit_count < self.MAX_RESUBMIT:
            order.resubmit_count += 1
            log.warning(
                "Requote detected — resubmitting order (attempt %d/%d)",
                order.resubmit_count, self.MAX_RESUBMIT,
            )
            await asyncio.sleep(0.1)
            if broker and order.request:
                resp = await broker.submit_order(order.request)
                await self.handle_response(order, resp, broker)
        elif reason == "latency_spike" and broker:
            log.warning("Latency spike — verifying order status before action")
            await self._verify_and_sync(order, broker)
        else:
            if _TRANSITIONS_AVAILABLE:
                order.reject()
            else:
                order.state = OrderStatus.REJECTED.value
            log.error("Order REJECTED: id=%s reason=%s", order.order_id, reason)

    async def _verify_and_sync(self, order: Order, broker) -> None:
        """After a latency spike, fetch true status from broker before acting."""
        try:
            live = await broker.get_order_status(order.broker_order_id)
            await self.handle_response(order, live, broker)
        except Exception as exc:
            log.error("Failed to verify order status: %s — cancelling", exc)
            try:
                await broker.cancel_order(order.broker_order_id)
            except Exception:
                pass
