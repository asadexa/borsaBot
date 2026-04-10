"""ExecutionEngine — orchestrates the full order execution lifecycle.

Receives a trading Signal, runs pre-execution checks (slippage, fill prob,
risk), selects an execution algorithm, and routes orders through the broker
gateway while managing order state via OrderManager.
"""

from __future__ import annotations

import asyncio
import logging
import time

from borsabot.brokers.base import BrokerGateway
from borsabot.core.events import OrderRequest, OrderSide, OrderStatus, OrderType, Signal
from borsabot.execution.algos import TWAP, VWAP, BaseAlgo, select_algo
from borsabot.execution.fill_predictor import FillPredictor
from borsabot.execution.order_fsm import Order, OrderManager, _TRANSITIONS_AVAILABLE
from borsabot.execution.slippage import SlippageEstimator
from borsabot.market_data.order_book import OrderBook
from borsabot.monitoring.metrics import (
    order_latency_ms,
    orders_submitted,
)

log = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Execution-aware order routing engine.

    Pipeline per signal:
      1. Slippage estimate → abort if cost > max_slippage_bps
      2. Fill probability  → abort if P(fill) < min_fill_prob
      3. Algorithm selection (TWAP / VWAP / POV)
      4. Order submission via BrokerGateway
      5. FSM state tracking

    All decisions are configurable via constructor parameters.
    """

    def __init__(
        self,
        broker: BrokerGateway,
        order_manager: OrderManager | None = None,
        slippage_estimator: SlippageEstimator | None = None,
        fill_predictor: FillPredictor | None = None,
        max_slippage_bps: float = 10.0,
        min_fill_prob: float = 0.50,
        default_twap_duration: int = 300,   # 5 minutes
        default_twap_slices: int = 10,
    ) -> None:
        self.broker            = broker
        self.order_manager     = order_manager or OrderManager()
        self.slippage          = slippage_estimator or SlippageEstimator()
        self.fill_predictor    = fill_predictor or FillPredictor()
        self.max_slippage_bps  = max_slippage_bps
        self.min_fill_prob     = min_fill_prob
        self._twap_duration    = default_twap_duration
        self._twap_slices      = default_twap_slices

    # ── Main entry point ──────────────────────────────────────────────────

    async def on_signal(
        self,
        signal: Signal,
        book: OrderBook,
        quantity: float,
        urgency: str = "medium",
    ) -> list[Order]:
        """
        Process a trading signal end-to-end.

        Args:
            signal:   Trading signal from AI model layer
            book:     Live order book for cost estimation
            quantity: Order quantity in base asset units
            urgency:  "low" | "medium" | "high" (affects algo selection)

        Returns:
            List of created Order objects (may be multiple slices).
        """
        symbol = signal.symbol
        side   = signal.side

        # ── Pre-execution checks ──────────────────────────────────────────

        signed_qty = quantity if side == OrderSide.BUY else -quantity

        slippage_bps = self.slippage.estimate_bps(signed_qty, book)
        if slippage_bps > self.max_slippage_bps:
            log.warning(
                "Signal BLOCKED: slippage %.1f bps > max %.1f bps [%s %s]",
                slippage_bps, self.max_slippage_bps, side.value, symbol,
            )
            return []

        if book.is_valid():
            ref_price = book.best_ask() if side == OrderSide.BUY else book.best_bid()
            fill_prob = self.fill_predictor.predict(book, ref_price, quantity)
        else:
            fill_prob = 1.0  # no book data -> assume market order will fill

        if fill_prob < self.min_fill_prob:
            log.warning(
                "Signal BLOCKED: fill_prob %.2f < min %.2f [%s %s]",
                fill_prob, self.min_fill_prob, side.value, symbol,
            )
            return []

        # ── Algorithm selection ───────────────────────────────────────────

        algo_name = select_algo(urgency=urgency, order_size_pct_adv=0.01)
        algo = self._build_algo(algo_name)

        log.info(
            "Executing: %s %s qty=%.4f algo=%s fill_prob=%.2f slippage=%.1fbps",
            side.value, symbol, quantity, algo_name, fill_prob, slippage_bps,
        )

        # ── Submit ────────────────────────────────────────────────────────

        t_start = time.monotonic()
        responses = await algo.execute(
            self.broker, symbol, side, quantity, 
            sl=getattr(signal, "sl", None), 
            tp=getattr(signal, "tp", None)
        )
        latency = (time.monotonic() - t_start) * 1000

        # ── Track orders ──────────────────────────────────────────────────

        orders: list[Order] = []
        for resp in responses:
            order = self.order_manager.create(
                OrderRequest(
                    symbol=symbol, side=side,
                    order_type=OrderType.MARKET, quantity=quantity,
                    client_order_id=resp.client_order_id,
                    sl=getattr(signal, "sl", None),
                    tp=getattr(signal, "tp", None)
                )
            )
            # Advance FSM: CREATED → SENT_TO_BROKER → PENDING before processing
            # broker response so that full_fill / partial_fill transitions are valid.
            if _TRANSITIONS_AVAILABLE:
                try:
                    order.submit()       # CREATED → SENT_TO_BROKER
                    order.acknowledge()  # SENT_TO_BROKER → PENDING
                except Exception:
                    pass  # guard against duplicate calls
            else:
                order.state = OrderStatus.PENDING.value
            order.broker_order_id = resp.broker_order_id
            await self.order_manager.handle_response(order, resp, self.broker)
            orders.append(order)

        # ── Metrics ───────────────────────────────────────────────────────
        orders_submitted.labels(broker=self.broker.name, symbol=symbol).inc(len(responses))
        order_latency_ms.labels(broker=self.broker.name).observe(latency)

        return orders

    def _build_algo(self, name: str) -> BaseAlgo:
        if name == "twap":
            return TWAP(duration_sec=self._twap_duration, slices=self._twap_slices)
        # Default fallback
        return TWAP(duration_sec=self._twap_duration, slices=self._twap_slices)
