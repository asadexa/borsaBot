"""TWAP, VWAP, and POV smart order execution algorithms."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import numpy as np

from borsabot.brokers.base import BrokerGateway
from borsabot.core.events import OrderRequest, OrderResponse, OrderSide, OrderType

log = logging.getLogger(__name__)


class BaseAlgo(ABC):
    """Abstract execution algorithm."""

    @abstractmethod
    async def execute(
        self,
        broker: BrokerGateway,
        symbol: str,
        side: OrderSide,
        total_qty: float,
        sl: float | None = None,
        tp: float | None = None,
    ) -> list[OrderResponse]:
        """Execute the algorithm. Returns list of order responses."""


# ─────────────────────────────────────────────────────────────────────────────
# TWAP — Time Weighted Average Price
# ─────────────────────────────────────────────────────────────────────────────

class TWAP(BaseAlgo):
    """
    Splits total_qty into `slices` equal market orders spaced `interval_sec` apart.

    Best for: large orders that must complete within a defined time window
    without regard for real-time volume.
    """

    def __init__(self, duration_sec: int, slices: int = 10) -> None:
        self.duration_sec = duration_sec
        self.slices       = slices
        self.interval_sec = duration_sec / slices

    async def execute(
        self,
        broker: BrokerGateway,
        symbol: str,
        side: OrderSide,
        total_qty: float,
        sl: float | None = None,
        tp: float | None = None,
    ) -> list[OrderResponse]:
        slice_qty = total_qty / self.slices
        responses: list[OrderResponse] = []

        log.info(
            "TWAP started: %s %s qty=%.4f in %d slices over %ds",
            side.value, symbol, total_qty, self.slices, self.duration_sec,
        )

        for i in range(self.slices):
            req = OrderRequest(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=slice_qty,
                sl=sl,
                tp=tp,
                client_order_id=f"twap_{symbol}_{i}",
            )
            resp = await broker.submit_order(req)
            responses.append(resp)
            log.debug("TWAP slice %d/%d submitted: %s", i + 1, self.slices, resp.status)

            if i < self.slices - 1:
                await asyncio.sleep(self.interval_sec)

        return responses


# ─────────────────────────────────────────────────────────────────────────────
# VWAP — Volume Weighted Average Price
# ─────────────────────────────────────────────────────────────────────────────

class VWAP(BaseAlgo):
    """
    Schedules order flow proportional to a historical intraday volume profile.

    Best for: minimizing market impact relative to the VWAP benchmark.
    Allocates more size during historically high-volume periods.
    """

    def __init__(
        self,
        volume_profile: np.ndarray,    # normalized fractions, shape (N,), must sum to 1
        interval_sec: int = 60,
    ) -> None:
        self.volume_profile = volume_profile / volume_profile.sum()   # normalize
        self.interval_sec   = interval_sec

    async def execute(
        self,
        broker: BrokerGateway,
        symbol: str,
        side: OrderSide,
        total_qty: float,
    ) -> list[OrderResponse]:
        planned = (total_qty * self.volume_profile).tolist()
        responses: list[OrderResponse] = []

        log.info("VWAP started: %s %s qty=%.4f in %d buckets", side.value, symbol, total_qty, len(planned))

        for i, qty in enumerate(planned):
            if qty <= 0:
                await asyncio.sleep(self.interval_sec)
                continue

            req = OrderRequest(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=round(qty, 8),
                client_order_id=f"vwap_{symbol}_{i}",
            )
            resp = await broker.submit_order(req)
            responses.append(resp)
            log.debug("VWAP bucket %d: qty=%.6f status=%s", i, qty, resp.status)

            if i < len(planned) - 1:
                await asyncio.sleep(self.interval_sec)

        return responses


# ─────────────────────────────────────────────────────────────────────────────
# POV — Percentage of Volume
# ─────────────────────────────────────────────────────────────────────────────

class POV(BaseAlgo):
    """
    Participates at a fixed percentage of real-time market volume.

    Best for: minimizing market impact when you want to "blend in"
    with organic volume without committing to a time schedule.
    """

    def __init__(self, pov_rate: float = 0.05) -> None:
        """
        Args:
            pov_rate: Fraction of each trade to participate (e.g. 0.05 = 5%).
        """
        self.pov_rate  = pov_rate
        self._remaining = 0.0

    async def execute(
        self,
        broker: BrokerGateway,
        symbol: str,
        side: OrderSide,
        total_qty: float,
    ) -> list[OrderResponse]:
        # POV is driven by on_trade() callbacks; this method initialises state
        self._remaining = total_qty
        log.info(
            "POV initialised: %s %s qty=%.4f rate=%.1f%%",
            side.value, symbol, total_qty, self.pov_rate * 100,
        )
        return []

    async def on_trade(
        self,
        market_qty: float,
        broker: BrokerGateway,
        symbol: str,
        side: OrderSide,
    ) -> OrderResponse | None:
        """
        Called for each observed market trade.
        Submits our participation slice.
        """
        if self._remaining <= 0:
            return None

        our_qty = min(market_qty * self.pov_rate, self._remaining)
        our_qty = round(our_qty, 8)

        if our_qty <= 0:
            return None

        req = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=our_qty,
        )
        resp = await broker.submit_order(req)
        self._remaining -= our_qty
        log.debug("POV slice submitted qty=%.6f remaining=%.6f", our_qty, self._remaining)
        return resp

    @property
    def remaining(self) -> float:
        return self._remaining

    @property
    def is_complete(self) -> bool:
        return self._remaining <= 0


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm selector
# ─────────────────────────────────────────────────────────────────────────────

def select_algo(
    urgency: str,
    order_size_pct_adv: float,
    has_volume_profile: bool = False,
) -> str:
    """
    Recommend an execution algorithm based on order characteristics.

    Args:
        urgency:              "low" | "medium" | "high"
        order_size_pct_adv:   Order size as % of average daily volume (ADV)
        has_volume_profile:   Whether an intraday volume profile is available

    Returns:
        Algorithm name: "twap" | "vwap" | "pov"
    """
    if urgency == "high":
        return "twap"
    if has_volume_profile and order_size_pct_adv > 0.01:
        return "vwap"
    if urgency == "low":
        return "pov"
    return "twap"
