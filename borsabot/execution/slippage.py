"""Slippage estimator based on Almgren-Chriss market impact model."""

from __future__ import annotations

from borsabot.market_data.order_book import OrderBook


class SlippageEstimator:
    """
    Estimates expected slippage for a given order size using a
    simplified Almgren-Chriss market impact model.

    Impact ∝ order_qty / available_liquidity (participation rate)

    Provides:
      - Expected slippage in basis points
      - Adjusted execution price
    """

    def __init__(self, eta: float = 0.01) -> None:
        """
        Args:
            eta: Market impact coefficient.
                 Typical range: 0.005 (liquid) – 0.05 (illiquid)
        """
        self.eta = eta

    def estimate_bps(
        self,
        order_qty: float,
        book: OrderBook,
        levels: int = 5,
    ) -> float:
        """
        Estimate slippage in basis points.

        Args:
            order_qty: Signed quantity (+ve = BUY, -ve = SELL)
            book:      Live order book
            levels:    Depth levels to consider for liquidity

        Returns:
            Estimated slippage in basis points (always positive)
        """
        # Guard: empty book produces near-zero liquidity → infinite slippage
        if not book.is_valid():
            return 0.0  # no book data; slippage unknown, skip check
        side_book = book.asks if order_qty > 0 else book.bids
        liquidity = sum(list(side_book.values())[:levels])
        if liquidity <= 0:
            return 0.0
        participation_rate = abs(order_qty) / liquidity
        return self.eta * participation_rate * 10_000

    def adjusted_price(
        self,
        order_qty: float,
        book: OrderBook,
        levels: int = 5,
    ) -> float:
        """
        Return the expected fill price after slippage.

        For BUY orders: price > best_ask
        For SELL orders: price < best_bid
        """
        if not book.is_valid():
            return 0.0  # caller should skip execution without a valid book
        slippage_bps = self.estimate_bps(order_qty, book, levels)
        if order_qty > 0:
            return book.best_ask() * (1 + slippage_bps / 10_000)
        else:
            return book.best_bid() * (1 - slippage_bps / 10_000)

    def walk_book(self, order_qty: float, book: OrderBook) -> float:
        """
        Simulate walking the book — compute actual average fill price
        by consuming liquidity level by level.

        Returns average fill price (most realistic for large orders).
        Returns 0.0 if the book is empty (caller should handle).
        """
        if not book.is_valid():
            return 0.0
        remaining = abs(order_qty)
        side_book = book.asks if order_qty > 0 else book.bids
        total_cost = 0.0
        total_filled = 0.0

        for price, qty in side_book.items():
            fill = min(remaining, qty)
            total_cost   += fill * price
            total_filled += fill
            remaining    -= fill
            if remaining <= 0:
                break

        if total_filled == 0:
            try:
                return book.mid_price()
            except (StopIteration, RuntimeError):
                return 0.0
        return total_cost / total_filled
