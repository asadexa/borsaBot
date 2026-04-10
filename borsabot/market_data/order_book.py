"""L2 Order Book engine with snapshot/delta updates and microstructure features."""

from __future__ import annotations

from sortedcontainers import SortedDict


class OrderBook:
    """
    Maintains a live L2 order book from snapshot + incremental delta updates.

    Compatible with Binance depth stream format (20-level, 100ms updates).
    The book uses SortedDict for O(log n) insertions and O(1) best-price access.
    """

    def __init__(self, symbol: str, depth: int = 20) -> None:
        self.symbol = symbol
        self.depth = depth
        # Bids: descending by price (best bid = first key)
        self.bids: SortedDict = SortedDict(lambda k: -k)
        # Asks: ascending by price (best ask = first key)
        self.asks: SortedDict = SortedDict()
        self.last_update_id: int = 0
        self.received_at_ns: int = 0

    # ── Update methods ────────────────────────────────────────────────────

    def apply_snapshot(self, snapshot: dict) -> None:
        """Replace full book state with a REST snapshot."""
        self.bids.clear()
        self.asks.clear()
        for price, qty in snapshot.get("bids", []):
            p, q = float(price), float(qty)
            if q > 0:
                self.bids[p] = q
        for price, qty in snapshot.get("asks", []):
            p, q = float(price), float(qty)
            if q > 0:
                self.asks[p] = q
        self.last_update_id = snapshot.get("lastUpdateId", 0)
        self._trim()

    def apply_delta(self, delta: dict) -> bool:
        """
        Apply an incremental update (Binance depthUpdate event).

        Returns False if delta is stale and should be discarded.
        """
        u = delta.get("u", 0)          # last update ID in this event
        U = delta.get("U", 0)          # first update ID in this event

        # Discard stale deltas
        if u <= self.last_update_id:
            return False

        for price, qty in delta.get("b", []):
            p, q = float(price), float(qty)
            if q == 0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = q

        for price, qty in delta.get("a", []):
            p, q = float(price), float(qty)
            if q == 0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = q

        self.last_update_id = u
        self._trim()
        return True

    def _trim(self) -> None:
        """Keep only top `depth` levels on each side."""
        while len(self.bids) > self.depth:
            self.bids.popitem(-1)
        while len(self.asks) > self.depth:
            self.asks.popitem(-1)

    # ── State checks ──────────────────────────────────────────────────────

    def is_valid(self) -> bool:
        """Book is valid when both sides have at least one level and crossed."""
        if not self.bids or not self.asks:
            return False
        return self.best_bid() < self.best_ask()

    # ── Best prices ───────────────────────────────────────────────────────

    def best_bid(self) -> float:
        return next(iter(self.bids))

    def best_ask(self) -> float:
        return next(iter(self.asks))

    def best_bid_qty(self) -> float:
        return self.bids[self.best_bid()]

    def best_ask_qty(self) -> float:
        return self.asks[self.best_ask()]

    # ── Derived prices ────────────────────────────────────────────────────

    def mid_price(self) -> float:
        return (self.best_bid() + self.best_ask()) / 2

    def spread(self) -> float:
        return self.best_ask() - self.best_bid()

    def spread_bps(self) -> float:
        mid = self.mid_price()
        return self.spread() / mid * 10_000 if mid > 0 else 0.0

    def microprice(self) -> float:
        """
        Microprice: volume-weighted mid price.

        Weights mid toward the side with more liquidity at best level.
        More predictive of short-term price direction than simple mid.
        """
        bb, bq = self.best_bid(), self.best_bid_qty()
        ba, aq = self.best_ask(), self.best_ask_qty()
        total = bq + aq
        if total == 0:
            return self.mid_price()
        return (bb * aq + ba * bq) / total

    # ── Microstructure features ───────────────────────────────────────────

    def order_book_imbalance(self, levels: int = 5) -> float:
        """
        OBI: (bid_volume - ask_volume) / (bid_volume + ask_volume) ∈ [-1, 1].

        Positive → buy pressure, Negative → sell pressure.
        """
        bid_vol = sum(list(self.bids.values())[:levels])
        ask_vol = sum(list(self.asks.values())[:levels])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    def depth_distribution(self, levels: int = 10) -> dict:
        """Return cumulative depth at each price level."""
        bid_levels = list(self.bids.items())[:levels]
        ask_levels = list(self.asks.items())[:levels]
        return {
            "bids": [(p, q) for p, q in bid_levels],
            "asks": [(p, q) for p, q in ask_levels],
        }

    def liquidity_imbalance(self, levels: int = 5) -> float:
        """
        Alternative liquidity imbalance using dollar-value weighted depth.
        """
        bid_liq = sum(p * q for p, q in list(self.bids.items())[:levels])
        ask_liq = sum(p * q for p, q in list(self.asks.items())[:levels])
        total = bid_liq + ask_liq
        return (bid_liq - ask_liq) / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Serialize book state for Redis caching."""
        return {
            "symbol": self.symbol,
            "last_update_id": self.last_update_id,
            "bids": list(self.bids.items())[:self.depth],
            "asks": list(self.asks.items())[:self.depth],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OrderBook":
        """Deserialize from Redis cache."""
        book = cls(symbol=data["symbol"])
        book.apply_snapshot({
            "lastUpdateId": data["last_update_id"],
            "bids": data["bids"],
            "asks": data["asks"],
        })
        return book
