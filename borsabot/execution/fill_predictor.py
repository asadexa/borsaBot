"""Fill probability predictor — estimates P(limit order fills within T seconds).

In live trading, uses a LightGBM model trained on historical order placement data.
In backtesting or when no model is available, uses a heuristic based on spread
and order book state.
"""

from __future__ import annotations

import logging

import numpy as np

from borsabot.market_data.order_book import OrderBook

log = logging.getLogger(__name__)


class FillPredictor:
    """
    Predicts the probability that a limit order placed at `order_price`
    will be filled within `horizon_sec` seconds.

    Training data required:
      - Historical limit order placements + whether they filled
      - Features: spread, OBI, distance from microprice, quantity, horizon

    If no trained model is provided, falls back to a rule-based heuristic.
    """

    def __init__(self, model=None) -> None:
        """
        Args:
            model: A fitted classifier with .predict_proba(X) method.
                   If None, uses heuristic estimation.
        """
        self._model = model

    def predict(
        self,
        book: OrderBook,
        order_price: float,
        order_qty: float,
        horizon_sec: int = 10,
    ) -> float:
        """
        Returns P(fill) ∈ [0, 1].

        Args:
            book:         Live order book state
            order_price:  Limit price of our order
            order_qty:    Positive = buy, negative = sell
            horizon_sec:  Time horizon for fill probability
        """
        if self._model is not None:
            return self._model_predict(book, order_price, order_qty, horizon_sec)
        return self._heuristic(book, order_price, order_qty, horizon_sec)

    def _model_predict(
        self,
        book: OrderBook,
        order_price: float,
        order_qty: float,
        horizon_sec: int,
    ) -> float:
        features = np.array([[
            book.spread(),
            book.order_book_imbalance(5),
            book.microprice() - order_price,
            abs(order_qty),
            float(horizon_sec),
        ]])
        try:
            proba = self._model.predict_proba(features)[0]
            return float(proba[1])
        except Exception as exc:
            log.warning("Fill model prediction failed: %s — using heuristic", exc)
            return self._heuristic(book, order_price, order_qty, horizon_sec)

    @staticmethod
    def _heuristic(
        book: OrderBook,
        order_price: float,
        order_qty: float,
        horizon_sec: int,
    ) -> float:
        """
        Rule-based fill probability:
          - Orders at or through best quote → high fill probability
          - Orders far from mid → low fill probability
          - OBI direction alignment → bonus
        """
        mid = book.mid_price()
        spread = book.spread()
        obi = book.order_book_imbalance(5)

        # Distance in spreads from best quote
        if order_qty > 0:   # BUY
            dist = (book.best_ask() - order_price) / (spread + 1e-9)
            obi_bonus = max(0, obi) * 0.15   # buy‐side aligned if OBI > 0
        else:                # SELL
            dist = (order_price - book.best_bid()) / (spread + 1e-9)
            obi_bonus = max(0, -obi) * 0.15  # sell‐side aligned if OBI < 0

        # Base probability decreases with distance from best quote
        base = max(0.0, 1.0 - dist * 0.5)

        # Scale by time horizon (more time = more fill probability)
        horizon_scale = min(1.0, np.log1p(horizon_sec) / np.log1p(60))

        prob = min(1.0, (base + obi_bonus) * horizon_scale)
        return float(prob)
