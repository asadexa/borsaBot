"""Market microstructure feature extraction.

Microstructure features describe the *execution environment* — not just price.
They are critical for execution-aware trading systems.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from borsabot.market_data.order_book import OrderBook


def order_book_imbalance(book: OrderBook, levels: int = 5) -> float:
    """
    OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol) ∈ [-1, 1]

    Positive → excess buy-side interest → upward price pressure
    Negative → excess sell-side interest → downward price pressure
    """
    return book.order_book_imbalance(levels)


def microprice(book: OrderBook) -> float:
    """
    Microprice: volume-weighted average of best bid and ask.

    More predictive of short-term price direction than simple mid price.
    Reference: Stoikov (2018), "The micro-price"
    """
    return book.microprice()


def spread_dynamics(book: OrderBook) -> dict:
    """
    Spread features in pips and basis points.

    Spread widens during low liquidity, stress, and news events.
    """
    return {
        "spread_abs": book.spread(),
        "spread_bps": book.spread_bps(),
        "mid_price":  book.mid_price(),
        "best_bid":   book.best_bid(),
        "best_ask":   book.best_ask(),
    }


def depth_profile(book: OrderBook, levels: int = 10) -> dict:
    """
    Cumulative depth distribution — how much liquidity is available
    within N price levels from the best quote.
    """
    bid_items = list(book.bids.items())[:levels]
    ask_items = list(book.asks.items())[:levels]

    bid_cum = np.cumsum([q for _, q in bid_items])
    ask_cum = np.cumsum([q for _, q in ask_items])

    return {
        "bid_depth_total": float(bid_cum[-1]) if len(bid_cum) > 0 else 0.0,
        "ask_depth_total": float(ask_cum[-1]) if len(ask_cum) > 0 else 0.0,
        "depth_ratio":     float(bid_cum[-1] / (ask_cum[-1] + 1e-9)) if len(bid_cum) > 0 else 1.0,
    }


def trade_flow_imbalance(trades: pd.DataFrame, window: int = 100) -> float:
    """
    Trade flow imbalance from recent trade stream.

    trades: DataFrame with columns ['side', 'qty'] where side ∈ {'B', 'S'}
    TFI = (buy_qty - sell_qty) / (buy_qty + sell_qty) ∈ [-1, 1]

    More granular execution signal than OBI (uses actual trades not resting orders).
    """
    recent = trades.tail(window)
    buy_vol  = recent.loc[recent["side"] == "B", "qty"].sum()
    sell_vol = recent.loc[recent["side"] == "S", "qty"].sum()
    total    = buy_vol + sell_vol
    return float((buy_vol - sell_vol) / (total + 1e-9))


def kyle_lambda(price_changes: pd.Series, signed_volume: pd.Series) -> float:
    """
    Kyle's Lambda — price impact coefficient.

    Estimates how much price moves per unit of signed order flow.
    High lambda → illiquid market, low lambda → liquid market.

    Reference: Kyle (1985), "Continuous Auctions and Insider Trading"
    """
    try:
        from sklearn.linear_model import LinearRegression
        X = signed_volume.values.reshape(-1, 1)
        y = price_changes.values
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        return float(reg.coef_[0])
    except Exception:
        return 0.0


def extract_all(book: OrderBook, trades: pd.DataFrame | None = None) -> dict:
    """
    Extract all microstructure features into a flat dict.
    Suitable for direct input to FeatureBuilder.
    """
    features: dict = {}

    # OBI at multiple levels
    features["obi_1"]  = book.order_book_imbalance(1)
    features["obi_5"]  = book.order_book_imbalance(5)
    features["obi_10"] = book.order_book_imbalance(10)

    # Liquidity imbalance (dollar-weighted)
    features["liq_imbalance_5"] = book.liquidity_imbalance(5)

    # Microprice
    features["microprice"]  = book.microprice()

    # Spread
    sd = spread_dynamics(book)
    features.update(sd)

    # Depth
    dp = depth_profile(book)
    features.update(dp)

    # Trade flow (optional)
    if trades is not None and not trades.empty:
        features["tfi_100"] = trade_flow_imbalance(trades, 100)

    return features
