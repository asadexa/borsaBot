"""FeatureBuilder — orchestrates the full feature engineering pipeline.

Combines stationarity + volatility normalization + microstructure features
into a single feature vector ready for AI model input.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from borsabot.features.frac_diff import frac_diff
from borsabot.features.microstructure import extract_all
from borsabot.features.volatility import rolling_zscore, volatility_scaled
from borsabot.market_data.order_book import OrderBook

log = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Transforms a price tick history + live order book → feature vector.

    Features produced:
      Stationarity group:    frac_return, log_price_fd
      Volatility group:      vol_scaled_ret, zscore_price, zscore_ret
      Microstructure group:  obi_1/5/10, liq_imbalance_5, microprice,
                             spread_abs, spread_bps, mid_price,
                             best_bid, best_ask, bid/ask depth, depth_ratio
    """

    def __init__(
        self,
        frac_d: float = 0.4,
        vol_window: int = 20,
        zscore_window: int = 20,
    ) -> None:
        self.d            = frac_d
        self.vol_window   = vol_window
        self.zscore_window = zscore_window

    # ── Live feature computation (single tick) ────────────────────────────

    def build(
        self,
        tick_history: pd.DataFrame,           # columns: [time, price, qty, side]
        book: OrderBook,
        trades: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Build a feature vector from the latest tick history + live book.

        Args:
            tick_history: Recent ticks (at least vol_window rows recommended)
            book:         Live order book state
            trades:       Recent trade stream for TFI calculation (optional)

        Returns:
            Feature Series with named float values, no NaN/Inf.
        """
        prices = tick_history["price"].astype(float)
        log_prices = np.log(prices + 1e-9)
        returns = prices.pct_change().fillna(0.0)

        features: dict[str, float] = {}

        # ── Stationarity ─────────────────────────────────────────────────
        fd = frac_diff(log_prices, self.d).dropna()
        features["frac_return"]  = float(fd.iloc[-1]) if len(fd) > 0 else 0.0
        features["log_price_fd"] = features["frac_return"]

        # ── Volatility normalization ──────────────────────────────────────
        vsr = volatility_scaled(returns, self.vol_window)
        features["vol_scaled_ret"] = float(vsr.iloc[-1]) if not vsr.empty else 0.0

        zp = rolling_zscore(prices, self.zscore_window)
        features["zscore_price"] = float(zp.iloc[-1]) if not zp.empty else 0.0

        zr = rolling_zscore(returns, self.zscore_window)
        features["zscore_ret"] = float(zr.iloc[-1]) if not zr.empty else 0.0

        # ── Microstructure ────────────────────────────────────────────────
        ms = extract_all(book, trades)
        features.update(ms)

        # ── Sanitise: replace NaN/Inf with 0 ─────────────────────────────
        clean = {
            k: (0.0 if (v != v or np.isinf(v)) else v)
            for k, v in features.items()
        }

        return pd.Series(clean, dtype=float)

    # ── Batch feature computation (for backtesting / training) ────────────

    def build_batch(self, ohlcv: pd.DataFrame, min_rows: int = 30) -> pd.DataFrame:
        """
        Build features for every row in an OHLCV dataframe.

        Uses a rolling window approach — rows before min_rows are dropped.
        For backtesting/training, not for live execution.

        ohlcv: DataFrame with columns [open, high, low, close, volume]
        """
        prices  = ohlcv["close"].astype(float)
        returns = prices.pct_change().fillna(0.0)

        rows = []

        fd_series  = frac_diff(np.log(prices + 1e-9), self.d).reindex(prices.index)
        vsr_series = volatility_scaled(returns, self.vol_window)
        zp_series  = rolling_zscore(prices, self.zscore_window)
        zr_series  = rolling_zscore(returns, self.zscore_window)

        for i in range(min_rows, len(ohlcv)):
            row: dict[str, float] = {
                "frac_return":   float(fd_series.iloc[i])  if not pd.isna(fd_series.iloc[i])  else 0.0,
                "vol_scaled_ret": float(vsr_series.iloc[i]) if not pd.isna(vsr_series.iloc[i]) else 0.0,
                "zscore_price":  float(zp_series.iloc[i])  if not pd.isna(zp_series.iloc[i])  else 0.0,
                "zscore_ret":    float(zr_series.iloc[i])   if not pd.isna(zr_series.iloc[i])  else 0.0,
                # Approximate spread from OHLCV (no order book available in batch mode)
                "spread_bps":   float((ohlcv["high"].iloc[i] - ohlcv["low"].iloc[i])
                                      / ohlcv["close"].iloc[i] * 1e4),
                "mid_price":    float(ohlcv["close"].iloc[i]),
            }
            rows.append(row)

        df = pd.DataFrame(rows, index=ohlcv.index[min_rows:])
        return df.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    @property
    def feature_names(self) -> list[str]:
        """List of feature names produced by build(). Useful for model training."""
        return [
            "frac_return", "log_price_fd",
            "vol_scaled_ret", "zscore_price", "zscore_ret",
            "obi_1", "obi_5", "obi_10",
            "liq_imbalance_5",
            "microprice", "spread_abs", "spread_bps",
            "mid_price", "best_bid", "best_ask",
            "bid_depth_total", "ask_depth_total", "depth_ratio",
        ]
