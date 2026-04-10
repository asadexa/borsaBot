"""ColabModelAdapter — wraps models saved by train_colab.ipynb.

Colab models are stored as plain dicts:
  {
    'model':     XGBClassifier / LGBMClassifier,
    'feat_cols': [...],          # 15–17 feature names
    'label_map': {-1:0, 0:1, 1:2},
    'label_rmap': {0:-1, 1:0, 2:1},
    'symbol':    'BTC-USD',
    'interval':  '1d',
    'threshold': float | None,  # meta-model only
  }

This module provides:
  ColabPrimaryAdapter  — computes daily OHLCV features, calls predict_side()
  ColabMetaAdapter     — adds primary_pred/primary_conf, calls should_trade()
  ColabFeaturePipeline — builds the shared feature DataFrame from a price Series

Symbol normalisation table is also here so one import fixes everything.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Symbol normalisation — exchange ticker → model slug → model filename slug
# ─────────────────────────────────────────────────────────────────────────────

# Maps broker symbols (Binance: "BTCUSDT") to Colab model slug ("BTC_USD")
SYMBOL_MAP: dict[str, str] = {
    "BTCUSDT":  "BTC_USD",
    "ETHUSDT":  "ETH_USD",
    "BNBUSDT":  "BNB_USD",
    "SOLUSDT":  "SOL_USD",
    "ADAUSDT":  "ADA_USD",
    "XRPUSDT":  "XRP_USD",
    "BTCUSD":   "BTC_USD",
    "ETHUSD":   "ETH_USD",
    "BTC-USD":  "BTC_USD",
    "ETH-USD":  "ETH_USD",
    "BTC/USDT": "BTC_USD",
    "ETH/USDT": "ETH_USD",
}


def broker_to_model_slug(broker_symbol: str) -> str:
    """Convert exchange symbol to model filename slug.

    Examples:
        'BTCUSDT' → 'BTC_USD'
        'BTC-USD' → 'BTC_USD'
        'BTC_USD' → 'BTC_USD'
    """
    if broker_symbol in SYMBOL_MAP:
        return SYMBOL_MAP[broker_symbol]
    # Fallback: already a slug with underscore
    return broker_symbol.replace("-", "_").replace("/", "_")


# ─────────────────────────────────────────────────────────────────────────────
# Feature pipeline that matches train_colab.ipynb
# ─────────────────────────────────────────────────────────────────────────────

class ColabFeaturePipeline:
    """
    Replicates the feature engineering from train_colab.ipynb.

    Inputs:
        prices: pd.Series with DatetimeIndex, daily close prices
        volume: optional pd.Series of daily volumes (same index)

    Output:
        pd.Series with the exact column names saved in 'feat_cols'
    """

    @staticmethod
    def compute(
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        feat_cols: list[str] | None = None,
    ) -> pd.Series:
        """Build a single row of features from price (and optional volume) history."""

        p = prices.astype(float).dropna()
        if len(p) < 25:
            raise ValueError(f"Need ≥25 price bars, got {len(p)}")

        feats: dict[str, float] = {}

        # ── Fractionally differenced close (d=0.4) ───────────────────────
        try:
            from borsabot.features.stationarity import frac_diff_fixed
            fd_series = frac_diff_fixed(p, d=0.4, thres=1e-4)
            # Explicitly catch StopIteration here — it must NOT escape a coroutine
            try:
                fd_val = float(fd_series.iloc[-1]) if not fd_series.empty else 0.0
            except StopIteration:
                fd_val = 0.0
            feats["fd_close"] = fd_val
        except Exception:
            feats["fd_close"] = float(p.pct_change().iloc[-1])

        # ── Returns ───────────────────────────────────────────────────────
        for lag in [1, 2, 3, 5, 10, 20]:
            key = f"ret_{lag}"
            if len(p) > lag:
                feats[key] = float(p.iloc[-1] / p.iloc[-lag - 1] - 1)
            else:
                feats[key] = 0.0

        # ── Volatility ────────────────────────────────────────────────────
        rets = p.pct_change().dropna()
        feats["vol_24"] = float(rets.tail(24).std()) if len(rets) >= 24 else float(rets.std())
        feats["vol_72"] = float(rets.tail(72).std()) if len(rets) >= 72 else float(rets.std())

        # ── RSI (14) ──────────────────────────────────────────────────────
        feats["rsi_14"] = ColabFeaturePipeline._rsi(p, 14)

        # ── ATR (14) — uses close-to-close as proxy (no H/L available) ───
        feats["atr_14"] = float(rets.abs().tail(14).mean()) if len(rets) >= 14 else float(rets.abs().mean())

        # ── OBV normalised ────────────────────────────────────────────────
        if volume is not None and len(volume) >= 5:
            vol = volume.astype(float).reindex(p.index).fillna(0)
            direction = np.sign(rets.reindex(p.index).fillna(0))
            obv = (direction * vol).cumsum()
            obv_norm_val = float(obv.iloc[-1] / (obv.abs().max() + 1e-9))
        else:
            obv_norm_val = 0.0
        feats["obv_norm"] = obv_norm_val

        # ── Bollinger Band width ──────────────────────────────────────────
        if len(p) >= 20:
            roll = p.tail(20)
            bb_width_val = float((roll.std() * 2) / (roll.mean() + 1e-9))
        else:
            bb_width_val = 0.0
        feats["bb_width"] = bb_width_val

        # ── Close / VWAP ratio ────────────────────────────────────────────
        if volume is not None and len(volume) >= 5:
            vol_s = volume.astype(float).reindex(p.index).fillna(0).tail(20)
            p_s   = p.tail(20)
            vwap  = (p_s * vol_s).sum() / (vol_s.sum() + 1e-9)
            feats["close_vwap"] = float(p.iloc[-1] / (vwap + 1e-9))
        else:
            feats["close_vwap"] = 1.0

        # ── Log volume ────────────────────────────────────────────────────
        if volume is not None and len(volume) >= 1:
            last_vol = float(volume.iloc[-1])
            feats["log_vol"] = float(np.log1p(last_vol))
        else:
            feats["log_vol"] = 0.0

        # ── Build final Series with exact column order ─────────────────────
        if feat_cols:
            # Fill any unexpected missing columns with 0
            return pd.Series({col: feats.get(col, 0.0) for col in feat_cols})
        return pd.Series(feats)

    @staticmethod
    def _rsi(prices: pd.Series, period: int = 14) -> float:
        delta = prices.diff().dropna()
        if len(delta) < period:
            return 50.0
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs   = gain / (loss + 1e-9)
        rsi  = 100 - 100 / (1 + rs)
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Primary Model Adapter
# ─────────────────────────────────────────────────────────────────────────────

class ColabPrimaryAdapter:
    """
    Wraps a Colab-saved primary model dict.

    Interface matches what LiveTrader expects:
        side_int, confidence = model.predict_side(feats_df)
    But also accepts raw price/volume series directly for internal use.
    """

    def __init__(self, model_dict: dict) -> None:
        self._d         = model_dict
        self._model     = model_dict["model"]
        self._feat_cols = model_dict["feat_cols"]
        self._rmap      = model_dict["label_rmap"]   # {0:-1, 1:0, 2:1}

    @classmethod
    def load(cls, path: str | Path) -> "ColabPrimaryAdapter":
        import pickle
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(path, "rb") as f:
                d = pickle.load(f)
        if "model" not in d:
            raise ValueError(f"Expected Colab model dict, got keys: {list(d.keys())}")
        log.info("ColabPrimaryAdapter loaded: %s  feature_count=%d",
                 path, len(d["feat_cols"]))
        return cls(d)

    def predict_side(self, feats_df: pd.DataFrame) -> tuple[int, float]:
        """
        Args:
            feats_df: DataFrame with at least the columns in _feat_cols

        Returns:
            (side_int, confidence)
              side_int ∈ {-1, 0, 1}  (SELL / HOLD / BUY)
              confidence ∈ [0, 1]
        """
        try:
            X = feats_df[self._feat_cols].astype(float)
        except KeyError as e:
            # Feature mismatch — compute features if raw prices available
            raise ValueError(
                f"Feature mismatch. Model needs: {self._feat_cols}\n"
                f"Got: {list(feats_df.columns)}"
            ) from e

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_pred = self._model.predict(X)[0]
            proba    = self._model.predict_proba(X)[0]

        side_int   = self._rmap[int(raw_pred)]
        confidence = float(proba.max())
        return side_int, confidence

    def predict_from_prices(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> tuple[int, float]:
        """Compute features inline and predict. Convenience method."""
        feats  = ColabFeaturePipeline.compute(prices, volume, self._feat_cols)
        X      = feats.to_frame().T
        return self.predict_side(X)

    @property
    def symbol(self) -> str:
        return self._d.get("symbol", "")

    @property
    def feature_names(self) -> list[str]:
        return list(self._feat_cols)


# ─────────────────────────────────────────────────────────────────────────────
# Meta Model Adapter
# ─────────────────────────────────────────────────────────────────────────────

class ColabMetaAdapter:
    """
    Wraps a Colab-saved meta model dict.

    Interface matches what LiveTrader expects:
        do_trade, confidence = model.should_trade(feats_df, side_int, primary_conf)
    """

    def __init__(self, model_dict: dict, threshold: float = 0.55) -> None:
        self._d         = model_dict
        self._model     = model_dict["model"]
        self._feat_cols = model_dict["feat_cols"]   # includes primary_pred + primary_conf
        self.threshold  = threshold

    @classmethod
    def load(
        cls,
        path: str | Path,
        threshold: float = 0.55,
    ) -> "ColabMetaAdapter":
        import pickle
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(path, "rb") as f:
                d = pickle.load(f)
        if "model" not in d:
            raise ValueError(f"Expected Colab model dict, got keys: {list(d.keys())}")
        saved_thr = d.get("threshold")
        effective_thr = saved_thr if saved_thr is not None else threshold
        log.info("ColabMetaAdapter loaded: %s  threshold=%.2f  feature_count=%d",
                 path, effective_thr, len(d["feat_cols"]))
        return cls(d, threshold=effective_thr)

    def should_trade(
        self,
        feats_df: pd.DataFrame,
        primary_side: int,
        primary_conf: float,
    ) -> tuple[bool, float]:
        """
        Args:
            feats_df:     DataFrame with base features (same as primary minus meta-cols)
            primary_side: predicted side (-1/0/1)
            primary_conf: primary model confidence

        Returns:
            (do_trade: bool, meta_confidence: float)
        """
        # Build meta-feature row by appending primary outputs
        base_feats = [c for c in self._feat_cols
                      if c not in ("primary_pred", "primary_conf")]
        try:
            X = feats_df[base_feats].copy()
        except KeyError:
            # Best-effort: fill missing with 0
            X = pd.DataFrame(
                {c: feats_df.get(c, pd.Series([0.0])) for c in base_feats}
            )

        X["primary_pred"] = float(primary_side)
        X["primary_conf"] = float(primary_conf)

        # Re-order to exact training column order
        X = X[self._feat_cols].astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = self._model.predict_proba(X)[0]

        # Meta model predicts "did primary signal lead to profit?" (binary)
        # Class 1 = "yes, trade it"
        meta_conf = float(proba[1]) if proba.shape[0] >= 2 else float(proba.max())
        do_trade  = meta_conf >= self.threshold
        return do_trade, meta_conf
