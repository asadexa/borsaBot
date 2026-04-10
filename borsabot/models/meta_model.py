"""LightGBM meta model — filters primary signals and predicts trade sizing.

Meta-labeling architecture (López de Prado, Ch.10):
  1. Primary model predicts the direction (side ∈ {-1, 0, +1})
  2. Meta model predicts CONFIDENCE that the primary prediction is correct
  3. If confidence < threshold → skip trade; if above → size proportionally

Meta-model input adds primary model output (side, confidence) to the
original feature vector, allowing the meta model to learn which
predictions from the primary model are trustworthy.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    from lightgbm import LGBMClassifier
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    log.warning("lightgbm not installed — MetaModel will not function.")


DEFAULT_PARAMS = {
    "n_estimators":  300,
    "num_leaves":    31,
    "learning_rate": 0.05,
    "subsample":     0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha":     0.1,
    "reg_lambda":    1.0,
    "random_state":  42,
    "n_jobs":        -1,
    "verbose":       -1,
}


class MetaModel:
    """
    LightGBM binary classifier: P(primary model is correct | features, primary_output).

    Target: meta_label ∈ {0, 1}
        1 → primary model predicted correctly (trade is profitable)
        0 → primary model predicted incorrectly (skip trade)

    Position size = base_size × meta_probability
    """

    DEFAULT_THRESHOLD = 0.55   # minimum confidence to trade

    def __init__(self, params: dict | None = None, threshold: float = DEFAULT_THRESHOLD) -> None:
        if not _LGB_AVAILABLE:
            raise ImportError("lightgbm is required for MetaModel")
        self.model     = LGBMClassifier(**(params or DEFAULT_PARAMS))
        self.threshold = threshold
        self._feature_names: list[str] = []
        self._is_fitted = False

    def _augment_features(
        self,
        X: pd.DataFrame,
        primary_side: np.ndarray | None = None,
        primary_conf: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Append primary model outputs to feature matrix."""
        X = X.copy()
        if primary_side is not None:
            X["primary_side"]        = primary_side
        if primary_conf is not None:
            X["primary_confidence"] = primary_conf
        return X

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,                         # meta labels: {0, 1}
        primary_side: np.ndarray | None = None,
        primary_conf: np.ndarray | None = None,
        sample_weight: pd.Series | None = None,
    ) -> "MetaModel":
        X_aug = self._augment_features(X, primary_side, primary_conf)
        self._feature_names = list(X_aug.columns)
        self.model.fit(X_aug, y, sample_weight=sample_weight)
        self._is_fitted = True
        log.info("MetaModel fitted on %d samples", len(X))
        return self

    def predict_confidence(
        self,
        X: pd.DataFrame | np.ndarray,
        primary_side: int | None = None,
        primary_conf: float | None = None,
    ) -> float:
        """
        Return P(trade is profitable) for a single feature vector.

        Args:
            X:             Feature vector (1 row)
            primary_side:  Side predicted by primary model {-1, 0, +1}
            primary_conf:  Confidence from primary model ∈ [0,1]

        Returns:
            float ∈ [0, 1] — meta-model confidence
        """
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame([X], columns=self._feature_names[:len(X)])
        else:
            X_df = X.copy()

        if primary_side is not None:
            X_df["primary_side"] = primary_side
        if primary_conf is not None:
            X_df["primary_confidence"] = primary_conf

        proba = self.model.predict_proba(X_df)[0]
        return float(proba[1])   # P(label=1 = trade is correct)

    def should_trade(
        self,
        X: pd.DataFrame | np.ndarray,
        primary_side: int | None = None,
        primary_conf: float | None = None,
    ) -> tuple[bool, float]:
        """
        Returns (trade_decision, meta_confidence).

        Example:
            trade, conf = meta.should_trade(features, side=1, conf=0.7)
            if trade:
                size = base_size * conf
        """
        confidence = self.predict_confidence(X, primary_side, primary_conf)
        return confidence >= self.threshold, confidence

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "MetaModel":
        with open(path, "rb") as f:
            return pickle.load(f)
