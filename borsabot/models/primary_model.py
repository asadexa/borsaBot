"""XGBoost primary model wrapper — predicts trade direction (side).

Primary model role in meta-labeling:
  Input:  microstructure + stationarity features
  Output: predicted side ∈ {BUY=1, SELL=-1}

The primary model is deliberately simple and optimized for recall
(capturing as many real moves as possible) rather than precision.
The meta model handles precision via confidence filtering.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    log.warning("xgboost not installed — PrimaryModel will not function.")


DEFAULT_PARAMS = {
    "n_estimators":      300,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "eval_metric":       "logloss",
    "random_state":      42,
    "n_jobs":            -1,
}


class PrimaryModel:
    """
    XGBoost classifier for trade direction prediction.

    Labels: +1 = BUY, -1 = SELL  (from Triple Barrier Method)
    The labels are remapped to [0, 1, 2] internally (XGBoost multiclass).
    """

    LABEL_MAP  = {-1: 0, 0: 1, 1: 2}
    LABEL_RMAP = {0: -1, 1: 0, 2: 1}

    def __init__(self, params: dict | None = None) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is required for PrimaryModel")
        self.model = XGBClassifier(**(params or DEFAULT_PARAMS))
        self._feature_names: list[str] = []
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        eval_set: list | None = None,
    ) -> "PrimaryModel":
        self._feature_names = list(X.columns)
        y_mapped = y.map(self.LABEL_MAP)
        self.model.fit(
            X, y_mapped,
            sample_weight=sample_weight,
            eval_set=[(eval_set[0], eval_set[1].map(self.LABEL_MAP))] if eval_set else None,
            verbose=False,
        )
        self._is_fitted = True
        log.info("PrimaryModel fitted on %d samples, %d features", len(X), X.shape[1])
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns array of {-1, 0, +1}."""
        raw = self.model.predict(X)
        return np.array([self.LABEL_RMAP[int(r)] for r in raw])

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns probability matrix shape (n, 3): [P(SELL), P(FLAT), P(BUY)]."""
        return self.model.predict_proba(X)

    def predict_side(self, X: pd.DataFrame | np.ndarray) -> tuple[int, float]:
        """
        Returns (side, confidence) for a single feature vector.

        side ∈ {-1, 0, 1}, confidence = max class probability.
        """
        proba = self.model.predict_proba(X.reshape(1, -1) if isinstance(X, np.ndarray) else X)
        class_idx = int(np.argmax(proba[0]))
        side = self.LABEL_RMAP[class_idx]
        confidence = float(proba[0][class_idx])
        return side, confidence

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("PrimaryModel saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "PrimaryModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info("PrimaryModel loaded from %s", path)
        return obj
