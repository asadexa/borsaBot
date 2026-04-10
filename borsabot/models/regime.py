"""Hidden Markov Model based market regime detector.

Markets cycle through distinct states (regimes):
  0 = Low volatility / ranging (mean-reversion strategies work well)
  1 = Trending       (momentum strategies work well)
  2 = High volatility / crisis (reduce exposure, widen stops)

The detector adapts strategy parameters and position sizes
according to the current inferred regime.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    log.warning("hmmlearn not installed — RegimeDetector will not function.")


# Regime-specific trading parameters
REGIME_PARAMS: dict[str, dict] = {
    "low_vol": {
        "strategy":       "mean_reversion",
        "position_pct":   0.10,    # 10% of NAV
        "stop_mult":      1.5,
        "profit_mult":    1.5,
        "description":    "Low volatility ranging market",
    },
    "trending": {
        "strategy":       "momentum",
        "position_pct":   0.15,    # 15% of NAV
        "stop_mult":      2.0,
        "profit_mult":    3.0,
        "description":    "Trending market with directional moves",
    },
    "high_vol": {
        "strategy":       "reduce_exposure",
        "position_pct":   0.05,    # 5% of NAV — defensive
        "stop_mult":      1.0,
        "profit_mult":    1.0,
        "description":    "High volatility / crisis — reduce exposure",
    },
}


class RegimeDetector:
    """
    Gaussian HMM with N=3 hidden states fitted on return + volatility observations.

    The regime label mapping (state → name) is learned post-fitting
    by ranking states by their mean volatility.
    """

    N_STATES = 3

    def __init__(self) -> None:
        if not _HMM_AVAILABLE:
            raise ImportError("hmmlearn is required for RegimeDetector")
        self.model = GaussianHMM(
            n_components=self.N_STATES,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self._state_to_regime: dict[int, str] = {}
        self._is_fitted = False

    def _build_obs(self, returns: np.ndarray) -> np.ndarray:
        """3-column observation matrix: return, |return|, sign(return)."""
        return np.column_stack([
            returns,
            np.abs(returns),
            np.sign(returns),
        ])

    def fit(self, returns: pd.Series | np.ndarray) -> "RegimeDetector":
        """
        Fit the HMM on historical returns.
        Post-fits a mapping from HMM state → regime name by ranking
        states by mean absolute return (volatility proxy).
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        obs = self._build_obs(returns)
        self.model.fit(obs)

        # Rank states by mean absolute return (volatility proxy)
        states = self.model.predict(obs)
        mean_vol = {s: float(np.mean(np.abs(returns[states == s]))) for s in range(self.N_STATES)}
        sorted_states = sorted(mean_vol, key=mean_vol.get)  # type: ignore[arg-type]

        self._state_to_regime = {
            sorted_states[0]: "low_vol",
            sorted_states[1]: "trending",
            sorted_states[2]: "high_vol",
        }
        self._is_fitted = True
        log.info(
            "RegimeDetector fitted — state mapping: %s",
            {f"state{k}": v for k, v in self._state_to_regime.items()},
        )
        return self

    def predict_regime(self, returns: pd.Series | np.ndarray) -> str:
        """Return the current regime name for the most recent observation window."""
        assert self._is_fitted, "Call .fit() first"
        if isinstance(returns, pd.Series):
            returns = returns.values
        obs = self._build_obs(returns)
        state = int(self.model.predict(obs)[-1])
        return self._state_to_regime.get(state, "low_vol")

    def predict_regime_series(self, returns: pd.Series) -> pd.Series:
        """Return per-timestep regime labels for a return series (backtesting use)."""
        assert self._is_fitted
        obs = self._build_obs(returns.values)
        states = self.model.predict(obs)
        return pd.Series(
            [self._state_to_regime.get(s, "low_vol") for s in states],
            index=returns.index,
            name="regime",
        )

    def get_params(self, regime: str) -> dict:
        """Return strategy parameters for a given regime."""
        return REGIME_PARAMS.get(regime, REGIME_PARAMS["low_vol"])

    def current_params(self, returns: pd.Series | np.ndarray) -> dict:
        """Shortcut: predict regime + return its parameters."""
        regime = self.predict_regime(returns)
        return {**self.get_params(regime), "regime": regime}

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeDetector":
        with open(path, "rb") as f:
            return pickle.load(f)
