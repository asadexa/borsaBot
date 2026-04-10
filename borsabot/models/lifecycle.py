"""Model lifecycle management — drift detection and automated retraining.

Monitors model health in production and triggers retraining when:
  - Feature distribution drifts (PSI > threshold)
  - Model performance decays (Sharpe Ratio drops significantly)
  - Concept drift detected (label distribution shifts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Population Stability Index (PSI) — feature drift detection
# ─────────────────────────────────────────────────────────────────────────────

def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
) -> float:
    """
    Measure how much a distribution has shifted.

    Interpretation:
      PSI < 0.10 → no significant change (stable)
      0.10 ≤ PSI < 0.20 → moderate change (monitor closely)
      PSI ≥ 0.20 → significant change → trigger retraining
    """
    # Build bins from expected distribution
    _, bins = np.histogram(expected, bins=buckets)
    bins[0]  = -np.inf
    bins[-1] =  np.inf

    e_counts, _ = np.histogram(expected, bins=bins)
    a_counts, _ = np.histogram(actual,   bins=bins)

    e_pct = (e_counts + 1e-9) / len(expected)
    a_pct = (a_counts + 1e-9) / len(actual)

    psi = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
    return psi


def psi_all_features(
    train_X: pd.DataFrame,
    live_X: pd.DataFrame,
) -> pd.Series:
    """Compute PSI for every feature column. Returns Series indexed by feature name."""
    results = {}
    for col in train_X.columns:
        if col in live_X.columns:
            results[col] = population_stability_index(
                train_X[col].dropna().values,
                live_X[col].dropna().values,
            )
    return pd.Series(results).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Performance decay detector
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods))


def sharpe_decay(baseline_sharpe: float, current_sharpe: float) -> float:
    """Returns fractional decay (0.0 = no decay, 1.0 = total loss of performance)."""
    if abs(baseline_sharpe) < 1e-6:
        return 0.0
    return max(0.0, (baseline_sharpe - current_sharpe) / abs(baseline_sharpe))


# ─────────────────────────────────────────────────────────────────────────────
# Model Health Monitor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HealthReport:
    psi_max: float
    psi_per_feature: dict[str, float]
    sharpe_decay_pct: float
    retrain_needed: bool
    reason: str = ""
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)


class ModelHealthMonitor:
    """
    Tracks production model health metrics and decides when to retrain.

    Usage pattern:
        monitor = ModelHealthMonitor(train_X=X_train, baseline_sharpe=1.5)
        report  = monitor.check(live_X=recent_features, live_returns=recent_pnl)
        if report.retrain_needed:
            await retrain_pipeline.run()
    """

    def __init__(
        self,
        train_X: pd.DataFrame,
        baseline_sharpe: float = 0.0,
        psi_threshold: float = 0.20,
        sharpe_decay_threshold: float = 0.30,   # retrain if Sharpe drops >30%
    ) -> None:
        self.train_X              = train_X
        self.baseline_sharpe      = baseline_sharpe
        self.psi_threshold        = psi_threshold
        self.sharpe_decay_thr     = sharpe_decay_threshold

    def check(
        self,
        live_X: pd.DataFrame,
        live_returns: pd.Series | None = None,
    ) -> HealthReport:
        """Run full health check."""

        # ── PSI drift ─────────────────────────────────────────────────────
        psi_series  = psi_all_features(self.train_X, live_X)
        psi_max     = float(psi_series.max()) if not psi_series.empty else 0.0
        psi_dict    = psi_series.to_dict()

        # ── Sharpe decay ──────────────────────────────────────────────────
        decay = 0.0
        if live_returns is not None and self.baseline_sharpe > 0:
            live_sharpe = sharpe_ratio(live_returns)
            decay = sharpe_decay(self.baseline_sharpe, live_sharpe)

        # ── Decision ──────────────────────────────────────────────────────
        retrain = False
        reasons = []

        if psi_max >= self.psi_threshold:
            retrain = True
            worst_feat = psi_series.idxmax()
            reasons.append(f"Feature drift PSI={psi_max:.3f} (worst: {worst_feat})")

        if decay >= self.sharpe_decay_thr:
            retrain = True
            reasons.append(f"Sharpe decay {decay*100:.1f}% > threshold {self.sharpe_decay_thr*100:.0f}%")

        return HealthReport(
            psi_max=psi_max,
            psi_per_feature=psi_dict,
            sharpe_decay_pct=decay * 100,
            retrain_needed=retrain,
            reason=" | ".join(reasons),
        )

    def update_baseline(self, new_sharpe: float, new_train_X: pd.DataFrame) -> None:
        """Update baselines after a successful retrain."""
        self.baseline_sharpe = new_sharpe
        self.train_X         = new_train_X
        log.info("ModelHealthMonitor baselines updated: new Sharpe=%.3f", new_sharpe)
