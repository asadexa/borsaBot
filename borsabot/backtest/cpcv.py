"""Combinatorial Purged Cross-Validation (CPCV).

Reference: Marcos López de Prado — Advances in Financial Machine Learning, Ch.12

Problem with standard k-fold CV on financial data:
  1. Overlapping samples (from triple barrier) cause data leakage
  2. Walk-forward only tests N-1 paths — overestimates consistency

CPCV Solution:
  1. Split data into N groups
  2. Train on N-k groups, test on k groups — C(N,k) unique test paths
  3. PURGE labels whose evaluation period overlaps training samples
  4. EMBARGO a gap of h samples after each training period

Example:
    N=6 groups, k=2 test groups → C(6,2) = 15 test paths
    Standard WFO: only 5 test paths

Usage:
    cpcv = CPCV(n_groups=6, n_test_groups=2, embargo_pct=0.01)
    for train_idx, test_idx in cpcv.split(X, events):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
"""

from __future__ import annotations

import itertools
import logging
from typing import Iterator

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class CPCV:
    """
    Combinatorial Purged Cross-Validation splitter.

    Produces C(n_groups, n_test_groups) unique (train, test) splits,
    each purged and embargoed to prevent look-ahead bias.

    Args:
        n_groups:       Number of groups to split data into (N)
        n_test_groups:  Number of groups used as test in each split (k)
        embargo_pct:    Fraction of total samples to embargo after each
                        training period end (default 0.01 = 1%)
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups must be < n_groups")
        self.n_groups       = n_groups
        self.n_test_groups  = n_test_groups
        self.embargo_pct    = embargo_pct

    @property
    def n_paths(self) -> int:
        """Total number of unique test paths = C(N, k)."""
        from math import comb
        return comb(self.n_groups, self.n_test_groups)

    # ── Core split generator ───────────────────────────────────────────────

    def split(
        self,
        X: pd.DataFrame,
        events: pd.DataFrame,       # columns: [t1, trgt, side] from labeling.get_events
        y: pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_indices, test_indices) for each CPCV fold.

        Args:
            X:      Feature matrix indexed by event timestamps
            events: Event dataframe with 't1' column (label end times)
            y:      Labels (unused directly, here for sklearn compat.)

        Yields:
            (train_idx, test_idx) as numpy integer arrays into X.index
        """
        n = len(X)
        embargo_n = max(1, int(n * self.embargo_pct))

        # Divide index into N roughly equal groups
        groups = self._make_groups(X.index, self.n_groups)

        # All C(N, k) combinations of test groups
        for test_group_ids in itertools.combinations(range(self.n_groups), self.n_test_groups):
            test_group_ids_set = set(test_group_ids)

            # Build raw test / train index sets
            test_positions:  set[int] = set()
            train_positions: set[int] = set()

            for g_id, group in enumerate(groups):
                pos_set = set(group)
                if g_id in test_group_ids_set:
                    test_positions.update(pos_set)
                else:
                    train_positions.update(pos_set)

            # Purge: remove training samples whose label end time overlaps
            # any test sample's event start time
            train_positions = self._purge(
                train_positions, test_positions, X.index, events
            )

            # Embargo: remove an h-sample buffer after each training block
            train_positions = self._embargo(
                train_positions, test_positions, X.index, embargo_n
            )

            if not train_positions or not test_positions:
                log.warning("Empty split encountered — skipping")
                continue

            yield (
                np.array(sorted(train_positions)),
                np.array(sorted(test_positions)),
            )

    # ── Purging ────────────────────────────────────────────────────────────

    def _purge(
        self,
        train_pos: set[int],
        test_pos:  set[int],
        index: pd.Index,
        events: pd.DataFrame,
    ) -> set[int]:
        """
        Remove training samples whose label evaluation period (t0, t1)
        overlaps with any test sample's observation start (t0).

        A training sample at time t0_train with label end at t1_train
        is purged if t1_train >= t0_test for any test sample t0_test.
        """
        if "t1" not in events.columns:
            return train_pos   # no events provided — skip purging

        test_starts = {index[i] for i in test_pos if i < len(index)}
        if not test_starts:
            return train_pos

        min_test_start = min(test_starts)
        purged = set()

        for pos in train_pos:
            if pos >= len(index):
                continue
            t0 = index[pos]
            if t0 in events.index:
                t1 = events.loc[t0, "t1"]
                if t1 >= min_test_start:
                    purged.add(pos)

        removed = len(purged)
        if removed > 0:
            log.debug("Purged %d training samples", removed)

        return train_pos - purged

    # ── Embargo ────────────────────────────────────────────────────────────

    def _embargo(
        self,
        train_pos: set[int],
        test_pos:  set[int],
        index: pd.Index,
        embargo_n: int,
    ) -> set[int]:
        """
        Remove `embargo_n` training samples immediately after each
        training→test boundary to prevent leakage via micro-features.
        """
        if embargo_n <= 0 or not test_pos:
            return train_pos

        # Find positions just before test blocks begin
        max_n = len(index)
        embargo_zone: set[int] = set()

        for pos in test_pos:
            # embargo the embargo_n train samples immediately preceding test
            for offset in range(1, embargo_n + 1):
                candidate = pos - offset
                if candidate >= 0 and candidate not in test_pos:
                    embargo_zone.add(candidate)

        return train_pos - embargo_zone

    # ── Group builder ──────────────────────────────────────────────────────

    @staticmethod
    def _make_groups(index: pd.Index, n_groups: int) -> list[list[int]]:
        """Split index positions into N roughly equal groups."""
        n = len(index)
        group_size = n // n_groups
        groups: list[list[int]] = []
        start = 0
        for g in range(n_groups):
            end = start + group_size + (1 if g < n % n_groups else 0)
            groups.append(list(range(start, min(end, n))))
            start = end
        return groups


# ─────────────────────────────────────────────────────────────────────────────
# CPCV Backtester — combines CPCV splits with BacktestSimulator
# ─────────────────────────────────────────────────────────────────────────────

class CPCVBacktester:
    """
    Runs a full CPCV backtest of any signal-generating model.

    For each CPCV fold:
      1. Fit model on (purged, embargoed) training set
      2. Generate signals on test set
      3. Simulate trades with fees + slippage
      4. Collect per-path metrics

    Aggregates results across all C(N,k) paths to produce:
      - Distribution of Sharpe ratios (not a single point estimate)
      - Max drawdown across paths
      - Probabilistic Sharpe Ratio (PSR)
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
        fee_bps: float = 5.0,
        slippage_bps: float = 3.0,
    ) -> None:
        self.cpcv = CPCV(n_groups, n_test_groups, embargo_pct)
        self.fee_bps      = fee_bps
        self.slippage_bps = slippage_bps

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prices: pd.Series,
        events: pd.DataFrame,
        model,              # any object with .fit(X, y) and .predict(X) → array of {-1,0,1}
    ) -> dict:
        """
        Run CPCV backtest.

        Args:
            X:      Feature matrix (indexed by event timestamps)
            y:      Labels {-1, 0, 1}
            prices: Close price series (must cover X.index)
            events: Events dataframe with 't1' column
            model:  Fitted or unfitted model with fit() + predict()

        Returns:
            Dict with keys: sharpe_distribution, mean_sharpe, std_sharpe,
                            max_drawdown_paths, psr, all_metrics, equity_paths
        """
        from borsabot.backtest.simulator import BacktestSimulator
        from borsabot.backtest.metrics import sharpe_ratio, max_drawdown

        sim = BacktestSimulator(
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
        )

        all_metrics: list[dict] = []
        equity_paths: list[pd.Series] = []
        sharpe_list:  list[float] = []

        n_paths = self.cpcv.n_paths
        log.info("Starting CPCV backtest: %d paths", n_paths)

        for fold_idx, (train_idx, test_idx) in enumerate(self.cpcv.split(X, events)):
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_te = X.iloc[test_idx]

            # Fit model on this fold's training data
            try:
                model.fit(X_tr, y_tr)
            except Exception as exc:
                log.warning("Fold %d: model fit failed: %s", fold_idx, exc)
                continue

            # Generate signals
            try:
                raw_preds = model.predict(X_te)
                signals   = pd.Series(raw_preds, index=X_te.index)
            except Exception as exc:
                log.warning("Fold %d: predict failed: %s", fold_idx, exc)
                continue

            # Simulate
            test_prices = prices.reindex(X_te.index).ffill()
            result = sim.run(signals, test_prices)

            metrics = result["metrics"]
            all_metrics.append(metrics)
            equity_paths.append(result["equity"])
            sharpe_list.append(metrics["sharpe"])

            log.debug(
                "Fold %d/%d: Sharpe=%.3f MDD=%.1f%%",
                fold_idx + 1, n_paths,
                metrics["sharpe"], metrics["max_drawdown"] * 100,
            )

        if not sharpe_list:
            return {"error": "All folds failed"}

        sharpe_arr = np.array(sharpe_list)

        # Probabilistic Sharpe Ratio: P(true Sharpe > 0)
        from scipy.stats import norm
        sr_mean = sharpe_arr.mean()
        sr_std  = sharpe_arr.std() + 1e-9
        psr = float(norm.cdf(sr_mean / sr_std))

        return {
            "sharpe_distribution":   sharpe_arr.tolist(),
            "mean_sharpe":           float(sr_mean),
            "std_sharpe":            float(sr_std),
            "max_drawdown_paths":    [m["max_drawdown"] for m in all_metrics],
            "psr":                   psr,
            "all_metrics":           all_metrics,
            "equity_paths":          equity_paths,
            "n_paths":               len(sharpe_list),
        }
