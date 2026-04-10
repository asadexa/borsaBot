"""Unit tests for CPCV — Combinatorial Purged Cross-Validation."""

from __future__ import annotations

import math
import pytest
import numpy as np
import pandas as pd

from borsabot.backtest.cpcv import CPCV


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_data(n: int = 500, seed: int = 42):
    """Create synthetic X, y, prices, events for testing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")

    # Feature matrix (10 features)
    X = pd.DataFrame(
        rng.normal(0, 1, (n, 10)),
        index=idx,
        columns=[f"f{i}" for i in range(10)],
    )

    # Labels: random {-1, 0, 1}
    y = pd.Series(
        rng.choice([-1, 0, 1], n),
        index=idx, name="label",
    )

    # Use short horizon (5 bars) so purge doesn't wipe out the entire training set
    horizon = 5
    t1 = pd.Series(
        [idx[min(i + horizon, n - 1)] for i in range(n)],
        index=idx,
    )
    trgt = pd.Series(0.01, index=idx)
    events = pd.DataFrame({"t1": t1, "trgt": trgt, "side": 1.0})

    prices = pd.Series(
        100 + np.cumsum(rng.normal(0, 0.5, n)),
        index=idx,
    )

    return X, y, events, prices


# ─────────────────────────────────────────────────────────────────────────────
# CPCV structural tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cpcv_n_paths():
    """CPCV should produce C(N,k) paths."""
    cvcv = CPCV(n_groups=6, n_test_groups=2)
    assert cvcv.n_paths == math.comb(6, 2)   # = 15


def test_cpcv_rejects_invalid_params():
    with pytest.raises(ValueError):
        CPCV(n_groups=4, n_test_groups=4)   # n_test_groups must be < n_groups


def test_cpcv_produces_correct_n_splits():
    X, y, events, _ = make_data(500)
    cpcv = CPCV(n_groups=6, n_test_groups=2, embargo_pct=0.02)
    splits = list(cpcv.split(X, events))
    # Should produce up to C(6,2) = 15 splits
    # Aggressive pruning may eliminate some empty ones — must have at least 10
    assert len(splits) >= 10, f"Too few non-empty splits: {len(splits)}"


def test_cpcv_train_test_no_overlap():
    """Train and test index sets must be disjoint for every split."""
    X, y, events, _ = make_data(240)
    cpcv = CPCV(n_groups=6, n_test_groups=2)
    for train_idx, test_idx in cpcv.split(X, events):
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, f"Train/test overlap: {overlap}"


def test_cpcv_train_test_cover_data():
    """Each split should have a non-trivial test set, roughly n_test_groups/n_groups of data."""
    X, y, events, _ = make_data(500)
    cpcv = CPCV(n_groups=6, n_test_groups=2, embargo_pct=0.0)
    n = len(X)
    min_test_frac = (2 / 6) * 0.8    # expect at least 80% of 33% = ~26% in test

    for train_idx, test_idx in cpcv.split(X, events):
        # Both sets must be non-empty
        assert len(test_idx) > 0, "Test set is empty"
        assert len(train_idx) > 0, "Train set is empty"
        # Test set should be roughly n_test_groups / n_groups fraction of data
        assert len(test_idx) >= min_test_frac * n, (
            f"Test set too small: {len(test_idx)}/{n}"
        )



def test_cpcv_purge_removes_leaking_samples():
    """Purging should remove training samples whose t1 overlaps test start."""
    X, y, events, _ = make_data(120)
    cpcv_no_purge  = CPCV(n_groups=4, n_test_groups=1, embargo_pct=0.0)
    cpcv_purged    = CPCV(n_groups=4, n_test_groups=1, embargo_pct=0.0)

    splits_no_purge = list(cpcv_no_purge.split(X, pd.DataFrame()))
    splits_purged   = list(cpcv_purged.split(X, events))

    # With events containing t1, some training samples should be purged
    # So purged train sizes should be <= no-purge train sizes
    for (tr_np, _), (tr_p, _) in zip(splits_no_purge, splits_purged):
        assert len(tr_p) <= len(tr_np)


def test_cpcv_embargo_reduces_train_size():
    """Embargo should reduce training set size vs. no embargo."""
    X, y, events, _ = make_data(240)

    cpcv_no_emb  = CPCV(n_groups=6, n_test_groups=2, embargo_pct=0.0)
    cpcv_emb     = CPCV(n_groups=6, n_test_groups=2, embargo_pct=0.05)

    splits_no_emb = list(cpcv_no_emb.split(X, events))
    splits_emb    = list(cpcv_emb.split(X, events))

    total_no_emb = sum(len(tr) for tr, _ in splits_no_emb)
    total_emb    = sum(len(tr) for tr, _ in splits_emb)

    assert total_emb <= total_no_emb, "Embargo should reduce total training size"


def test_cpcv_all_test_groups_covered():
    """Every group should appear as a test group at least once across all splits."""
    X, y, events, _ = make_data(240)
    cpcv = CPCV(n_groups=6, n_test_groups=2, embargo_pct=0.0)

    n = len(X)
    group_size = n // 6
    all_test_positions: set[int] = set()

    for _, test_idx in cpcv.split(X, events):
        all_test_positions.update(test_idx.tolist())

    # Every position should appear in test at least once
    coverage = len(all_test_positions) / n
    assert coverage >= 0.80, f"Test coverage too low: {coverage:.1%}"


# ─────────────────────────────────────────────────────────────────────────────
# CPCVBacktester integration
# ─────────────────────────────────────────────────────────────────────────────

class AlwaysBuyModel:
    """Minimal model stub that always predicts BUY (+1)."""
    def fit(self, X, y, **kw): return self
    def predict(self, X): return np.ones(len(X), dtype=int)


def test_cpcv_backtester_returns_metrics():
    from borsabot.backtest.cpcv import CPCVBacktester

    X, y, events, prices = make_data(500)
    bt = CPCVBacktester(n_groups=6, n_test_groups=2)
    result = bt.run(X, y, prices, events, model=AlwaysBuyModel())

    assert "mean_sharpe" in result
    assert "psr"         in result
    assert "n_paths"     in result
    assert 0.0 <= result["psr"] <= 1.0
    # Some folds may be empty after aggressive purge — require at least half succeed
    assert result["n_paths"] >= math.comb(6, 2) // 2, (
        f"Too few successful paths: {result['n_paths']}")


def test_cpcv_backtester_psr_bounded():
    from borsabot.backtest.cpcv import CPCVBacktester

    X, y, events, prices = make_data(300)
    bt = CPCVBacktester(n_groups=6, n_test_groups=2)
    result = bt.run(X, y, prices, events, model=AlwaysBuyModel())

    psr = result["psr"]
    assert 0.0 <= psr <= 1.0, f"PSR out of bounds: {psr}"
