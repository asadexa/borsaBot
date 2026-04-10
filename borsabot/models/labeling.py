"""Triple Barrier Method for financial ML labeling.

Reference: Marcos López de Prado — Advances in Financial Machine Learning, Ch.3

Labels each trade event as:
  +1 → profit-take barrier hit first
  -1 → stop-loss barrier hit first
   0 → time expiry (neither barrier hit)
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_daily_volatility(
    close: pd.Series,
    span: int = 100,
) -> pd.Series:
    """
    Exponentially weighted daily volatility estimate.
    Used as the dynamic width for profit-take and stop-loss barriers.
    """
    returns = close.pct_change().dropna()
    return returns.ewm(span=span, min_periods=span).std()


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: list[float],
    trgt: pd.Series,
    min_ret: float = 0.0,
    num_threads: int = 1,
    t1: pd.Series | None = None,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute event dataframe for triple barrier labeling.

    Args:
        close:     Price series
        t_events:  Timestamps to sample
        pt_sl:     [profit_take_mult, stop_loss_mult]
        trgt:      Target (volatility) series aligned to close
        min_ret:   Minimum return required to create an event
        t1:        Expiry time for each event (if None: next available timestamp)
        side:      If provided, use meta-labeling (side must be +1 or -1)

    Returns:
        DataFrame with index=t0, columns=[t1, trgt, side]
    """
    trgt = trgt.reindex(t_events).dropna()
    if min_ret > 0:
        trgt = trgt[trgt > min_ret]

    if t1 is None:
        # Expiry = 1 bar ahead (minimum holding period)
        t1 = pd.Series(
            [close.index[close.index.searchsorted(t) + 1]
             if close.index.searchsorted(t) + 1 < len(close)
             else close.index[-1]
             for t in trgt.index],
            index=trgt.index,
        )

    if side is None:
        side_ = pd.Series(1.0, index=trgt.index)  # always long (direction-agnostic)
    else:
        side_ = side.reindex(trgt.index).fillna(0)

    return pd.DataFrame({"t1": t1, "trgt": trgt, "side": side_})


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: list[float] = (1.0, 1.0),
) -> pd.Series:
    """
    Apply Triple Barrier Method to assign event labels.

    Args:
        close:  Price series
        events: DataFrame from get_events() with columns [t1, trgt, side]
        pt_sl:  [profit_mult, stop_loss_mult]

    Returns:
        Series of labels {-1, 0, +1} indexed by event timestamp.
    """
    labels: dict[pd.Timestamp, int] = {}

    for t0, row in events.iterrows():
        t1     = row["t1"]
        trgt   = row["trgt"]
        side   = row.get("side", 1.0)

        if t0 not in close.index:
            continue

        path = close.loc[t0:t1]
        if path.empty:
            labels[t0] = 0
            continue

        ret = (path / close.loc[t0]) - 1.0

        # Directional barriers (meta-labeling: multiply by side)
        pt_barrier =  trgt * pt_sl[0] * side
        sl_barrier = -trgt * pt_sl[1] * abs(side)

        # Check which barrier is touched first
        up_cross   = ret[ret >= pt_barrier].index
        down_cross = ret[ret <= sl_barrier].index

        up_t   = up_cross[0]   if len(up_cross)   > 0 else pd.NaT
        down_t = down_cross[0] if len(down_cross) > 0 else pd.NaT

        if pd.isna(up_t) and pd.isna(down_t):
            labels[t0] = 0    # time expiry
        elif pd.isna(down_t) or (not pd.isna(up_t) and up_t <= down_t):
            labels[t0] = 1    # profit-take hit first
        else:
            labels[t0] = -1   # stop-loss hit first

    return pd.Series(labels, name="label")


def sample_weights(
    t1: pd.Series,
    events: pd.DataFrame,
    close: pd.Series,
) -> pd.Series:
    """
    Compute sample weights based on uniqueness of each label's observations.
    Reduces oversampling of overlapping events (López de Prado, Ch.4).
    """
    out = pd.Series(1.0, index=events.index)
    # Count how many events overlap at each time step
    for t0, t1_val in t1.items():
        count = events[(events.index <= t0) & (t1 > t0)].shape[0]
        out[t0] = 1.0 / max(count, 1)
    return out
