"""Market data feed integrity monitor.

Detects gaps, stale feeds, and unexpected sequence breaks.
Publishes risk.feed_gap events on the message bus when anomalies are found.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class FeedStats:
    symbol: str
    last_ts_ns: int = 0
    last_seq: int = 0
    gap_count: int = 0
    seq_break_count: int = 0
    total_events: int = 0
    max_gap_ms: float = 0.0


class FeedMonitor:
    """
    Tracks per-symbol feed health.

    - Gap detection: alerts when time between consecutive events exceeds threshold
    - Sequence break detection: alerts when sequence IDs have gaps
    - Stale feed detection: alerts when a symbol stops sending data entirely
    """

    def __init__(
        self,
        max_gap_ms: float = 500.0,
        stale_feed_ms: float = 5_000.0,
    ) -> None:
        self._max_gap_ns = int(max_gap_ms * 1_000_000)
        self._stale_ns   = int(stale_feed_ms * 1_000_000)
        self._stats: dict[str, FeedStats] = {}

    # ── Core API ──────────────────────────────────────────────────────────

    def record(
        self,
        symbol: str,
        ts_ns: int,
        sequence_id: int | None = None,
    ) -> list[str]:
        """
        Record a new market data event. Returns list of alert messages (empty = healthy).
        """
        alerts: list[str] = []

        if symbol not in self._stats:
            self._stats[symbol] = FeedStats(symbol=symbol)

        stats = self._stats[symbol]
        stats.total_events += 1

        # ── Gap check ────────────────────────────────────────────────────
        if stats.last_ts_ns > 0:
            gap_ns = ts_ns - stats.last_ts_ns
            gap_ms = gap_ns / 1_000_000

            if gap_ns > self._max_gap_ns:
                stats.gap_count += 1
                stats.max_gap_ms = max(stats.max_gap_ms, gap_ms)
                msg = f"FEED_GAP {symbol}: {gap_ms:.1f}ms gap detected"
                log.warning(msg)
                alerts.append(msg)

        # ── Sequence break check ──────────────────────────────────────────
        if sequence_id is not None and stats.last_seq > 0:
            expected = stats.last_seq + 1
            if sequence_id > expected:
                missed = sequence_id - expected
                stats.seq_break_count += 1
                msg = f"SEQ_BREAK {symbol}: missed {missed} events (expected {expected}, got {sequence_id})"
                log.warning(msg)
                alerts.append(msg)

        stats.last_ts_ns = ts_ns
        if sequence_id is not None:
            stats.last_seq = sequence_id

        return alerts

    def check_stale(self) -> list[str]:
        """
        Check all tracked symbols for stale feeds.
        Call this periodically (e.g. every second).
        """
        now = time.time_ns()
        alerts: list[str] = []
        for sym, stats in self._stats.items():
            if stats.last_ts_ns > 0:
                silence_ns = now - stats.last_ts_ns
                silence_ms = silence_ns / 1_000_000
                if silence_ns > self._stale_ns:
                    msg = f"STALE_FEED {sym}: no data for {silence_ms:.0f}ms"
                    log.error(msg)
                    alerts.append(msg)
        return alerts

    def get_stats(self, symbol: str) -> FeedStats | None:
        return self._stats.get(symbol)

    def all_stats(self) -> dict[str, FeedStats]:
        return dict(self._stats)

    def reset(self, symbol: str) -> None:
        """Reset tracking for a symbol (e.g. after reconnect)."""
        self._stats.pop(symbol, None)
