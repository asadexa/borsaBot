"""TimescaleDB async writer with integrity guarantees.

Features:
- Duplicate suppression via sequence_id deduplication
- ON CONFLICT DO NOTHING for idempotent writes
- Automatic retry with exponential backoff on connection drops
- Dead-letter Parquet file for writes that fail after all retries
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

try:
    import asyncpg
    _ASYNCPG_AVAILABLE = True
except ImportError:
    _ASYNCPG_AVAILABLE = False

# SQL DDL (run via seed_db.py at startup)
CREATE_TICKS_SQL = """
CREATE TABLE IF NOT EXISTS ticks (
    time         TIMESTAMPTZ  NOT NULL,
    symbol       TEXT         NOT NULL,
    price        DOUBLE PRECISION,
    qty          DOUBLE PRECISION,
    side         CHAR(1),
    sequence_id  BIGINT       NOT NULL
);

DO $$ BEGIN
    PERFORM create_hypertable('ticks', 'time', if_not_exists => TRUE);
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS ticks_symbol_seq_idx
    ON ticks (symbol, time, sequence_id);
"""


class IntegrityWriter:
    """Async TimescaleDB writer with dedup + retry + dead-letter fallback."""

    MAX_RETRIES = 3
    DEAD_LETTER_PATH = Path("data/dead_letter")

    def __init__(self, dsn: str, dedup_window: int = 100_000) -> None:
        self._dsn = dsn
        self._seen: dict[str, set[int]] = {}   # symbol → set of seq_ids
        self._dedup_window = dedup_window
        self._pool: "asyncpg.Pool | None" = None

    async def start(self) -> None:
        if not _ASYNCPG_AVAILABLE:
            raise RuntimeError("asyncpg not installed")
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        log.info("TimescaleDB pool ready: %s", self._dsn)

    async def stop(self) -> None:
        if self._pool:
            await self._pool.close()

    async def write_tick(self, tick: dict) -> bool:
        """
        Write a single tick. Returns True on success, False on all-retry failure.

        tick = {symbol, price, qty, side, sequence_id, timestamp_ns?}
        """
        seq = tick["sequence_id"]
        sym = tick["symbol"]

        # ── Deduplication ─────────────────────────────────────────────────
        seen_set = self._seen.setdefault(sym, set())
        if seq in seen_set:
            return True                         # duplicate, silently skip
        seen_set.add(seq)
        # Rolling window: evict oldest to bound memory
        if len(seen_set) > self._dedup_window:
            seen_set.discard(min(seen_set))

        # ── Write with retry ──────────────────────────────────────────────
        ts = datetime.datetime.fromtimestamp(
            tick.get("timestamp_ns", time.time_ns()) / 1e9,
            tz=datetime.timezone.utc,
        )
        for attempt in range(self.MAX_RETRIES):
            try:
                async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                    await conn.execute(
                        """
                        INSERT INTO ticks(time, symbol, price, qty, side, sequence_id)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT DO NOTHING
                        """,
                        ts,
                        sym,
                        float(tick.get("price", 0)),
                        float(tick.get("qty", 0)),
                        tick.get("side", ""),
                        int(seq),
                    )
                return True
            except Exception as exc:
                wait = 0.1 * (2**attempt)
                log.warning("DB write attempt %d failed: %s — retry in %.2fs", attempt + 1, exc, wait)
                await asyncio.sleep(wait)

        # ── Dead-letter fallback ──────────────────────────────────────────
        log.error("All retries exhausted for seq=%d — writing to dead-letter", seq)
        self._write_dead_letter(tick)
        return False

    def _write_dead_letter(self, tick: dict) -> None:
        self.DEAD_LETTER_PATH.mkdir(parents=True, exist_ok=True)
        path = self.DEAD_LETTER_PATH / f"dead_{tick['symbol']}_{int(time.time())}.parquet"
        table = pa.table({k: [v] for k, v in tick.items()})
        pq.write_table(table, path)
        log.info("Dead-letter written to %s", path)
