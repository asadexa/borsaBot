"""Database State Manager for recovering NAV and positions."""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    import asyncpg
except ImportError:
    asyncpg = None

log = logging.getLogger(__name__)


class StateDB:
    """Handles persistence and recovery of portfolio state via TimescaleDB."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: "asyncpg.Pool | None" = None

    async def connect(self) -> None:
        """Initialize the database connection pool."""
        if not asyncpg:
            log.warning("asyncpg not installed, DB state recovery disabled.")
            return

        try:
            self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
            log.info("StateDB connected to %s", self.dsn)
        except Exception as exc:
            log.error("Failed to connect StateDB: %s", exc)

    async def disconnect(self) -> None:
        """Close the database connection pool."""
        if self._pool:
            await self._pool.close()

    async def save_snapshot(self, nav: float, total_exposure: float, positions: dict[str, float]) -> None:
        """Save a snapshot of the portfolio state."""
        if not self._pool:
            return

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO portfolio_snapshots(time, nav, total_exposure, positions)
                    VALUES (NOW(), $1, $2, $3::jsonb)
                    """,
                    nav,
                    total_exposure,
                    json.dumps(positions),
                )
            log.debug("Portfolio snapshot saved: NAV=%.2f", nav)
        except Exception as exc:
            log.error("Failed to save portfolio snapshot: %s", exc)

    async def load_latest_snapshot(self) -> dict[str, Any] | None:
        """Load the most recent portfolio snapshot to recover state."""
        if not self._pool:
            return None

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT nav, total_exposure, positions
                    FROM portfolio_snapshots
                    ORDER BY time DESC
                    LIMIT 1
                    """
                )

            if row:
                return {
                    "nav":             row["nav"],
                    "total_exposure":  row["total_exposure"],
                    "positions":       json.loads(row["positions"]),
                }
        except Exception as exc:
            log.error("Failed to load portfolio snapshot: %s", exc)

        return None
