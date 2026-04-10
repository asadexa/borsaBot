"""Redis hot cache for latest market data (book state, tick, features).

Uses msgpack for compact binary serialization.
All keys have a configurable TTL to prevent stale data accumulation.
"""

from __future__ import annotations

import logging
from typing import Any

import msgpack

log = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class MarketCache:
    """
    Thin async Redis wrapper for hot-path market data caching.

    Key namespaces:
      book:<symbol>      → serialized OrderBook state (dict)
      tick:<symbol>      → latest tick payload (dict)
      features:<symbol>  → latest feature vector (dict)
    """

    DEFAULT_TTL = 60  # seconds

    def __init__(self, redis_url: str, ttl: int = DEFAULT_TTL) -> None:
        self._url = redis_url
        self._ttl = ttl
        self._client: "aioredis.Redis | None" = None

    async def start(self) -> None:
        if not _REDIS_AVAILABLE:
            raise RuntimeError("redis[hiredis] not installed")
        self._client = aioredis.from_url(self._url, decode_responses=False)
        await self._client.ping()
        log.info("Redis cache connected: %s", self._url)

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()

    # ── Generic get/set ───────────────────────────────────────────────────

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        assert self._client, "Cache not started"
        packed = msgpack.packb(value, use_bin_type=True)
        await self._client.setex(key, ttl or self._ttl, packed)

    async def get(self, key: str) -> Any | None:
        assert self._client, "Cache not started"
        raw = await self._client.get(key)
        return msgpack.unpackb(raw, raw=False) if raw else None

    # ── Named helpers ─────────────────────────────────────────────────────

    async def set_book(self, symbol: str, book_dict: dict) -> None:
        await self.set(f"book:{symbol}", book_dict)

    async def get_book(self, symbol: str) -> dict | None:
        return await self.get(f"book:{symbol}")

    async def set_tick(self, symbol: str, tick: dict) -> None:
        await self.set(f"tick:{symbol}", tick)

    async def get_tick(self, symbol: str) -> dict | None:
        return await self.get(f"tick:{symbol}")

    async def set_features(self, symbol: str, features: dict) -> None:
        await self.set(f"features:{symbol}", features)

    async def get_features(self, symbol: str) -> dict | None:
        return await self.get(f"features:{symbol}")

    async def health_check(self) -> bool:
        """Returns True if Redis responds to PING."""
        try:
            assert self._client
            return await self._client.ping()
        except Exception:
            return False
