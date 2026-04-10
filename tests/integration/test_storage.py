"""Integration tests for storage layer.

These tests require running Docker infrastructure:
    docker compose up -d timescaledb redis

Run with:
    pytest tests/integration/test_storage.py -v -m integration

All tests are marked @pytest.mark.integration and will be skipped
automatically if the services are not reachable.
"""

from __future__ import annotations

import asyncio
import time
import uuid
import pytest
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

TIMESCALE_DSN = "postgresql://borsabot:secret@localhost:5432/borsabot"
REDIS_URL     = "redis://localhost:6379"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires external services")


async def _redis_reachable() -> bool:
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(REDIS_URL, socket_connect_timeout=1)
        await asyncio.wait_for(r.ping(), timeout=1.0)
        await r.aclose()
        return True
    except Exception:
        return False


async def _tsdb_reachable() -> bool:
    try:
        import asyncpg
        conn = await asyncio.wait_for(asyncpg.connect(TIMESCALE_DSN), timeout=2.0)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def redis_available():
    return await _redis_reachable()


@pytest.fixture(scope="module")
async def tsdb_available():
    return await _tsdb_reachable()


def make_tick(symbol: str = "BTCUSDT", seq: int = 1) -> dict:
    return {
        "symbol":       symbol,
        "price":        50_000.0 + seq,
        "qty":          0.01,
        "side":         "B",
        "sequence_id":  seq,
        "timestamp_ns": time.time_ns(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Redis cache tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_ping(redis_available):
    if not redis_available:
        pytest.skip("Redis not reachable")
    from borsabot.storage.redis_cache import MarketCache
    cache = MarketCache(REDIS_URL, ttl=5)
    await cache.start()
    assert await cache.health_check()
    await cache.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_cache_set_get_tick(redis_available):
    if not redis_available:
        pytest.skip("Redis not reachable")
    from borsabot.storage.redis_cache import MarketCache
    cache = MarketCache(REDIS_URL, ttl=5)
    await cache.start()

    tick = make_tick()
    await cache.set_tick("BTCUSDT", tick)
    retrieved = await cache.get_tick("BTCUSDT")

    assert retrieved is not None
    assert retrieved["symbol"] == "BTCUSDT"
    assert retrieved["price"] == pytest.approx(50_001.0)

    await cache.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_cache_set_book(redis_available):
    if not redis_available:
        pytest.skip("Redis not reachable")
    from borsabot.storage.redis_cache import MarketCache
    from borsabot.market_data.order_book import OrderBook

    cache = MarketCache(REDIS_URL, ttl=5)
    await cache.start()

    book = OrderBook("BTCUSDT")
    book.apply_snapshot({
        "lastUpdateId": 9999,
        "bids": [[50000.0, 1.0]],
        "asks": [[50010.0, 1.0]],
    })
    await cache.set_book("BTCUSDT", book.to_dict())
    raw = await cache.get_book("BTCUSDT")

    assert raw is not None
    restored = OrderBook.from_dict(raw)
    assert restored.best_bid() == pytest.approx(50000.0)
    assert restored.best_ask() == pytest.approx(50010.0)

    await cache.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_cache_ttl_expiry(redis_available):
    """Set with TTL=1s, wait 2s, confirm key is gone."""
    if not redis_available:
        pytest.skip("Redis not reachable")
    from borsabot.storage.redis_cache import MarketCache

    cache = MarketCache(REDIS_URL, ttl=1)
    await cache.start()

    key = f"test:ttl:{uuid.uuid4().hex}"
    await cache.set(key, {"v": 1}, ttl=1)

    val = await cache.get(key)
    assert val is not None

    await asyncio.sleep(1.2)
    val_after = await cache.get(key)
    assert val_after is None

    await cache.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Parquet lake tests (no Docker required)
# ─────────────────────────────────────────────────────────────────────────────

def test_parquet_lake_write_read(tmp_path):
    """ParquetLake should write and read ticks round-trip (no Docker needed)."""
    from borsabot.storage.parquet_lake import ParquetLake
    import datetime

    lake = ParquetLake(base_path=tmp_path / "lake", batch_size=5)

    for i in range(10):
        lake.write({
            "symbol":      "BTCUSDT",
            "price":       50_000.0 + i,
            "qty":         0.01,
            "side":        "B",
            "sequence_id": i,
            "timestamp_ns": time.time_ns() + i,
        })

    lake.flush()

    date_str = datetime.date.today().isoformat()
    df = lake.read("BTCUSDT", date_str)
    assert len(df) == 10
    assert "price" in df.columns
    assert df["price"].min() == pytest.approx(50_000.0)


def test_parquet_lake_appends_intraday(tmp_path):
    """Second flush should append to the same day's file."""
    from borsabot.storage.parquet_lake import ParquetLake
    import datetime

    lake = ParquetLake(base_path=tmp_path / "lake2", batch_size=3)

    for i in range(3):
        lake.write({
            "symbol": "ETHUSDT", "price": 3000.0 + i, "qty": 0.1,
            "side": "S", "sequence_id": i, "timestamp_ns": time.time_ns(),
        })
    lake.flush()

    for i in range(3, 6):
        lake.write({
            "symbol": "ETHUSDT", "price": 3000.0 + i, "qty": 0.1,
            "side": "S", "sequence_id": i, "timestamp_ns": time.time_ns(),
        })
    lake.flush()

    date_str = datetime.date.today().isoformat()
    df = lake.read("ETHUSDT", date_str)
    assert len(df) == 6


# ─────────────────────────────────────────────────────────────────────────────
# TimescaleDB tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_timescale_write_and_deduplicate(tsdb_available):
    if not tsdb_available:
        pytest.skip("TimescaleDB not reachable")

    from borsabot.storage.timescale import IntegrityWriter

    writer = IntegrityWriter(TIMESCALE_DSN)
    await writer.start()

    # Create table (normally done by seed_db.py)
    import asyncpg
    conn = await asyncpg.connect(TIMESCALE_DSN)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS ticks (
            time TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            price DOUBLE PRECISION,
            qty DOUBLE PRECISION,
            side CHAR(1),
            sequence_id BIGINT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ticks_test_dedup_idx
            ON ticks (symbol, time, sequence_id);
    """)
    await conn.close()

    tick = make_tick(symbol="TESTUSDT", seq=99999)

    # First write — should succeed
    result1 = await writer.write_tick(tick)
    assert result1 is True

    # Second write with same seq — should be deduplicated (still True)
    result2 = await writer.write_tick(tick)
    assert result2 is True   # dedup in memory = silent skip

    await writer.stop()
