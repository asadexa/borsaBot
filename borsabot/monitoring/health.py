"""Health-check HTTP endpoints for Docker / Kubernetes probes.

Exposes two endpoints on a configurable port (default 8080):

    GET /health  → liveness probe  (200 = process is alive)
    GET /ready   → readiness probe (200 = all dependencies reachable)
    GET /status  → detailed JSON status of all subsystems

Usage:
    server = HealthServer(redis_url="redis://localhost:6379",
                          timescale_dsn="postgresql://...")
    await server.start()   # runs in background
    # ... rest of application
    await server.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

try:
    from aiohttp import web
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False
    log.warning("aiohttp not installed — HealthServer will not start. Run: pip install aiohttp")


@dataclass
class SubsystemStatus:
    name: str
    healthy: bool
    latency_ms: float = 0.0
    error: str = ""
    checked_at: float = field(default_factory=time.time)


class HealthServer:
    """
    Lightweight aiohttp server for liveness / readiness probes.

    Tracks dynamic subsystem health via register_check().
    """

    def __init__(
        self,
        port: int = 8080,
        redis_url: str | None = None,
        timescale_dsn: str | None = None,
    ) -> None:
        self.port           = port
        self._redis_url     = redis_url
        self._tsdb_dsn      = timescale_dsn
        self._runner        = None
        self._site          = None
        self._start_time    = time.time()
        self._custom_checks: dict[str, "asyncio.Callable"] = {}

    # ── Probe registration ────────────────────────────────────────────────

    def register_check(self, name: str, check_fn) -> None:
        """
        Register a custom async health check.

        check_fn must be an async callable returning (bool, str):
            True, ""              → healthy
            False, "error msg"   → unhealthy
        """
        self._custom_checks[name] = check_fn

    # ── Built-in checks ───────────────────────────────────────────────────

    async def _check_redis(self) -> SubsystemStatus:
        t = time.monotonic()
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(self._redis_url, socket_connect_timeout=1)
            await r.ping()
            await r.aclose()
            return SubsystemStatus("redis", True, (time.monotonic() - t) * 1000)
        except Exception as exc:
            return SubsystemStatus("redis", False, error=str(exc))

    async def _check_timescaledb(self) -> SubsystemStatus:
        t = time.monotonic()
        try:
            import asyncpg
            conn = await asyncio.wait_for(
                asyncpg.connect(self._tsdb_dsn), timeout=2.0
            )
            await conn.fetchval("SELECT 1")
            await conn.close()
            return SubsystemStatus("timescaledb", True, (time.monotonic() - t) * 1000)
        except Exception as exc:
            return SubsystemStatus("timescaledb", False, error=str(exc))

    async def _gather_status(self) -> list[SubsystemStatus]:
        checks = []
        if self._redis_url:
            checks.append(self._check_redis())
        if self._tsdb_dsn:
            checks.append(self._check_timescaledb())
        for name, fn in self._custom_checks.items():
            async def _wrapped(n=name, f=fn):
                t = time.monotonic()
                try:
                    ok, err = await f()
                    return SubsystemStatus(n, ok, (time.monotonic() - t) * 1000, err)
                except Exception as exc:
                    return SubsystemStatus(n, False, error=str(exc))
            checks.append(_wrapped())

        return list(await asyncio.gather(*checks, return_exceptions=False))

    # ── Request handlers ──────────────────────────────────────────────────

    async def handle_health(self, request: "web.Request") -> "web.Response":
        """Liveness probe — always 200 if server is running."""
        return web.Response(
            text=json.dumps({"status": "ok", "uptime_sec": round(time.time() - self._start_time, 1)}),
            content_type="application/json",
        )

    async def handle_ready(self, request: "web.Request") -> "web.Response":
        """Readiness probe — 200 only if all registered subsystems are healthy."""
        statuses = await self._gather_status()
        all_healthy = all(s.healthy for s in statuses)
        body = {
            "ready": all_healthy,
            "checks": {s.name: {"ok": s.healthy, "latency_ms": round(s.latency_ms, 2), "error": s.error}
                       for s in statuses},
        }
        return web.Response(
            status=200 if all_healthy else 503,
            text=json.dumps(body),
            content_type="application/json",
        )

    async def handle_status(self, request: "web.Request") -> "web.Response":
        """Detailed JSON status for dashboards and monitoring."""
        statuses = await self._gather_status()
        body = {
            "status":     "ok" if all(s.healthy for s in statuses) else "degraded",
            "uptime_sec": round(time.time() - self._start_time, 1),
            "version":    "0.1.0",
            "subsystems": {
                s.name: {
                    "healthy":    s.healthy,
                    "latency_ms": round(s.latency_ms, 2),
                    "error":      s.error,
                    "checked_at": s.checked_at,
                }
                for s in statuses
            },
        }
        return web.Response(
            text=json.dumps(body, indent=2),
            content_type="application/json",
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not _AIOHTTP_AVAILABLE:
            log.warning("HealthServer not started: aiohttp not installed")
            return

        app = web.Application()
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/ready",  self.handle_ready)
        app.router.add_get("/status", self.handle_status)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await self._site.start()
        log.info("HealthServer started on port %d", self.port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            log.info("HealthServer stopped")
