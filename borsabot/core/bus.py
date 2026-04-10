"""ZeroMQ-based publish/subscribe message bus.

All platform components communicate exclusively through this bus,
ensuring complete decoupling between producers and consumers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Callable

import msgpack
import zmq
import zmq.asyncio

from borsabot.core.events import Event

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────


class Publisher:
    """Async ZeroMQ PUB socket wrapper."""

    def __init__(self, bind_addr: str) -> None:
        self._addr = bind_addr
        self._ctx: zmq.asyncio.Context | None = None
        self._sock: zmq.asyncio.Socket | None = None

    async def start(self) -> None:
        self._ctx = zmq.asyncio.Context()           # fresh context per instance
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.set_hwm(100_000)                  # high-water mark
        self._sock.bind(self._addr)
        await asyncio.sleep(0.05)                    # allow bind to settle
        log.info("Publisher bound to %s", self._addr)

    async def publish(self, event: Event) -> None:
        if self._sock is None:
            raise RuntimeError("Publisher not started. Call await start() first.")
        topic = event.topic.encode()
        data = msgpack.packb(
            {
                "event_type": event.event_type,
                "symbol": event.symbol,
                "source": event.source,
                "timestamp_ns": event.timestamp_ns,
                "sequence_id": event.sequence_id,
                "payload": event.payload,
            },
            use_bin_type=True,
        )
        await self._sock.send_multipart([topic, data])

    async def stop(self) -> None:
        if self._sock:
            self._sock.close(linger=0)
            self._sock = None
            await asyncio.sleep(0.01)  # Allow asyncio event loop to remove reader/writer FD
        if self._ctx:
            self._ctx.term()
            self._ctx = None
        log.info("Publisher stopped")



class Subscriber:
    """Async ZeroMQ SUB socket wrapper."""

    def __init__(self, connect_addr: str) -> None:
        self._addr = connect_addr
        self._ctx: zmq.asyncio.Context | None = None
        self._sock: zmq.asyncio.Socket | None = None

    async def start(self, topics: list[str]) -> None:
        self._ctx = zmq.asyncio.Context()           # fresh context per instance
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.set_hwm(100_000)
        self._sock.connect(self._addr)
        for t in topics:
            self._sock.setsockopt(zmq.SUBSCRIBE, t.encode())
        log.info("Subscriber connected to %s, topics=%s", self._addr, topics)

    async def receive(self) -> Event:
        if self._sock is None:
            raise RuntimeError("Subscriber not started. Call await start() first.")
        _topic, raw = await self._sock.recv_multipart()
        d = msgpack.unpackb(raw, raw=False)
        return Event(
            event_type=d["event_type"],
            symbol=d["symbol"],
            source=d["source"],
            timestamp_ns=d["timestamp_ns"],
            sequence_id=d["sequence_id"],
            payload=d["payload"],
        )

    async def stream(self) -> AsyncIterator[Event]:
        """Continuously yield events from the bus."""
        while True:
            yield await self.receive()

    async def stop(self) -> None:
        if self._sock:
            self._sock.close(linger=0)
            self._sock = None
            await asyncio.sleep(0.01)  # Allow asyncio event loop to remove reader/writer FD
        if self._ctx:
            self._ctx.term()
            self._ctx = None
        log.info("Subscriber stopped")



# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper to start a background consumer
# ─────────────────────────────────────────────────────────────────────────────

async def consume(
    sub: Subscriber,
    handler: Callable[[Event], None | asyncio.coroutine],
) -> None:
    """Run handler on every event received by sub (blocking loop)."""
    async for event in sub.stream():
        result = handler(event)
        if asyncio.iscoroutine(result):
            await result
