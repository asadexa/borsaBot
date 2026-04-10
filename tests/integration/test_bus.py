"""Integration tests for ZeroMQ message bus (Publisher + Subscriber).

On Windows, pytest-asyncio uses ProactorEventLoop which is incompatible with
zmq.asyncio (requires add_reader()). We work around this by running each test
body via asyncio.run() with WindowsSelectorEventLoopPolicy explicitly set,
rather than relying on pytest-asyncio's loop management.
"""

import asyncio
import sys

import pytest

from borsabot.core.bus import Publisher, Subscriber
from borsabot.core.events import Event, EventType


def _run(coro):
    """Run a coroutine with WindowsSelectorEventLoopPolicy on Windows."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Tests are written as plain (sync) functions that call _run() internally.
# This sidesteps pytest-asyncio's loop management entirely.
# ─────────────────────────────────────────────────────────────────────────────

def test_pub_sub_round_trip():
    """Published event should be received by subscriber with correct fields."""
    async def _body():
        pub = Publisher("tcp://127.0.0.1:15558")
        sub = Subscriber("tcp://127.0.0.1:15558")

        await pub.start()
        await sub.start([EventType.TICK])
        await asyncio.sleep(0.3)

        event = Event(
            event_type=EventType.TICK,
            symbol="BTCUSDT",
            source="test",
            payload={"price": 50000.0, "qty": 0.01},
        )
        await pub.publish(event)

        received = await asyncio.wait_for(sub.receive(), timeout=5.0)

        assert received.event_type == event.event_type
        assert received.symbol == event.symbol
        assert received.payload["price"] == pytest.approx(50000.0)

        await pub.stop()
        await sub.stop()

    _run(_body())


def test_subscriber_filters_by_topic():
    """Subscriber subscribed to BOOK_UPDATE should NOT receive TICK events."""
    async def _body():
        pub = Publisher("tcp://127.0.0.1:15559")
        sub = Subscriber("tcp://127.0.0.1:15559")

        await pub.start()
        await sub.start([EventType.BOOK_UPDATE])
        await asyncio.sleep(0.3)

        # Publish a TICK (not subscribed) — should be filtered out
        tick = Event(event_type=EventType.TICK, symbol="ETHUSDT", payload={})
        await pub.publish(tick)

        # Publish a BOOK_UPDATE (subscribed) — should be received
        book = Event(event_type=EventType.BOOK_UPDATE, symbol="ETHUSDT", payload={"u": 42})
        await pub.publish(book)

        received = await asyncio.wait_for(sub.receive(), timeout=5.0)
        assert received.event_type == EventType.BOOK_UPDATE

        await pub.stop()
        await sub.stop()

    _run(_body())


def test_event_sequence_ids_monotonic():
    """Multiple published events should have increasing sequence IDs."""
    async def _body():
        pub = Publisher("tcp://127.0.0.1:15560")
        sub = Subscriber("tcp://127.0.0.1:15560")

        await pub.start()
        await sub.start([EventType.TICK])
        await asyncio.sleep(0.3)

        events: list[Event] = []
        for i in range(5):
            e = Event(event_type=EventType.TICK, symbol="BTCUSDT", payload={"i": i})
            await pub.publish(e)

        for _ in range(5):
            ev = await asyncio.wait_for(sub.receive(), timeout=5.0)
            events.append(ev)

        seq_ids = [e.sequence_id for e in events]
        assert seq_ids == sorted(seq_ids), "Sequence IDs not monotonically increasing"

        await pub.stop()
        await sub.stop()

    _run(_body())
