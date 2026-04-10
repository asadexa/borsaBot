"""Tick normalizer — converts broker-specific raw tick formats to the
canonical Event format used by the rest of the platform."""

from __future__ import annotations

import time

from borsabot.core.events import Event, EventType


class TickNormalizer:
    """
    Normalizes raw tick/trade dicts from different brokers into a
    consistent Event payload:

        {
            "price": float,
            "qty":   float,
            "side":  "B" | "S" | "",
            "bid":   float | None,
            "ask":   float | None,
        }
    """

    # ── Binance aggTrade ──────────────────────────────────────────────────

    @staticmethod
    def from_binance_agg_trade(raw: dict) -> Event:
        return Event(
            event_type=EventType.TICK,
            symbol=raw.get("s", raw.get("symbol", "")),
            source="binance",
            timestamp_ns=int(raw.get("T", time.time() * 1000)) * 1_000_000,
            payload={
                "price": float(raw["p"]),
                "qty":   float(raw["q"]),
                "side":  "S" if raw.get("m") else "B",  # m=True → market sell
                "bid":   None,
                "ask":   None,
            },
        )

    # ── Binance depth update → book event ────────────────────────────────

    @staticmethod
    def from_binance_depth(raw: dict) -> Event:
        stream = raw.get("stream", "")
        symbol = stream.split("@")[0].upper() if "@" in stream else raw.get("s", "")
        data = raw.get("data", raw)
        return Event(
            event_type=EventType.BOOK_UPDATE,
            symbol=symbol,
            source="binance",
            timestamp_ns=int(data.get("E", time.time() * 1000)) * 1_000_000,
            payload={
                "U":   data.get("U", 0),       # first update ID
                "u":   data.get("u", 0),       # last update ID
                "b":   data.get("b", []),      # bid changes
                "a":   data.get("a", []),      # ask changes
            },
        )

    # ── Interactive Brokers tick ──────────────────────────────────────────

    @staticmethod
    def from_ib_tick(raw: dict) -> Event:
        return Event(
            event_type=EventType.TICK,
            symbol=raw.get("symbol", ""),
            source="interactive_brokers",
            timestamp_ns=raw.get("ts", time.time_ns()),
            payload={
                "price": float(raw.get("price", 0)),
                "qty":   float(raw.get("qty", 0)),
                "side":  "",
                "bid":   raw.get("bid"),
                "ask":   raw.get("ask"),
            },
        )

    # ── MetaTrader 5 tick ─────────────────────────────────────────────────

    @staticmethod
    def from_mt5_tick(raw: dict) -> Event:
        bid = raw.get("bid", 0.0)
        ask = raw.get("ask", 0.0)
        return Event(
            event_type=EventType.TICK,
            symbol=raw.get("symbol", ""),
            source="metatrader5",
            timestamp_ns=raw.get("ts", time.time_ns()),
            payload={
                "price": (bid + ask) / 2,
                "qty":   0.0,
                "side":  "",
                "bid":   bid,
                "ask":   ask,
            },
        )
