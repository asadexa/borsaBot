"""Unit tests for LiveTrader — the live trading orchestrator."""

from __future__ import annotations

import asyncio
import logging
import math
import pytest
import numpy as np
import pandas as pd

from borsabot.core.trader import LiveTrader, TraderConfig
from borsabot.brokers.base import MockBrokerGateway


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_broker():
    return MockBrokerGateway()


@pytest.fixture
def config():
    return TraderConfig(
        # Point model paths to /nonexistent/ so no real models get loaded
        primary_model_path = "/nonexistent/{symbol}_primary.pkl",
        meta_model_path    = "/nonexistent/{symbol}_meta.pkl",
        regime_model_path  = "/nonexistent/regime.pkl",
        nav=100_000.0,
        max_position_usd=10_000.0,
        max_drawdown_pct=0.05,
        meta_threshold=0.55,
        tick_history_size=50,
        regime_window=20,
        health_check_interval=99_999,   # don't fire during tests
        twap_duration_sec=1,            # 1s total = no meaningful sleep
        twap_slices=1,                  # single slice -> TWAP never sleeps between slices
        max_slippage_bps=10_000.0,      # very high threshold so tests aren't blocked
    )


@pytest.fixture
def trader(mock_broker, config):
    return LiveTrader.from_config(config=config, broker=mock_broker)


# ─────────────────────────────────────────────────────────────────────────────
# Construction & configuration
# ─────────────────────────────────────────────────────────────────────────────

def test_trader_from_config(mock_broker, config):
    t = LiveTrader.from_config(config=config, broker=mock_broker)
    assert t is not None
    assert t.cfg.nav == 100_000.0


def test_trader_is_running_initially(trader):
    assert trader.is_running is True


def test_trader_stop_sets_flag(trader):
    trader.stop()
    assert trader.is_running is False


def test_portfolio_summary_empty_before_setup(trader):
    # Risk engine not initialised before _setup() — summary should be {}
    assert trader.portfolio_summary == {}


# ─────────────────────────────────────────────────────────────────────────────
# _setup() initialises risk engine and per-symbol state
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_setup_creates_risk_engine(trader):
    await trader._setup(["BTCUSDT"])
    assert trader._risk_engine is not None


@pytest.mark.asyncio
async def test_setup_creates_books(trader):
    await trader._setup(["BTCUSDT", "ETHUSDT"])
    assert "BTCUSDT" in trader._books
    assert "ETHUSDT" in trader._books


@pytest.mark.asyncio
async def test_setup_creates_tick_history(trader):
    await trader._setup(["BTCUSDT"])
    assert "BTCUSDT" in trader._tick_history
    assert len(trader._tick_history["BTCUSDT"]) == 0


@pytest.mark.asyncio
async def test_setup_portfolio_summary_populated(trader):
    await trader._setup(["BTCUSDT"])
    summary = trader.portfolio_summary
    assert "nav"            in summary
    assert "total_exposure" in summary
    assert "daily_pnl"      in summary
    assert summary["nav"]   == pytest.approx(100_000.0)


# ─────────────────────────────────────────────────────────────────────────────
# _on_tick — insufficient history → no inference
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_on_tick_skips_with_insufficient_history(trader):
    """With fewer ticks than minimum, _on_tick should return silently."""
    await trader._setup(["BTCUSDT"])

    # Apply a book snapshot so mid_price() is available if needed
    trader._books["BTCUSDT"].apply_snapshot({
        "lastUpdateId": 1,
        "bids": [[49_990.0, 1.0]],
        "asks": [[50_010.0, 1.0]],
    })

    # Feed just a few ticks (below the min_rows threshold) — always include price
    for i in range(5):
        tick = {"symbol": "BTCUSDT", "price": 50_000.0 + i, "qty": 0.01, "side": "B"}
        await trader._on_tick("BTCUSDT", tick)

    # No primary model loaded → no trades opened
    summary = trader.portfolio_summary
    assert summary["total_exposure"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# _on_tick — no model loaded → no execution
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_on_tick_no_model_no_trades(trader):
    """Without a loaded model, no orders should be placed."""
    await trader._setup(["BTCUSDT"])

    # Feed enough ticks to cross history threshold — always include price
    for i in range(60):
        tick = {"symbol": "BTCUSDT", "price": 50_000.0 + i * 10, "qty": 0.01, "side": "B"}
        await trader._on_tick("BTCUSDT", tick)

    # Still no model — should remain at zero exposure
    assert trader.portfolio_summary["total_exposure"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# _on_tick with stub primary model → orders should flow through risk + exec
# ─────────────────────────────────────────────────────────────────────────────

class AlwaysBuyPrimary:
    """Stub primary model that always predicts BUY with confidence 0.8."""
    def predict_side(self, X):
        return 1, 0.8   # side=BUY, confidence=0.8


class AlwaysTradesMeta:
    """Stub meta model that always says 'yes, trade' with confidence 0.9."""
    threshold = 0.55
    def should_trade(self, X, primary_side, primary_conf):
        return True, 0.9


@pytest.mark.asyncio
async def test_on_tick_with_stub_model_executes_trade(trader):
    """With stub models that always signal BUY, the execution path should be reached
    and at least one order should fill, updating the tracked position."""
    await trader._setup(["BTCUSDT"])

    # Inject stub models
    trader._primary_models["BTCUSDT"] = AlwaysBuyPrimary()
    trader._meta_models["BTCUSDT"]   = AlwaysTradesMeta()

    # Apply a static book snapshot so microprice / spread are available
    book = trader._books["BTCUSDT"]
    book.apply_snapshot({
        "lastUpdateId": 1,
        "bids": [[49_990.0, 1.0]],
        "asks": [[50_010.0, 1.0]],
    })

    # Feed enough ticks to pass min-history threshold — always include price
    for i in range(60):
        tick = {"symbol": "BTCUSDT", "price": 50_000.0 + i * 5, "qty": 0.01, "side": "B"}
        await trader._on_tick("BTCUSDT", tick)

    # At least one BUY signal should have gone through the full pipeline
    summary = trader.portfolio_summary
    assert summary["total_exposure"] > 0, (
        "Expected non-zero exposure after BUY signals with stub models"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Risk engine integration via _on_tick
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_drawdown_halt_blocks_new_trade(trader):
    """After update_pnl triggers a halt, no new trades should be opened."""
    await trader._setup(["BTCUSDT"])

    # Inject stub models
    trader._primary_models["BTCUSDT"] = AlwaysBuyPrimary()
    trader._meta_models["BTCUSDT"]   = AlwaysTradesMeta()

    book = trader._books["BTCUSDT"]
    book.apply_snapshot({
        "lastUpdateId": 1,
        "bids": [[49_990.0, 1.0]],
        "asks": [[50_010.0, 1.0]],
    })

    # Trigger a halt first
    trader._risk_engine.update_pnl(-6_000.0)   # > 5% of 100k NAV

    before = trader.portfolio_summary["total_exposure"]

    # Now try feeding ticks — should all be blocked by halt; always include price
    for i in range(60):
        await trader._on_tick("BTCUSDT", {
            "symbol": "BTCUSDT", "price": 50_000.0 + i * 5, "qty": 0.01, "side": "B",
        })

    after = trader.portfolio_summary["total_exposure"]
    assert after == pytest.approx(before), "Halted engine should not allow new positions"


# ─────────────────────────────────────────────────────────────────────────────
# Short run → stop
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trader_run_and_stop(mock_broker, config):
    """LiveTrader.run() should complete cleanly when stop() is called.

    We stop the trader immediately by setting the stop event *before*
    calling run() so that the tick loop never reaches the 1-second sleep
    in MockBrokerGateway. This keeps the test fast.
    """
    config.tick_history_size = 20
    trader = LiveTrader.from_config(config=config, broker=mock_broker)

    # Signal stop immediately — run() should return without blocking
    trader.stop()
    await trader.run(symbols=["BTCUSDT"])

    assert not trader.is_running
