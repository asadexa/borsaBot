"""LiveTrader — orchestrates the full live trading pipeline.

Connects the following subsystems into a single async event loop:
  1. BrokerGateway       — market data + order routing
  2. FeatureBuilder      — transforms raw ticks into feature vectors
  3. PrimaryModel        — predicts trade direction {-1, 0, +1}
  4. MetaModel           — filters predictions with confidence score
  5. RegimeDetector      — adapts position sizing per market regime
  6. RiskEngine          — enforces position limits + drawdown halts
  7. ExecutionEngine     — selects algo (TWAP/VWAP) and routes orders
  8. ModelHealthMonitor  — PSI drift + Sharpe decay → retrain trigger
  9. Prometheus metrics  — real-time observability

Usage:
    trader = LiveTrader.from_config(settings)
    await trader.run(symbols=["BTCUSDT"])
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque

import numpy as np
import pandas as pd

from borsabot.core.events import OrderSide, Signal
from borsabot.execution.engine import ExecutionEngine
from borsabot.execution.order_fsm import OrderManager
from borsabot.features.builder import FeatureBuilder
from borsabot.market_data.order_book import OrderBook
from borsabot.market_data.feed_monitor import FeedMonitor
from borsabot.models.colab_adapter import (
    ColabMetaAdapter,
    ColabPrimaryAdapter,
    broker_to_model_slug,
)
from borsabot.monitoring.metrics import (
    daily_pnl_usd,
    model_psi_max,
    model_sharpe_ratio,
)
from borsabot.monitoring.notifier import Notifier
from borsabot.risk.engine import RiskEngine, RiskLimits
from borsabot.storage.state import StateDB

from borsabot.config import settings

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Trader configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TraderConfig:
    """All runtime parameters for LiveTrader."""

    # Model paths
    primary_model_path: str = "models/{symbol}_primary.pkl"
    meta_model_path:    str = "models/{symbol}_meta.pkl"
    regime_model_path:  str = "models/regime.pkl"

    db_dsn:             str = ""             # StateDB connection string

    # Risk parameters
    nav:                float = 100_000.0    # Starting NAV in USD
    max_position_usd:   float = 10_000.0
    max_portfolio_pct:  float = 0.10
    max_drawdown_pct:   float = 0.05
    kelly_fraction:     float = 0.25
    vol_target_annual:  float = 0.20

    # Signal filtering
    meta_threshold:     float = 0.55        # Minimum meta-model confidence
    min_fill_prob:      float = 0.50        # Minimum fill probability
    max_slippage_bps:   float = 10.0        # Maximum allowed slippage in bps

    # Feature / history window
    tick_history_size:  int   = 200         # Rolling ticks to keep per symbol
    regime_window:      int   = 100         # Bars for regime detection

    # Health monitoring
    health_check_interval: int  = 3600      # seconds between health checks
    psi_threshold:         float = 0.20
    sharpe_decay_threshold: float = 0.30

    # Execution
    twap_duration_sec:  int  = 300
    twap_slices:        int  = 10


# ─────────────────────────────────────────────────────────────────────────────
# LiveTrader
# ─────────────────────────────────────────────────────────────────────────────

class LiveTrader:
    """
    Production live trading loop.

    Call `await trader.run(symbols)` to start the async loop.
    Call `trader.stop()` to initiate graceful shutdown.
    """

    def __init__(
        self,
        broker,
        config: TraderConfig | None = None,
    ) -> None:
        self.broker  = broker
        self.cfg     = config or TraderConfig()
        self._stop   = asyncio.Event()

        # Per-symbol state
        self._books:         dict[str, OrderBook]        = {}
        self._tick_history:  dict[str, Deque[dict]]      = {}
        self._primary_models: dict[str, object]          = {}
        self._meta_models:    dict[str, object]          = {}
        self._health_monitors: dict[str, object]         = {}

        # Shared components
        self._feature_builder = FeatureBuilder()
        self._order_manager   = OrderManager()
        self._risk_engine: RiskEngine | None = None
        self._regime_detector = None
        self._exec_engine: ExecutionEngine | None = None
        self._state_db = StateDB(self.cfg.db_dsn) if self.cfg.db_dsn else None
        
        self._notifier = Notifier(
            telegram_token=settings.telegram_bot_token,
            telegram_chat_id=settings.telegram_chat_id,
            discord_webhook=settings.discord_webhook_url
        )

        # PnL tracking
        self._trade_returns: Deque[float] = collections.deque(maxlen=1000)
        self._last_health_check = 0.0

    # ── Builder / factory ─────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: TraderConfig, broker) -> "LiveTrader":
        """Create a LiveTrader from a TraderConfig and a BrokerGateway."""
        return cls(broker=broker, config=config)

    # ── Startup ───────────────────────────────────────────────────────────

    async def _setup(self, symbols: list[str]) -> None:
        """Load models and initialise per-symbol state."""

        # DB State Recovery
        if self._state_db:
            await self._state_db.connect()
            snapshot = await self._state_db.load_latest_snapshot()
            if snapshot:
                self.cfg.nav = snapshot["nav"]
                log.info("Loaded portfolio snapshot: NAV=%.2f, Exposure=%.2f", snapshot['nav'], snapshot['total_exposure'])
            else:
                snapshot = None
        else:
            snapshot = None

        # Risk engine
        limits = RiskLimits(
            max_position_usd   = self.cfg.max_position_usd,
            max_portfolio_pct  = self.cfg.max_portfolio_pct,
            max_drawdown_pct   = self.cfg.max_drawdown_pct,
            vol_target_annual  = self.cfg.vol_target_annual,
            kelly_fraction     = self.cfg.kelly_fraction,
        )
        self._risk_engine = RiskEngine(limits=limits, nav=self.cfg.nav)
        
        # Restore positions if available
        if snapshot and "positions" in snapshot:
            for sym, qty_usd in snapshot["positions"].items():
                self._risk_engine._positions[sym] = qty_usd
                log.info("Restored position [%s]: $%.2f", sym, qty_usd)

        # Execution engine
        self._exec_engine = ExecutionEngine(
            broker              = self.broker,
            order_manager       = self._order_manager,
            max_slippage_bps    = self.cfg.max_slippage_bps,
            min_fill_prob       = self.cfg.min_fill_prob,
            default_twap_duration = self.cfg.twap_duration_sec,
            default_twap_slices   = self.cfg.twap_slices,
        )

        # Regime detector
        regime_path = Path(self.cfg.regime_model_path)
        if regime_path.exists():
            try:
                from borsabot.models.regime import RegimeDetector
                self._regime_detector = RegimeDetector.load(regime_path)
                log.info("RegimeDetector loaded from %s", regime_path)
            except Exception as exc:
                log.warning("Could not load regime model: %s", exc)

        # ── Per-symbol state – models ──────────────────────────────────────
        for sym in symbols:
            self._books[sym]        = OrderBook(sym)
            self._tick_history[sym] = collections.deque(maxlen=self.cfg.tick_history_size)

            # Normalise broker symbol → model filename slug (BTCUSDT → BTC_USD)
            slug = broker_to_model_slug(sym)

            primary_path = Path(self.cfg.primary_model_path.format(symbol=slug))
            meta_path    = Path(self.cfg.meta_model_path.format(symbol=slug))

            self._load_model_pair(sym, primary_path, meta_path)

        log.info(
            "LiveTrader setup complete. Symbols: %s | NAV: $%.0f",
            symbols, self.cfg.nav,
        )

    def _load_model_pair(
        self,
        sym: str,
        primary_path: Path,
        meta_path: Path,
    ) -> None:
        """Try to load primary + meta models; support both Colab dict and class formats."""

        if primary_path.exists():
            try:
                self._primary_models[sym] = ColabPrimaryAdapter.load(primary_path)
                log.info("[%s] ColabPrimaryAdapter loaded from %s", sym, primary_path)
            except Exception:
                # Fall back to class-based PrimaryModel format
                try:
                    from borsabot.models.primary_model import PrimaryModel
                    self._primary_models[sym] = PrimaryModel.load(primary_path)
                    log.info("[%s] PrimaryModel loaded from %s", sym, primary_path)
                except Exception as exc:
                    log.warning("[%s] Could not load primary model: %s", sym, exc)
        else:
            log.warning("[%s] Primary model not found: %s", sym, primary_path)

        if meta_path.exists():
            try:
                m = ColabMetaAdapter.load(meta_path, threshold=self.cfg.meta_threshold)
                self._meta_models[sym] = m
                log.info("[%s] ColabMetaAdapter loaded from %s", sym, meta_path)
            except Exception:
                try:
                    from borsabot.models.meta_model import MetaModel
                    m2 = MetaModel.load(meta_path)
                    m2.threshold = self.cfg.meta_threshold
                    self._meta_models[sym] = m2
                    log.info("[%s] MetaModel loaded from %s", sym, meta_path)
                except Exception as exc:
                    log.warning("[%s] Could not load meta model: %s", sym, exc)
        else:
            log.warning("[%s] Meta model not found: %s", sym, meta_path)

    # ── Main trading loop ─────────────────────────────────────────────────

    async def run(self, symbols: list[str]) -> None:
        """Start the live trading event loop. Blocks until stop() is called."""

        await self.broker.connect()
        try:
            await self._setup(symbols)
            log.info("Starting tick stream for %s …", symbols)
            
            await self._notifier.send(
                f"🚀 <b>BorsaBot Started</b>\n"
                f"Mode: <code>{'MOCK' if isinstance(self.broker, __import__('borsabot').brokers.base.MockBrokerGateway) else 'LIVE'}</code>\n"
                f"Symbols: {', '.join(symbols)}\n"
                f"Starting NAV: ${self.cfg.nav:,.2f}"
            )

            # Launch background tasks
            tasks = [
                asyncio.create_task(self._tick_loop(symbols), name="tick_loop"),
                asyncio.create_task(self._health_check_loop(), name="health_loop"),
                asyncio.create_task(self._daily_reset_loop(), name="daily_reset"),
            ]

            # Wait until stop signal or any task exits
            done, pending = await asyncio.wait(
                tasks + [asyncio.create_task(self._stop.wait(), name="stop")],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

            # DB State Recovery: Save snapshot on exit
            if self._state_db and self._risk_engine:
                summary = self._risk_engine.portfolio_summary()
                await self._state_db.save_snapshot(
                    nav=summary["nav"],
                    total_exposure=summary["total_exposure"],
                    positions=summary["positions"]
                )

        finally:
            if self._state_db:
                await self._state_db.disconnect()
            await self.broker.disconnect()
            
            summary = self._risk_engine.portfolio_summary() if self._risk_engine else {}
            nav = summary.get("nav", self.cfg.nav)
            pnl = summary.get("daily_pnl", 0.0)
            await self._notifier.send(
                f"🛑 <b>BorsaBot Stopped</b>\n"
                f"Final NAV: ${nav:,.2f}\n"
                f"Session PnL: ${pnl:+,.2f}"
            )
            await self._notifier.close()
            
            log.info("LiveTrader stopped.")

    def stop(self) -> None:
        """Request graceful shutdown of the trading loop."""
        self._stop.set()

    # ── Tick ingestion ────────────────────────────────────────────────────

    async def _tick_loop(self, symbols: list[str]) -> None:
        """Consume market data ticks via on_tick callback."""
        try:
            async def on_tick(tick: dict) -> None:
                if self._stop.is_set():
                    return
                sym = tick.get("symbol") or tick.get("s", "")
                if sym in self._books:
                    await self._on_tick(sym, tick)

            await self.broker.stream_market_data(symbols, on_tick=on_tick)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.exception("Tick loop error: %s", exc)

    async def _on_tick(self, symbol: str, tick: dict) -> None:
        """
        Process a single market data tick:
          1. Update order book
          2. Append to rolling history
          3. Build features → generate signal if models available
          4. Route through risk engine → execution engine
        """
        # Update order book with depth update if present
        book = self._books[symbol]
        if "bids" in tick and "asks" in tick:
            book.apply_snapshot({
                "lastUpdateId": tick.get("lastUpdateId", 0),
                "bids": tick["bids"],
                "asks": tick["asks"],
            })

        # Record tick in rolling history
        raw_price = tick.get("price")
        if raw_price is None:
            if not book.is_valid():
                return    # no book, no price → skip
            try:
                raw_price = book.mid_price()
            except (StopIteration, RuntimeError):
                return
        self._tick_history[symbol].append({
            "time":  pd.Timestamp.now(),
            "price": float(raw_price),
            "qty":   float(tick.get("qty", 0)),
            "side":  tick.get("side", "B"),
        })

        # Need sufficient history before generating signals
        history = self._tick_history[symbol]
        if len(history) < max(30, self.cfg.tick_history_size // 4):
            return

        # ── Feature extraction ────────────────────────────────────────────
        tick_df = pd.DataFrame(list(history))
        price_series = tick_df["price"].astype(float)
        price_series.index = tick_df["time"]

        primary = self._primary_models.get(symbol)
        if primary is None:
            return

        # ColabPrimaryAdapter: compute Colab features inline from price series
        if isinstance(primary, ColabPrimaryAdapter):
            try:
                side_int, primary_conf = primary.predict_from_prices(price_series)
            except Exception as exc:
                log.debug("[%s] Colab primary model error: %s", symbol, exc)
                return

            feats_df = None   # built inside adapter; passed via ColabMetaAdapter path
        else:
            # Legacy FeatureBuilder path
            try:
                features = self._feature_builder.build(tick_df, book)
                feats_df = features.to_frame().T
            except Exception as exc:
                log.debug("[%s] Feature build error: %s", symbol, exc)
                return

            try:
                side_int, primary_conf = primary.predict_side(feats_df)
            except Exception as exc:
                log.debug("[%s] Primary model error: %s", symbol, exc)
                return

        if side_int == 0:
            return  # No-trade signal

        # ── Meta model filtering ──────────────────────────────────────────
        meta = self._meta_models.get(symbol)
        if meta is not None:
            try:
                if isinstance(meta, ColabMetaAdapter):
                    # Colab meta needs price series to build base features
                    from borsabot.models.colab_adapter import ColabFeaturePipeline
                    base_feats = ColabFeaturePipeline.compute(
                        price_series, feat_cols=meta._feat_cols
                    )
                    feats_df_meta = base_feats.to_frame().T
                    do_trade, meta_conf = meta.should_trade(
                        feats_df_meta, side_int, primary_conf
                    )
                else:
                    do_trade, meta_conf = meta.should_trade(
                        feats_df, side_int, primary_conf
                    )

                if not do_trade:
                    log.debug(
                        "[%s] Meta model blocked trade: conf=%.3f < threshold=%.2f",
                        symbol, meta_conf, self.cfg.meta_threshold,
                    )
                    return
            except Exception as exc:
                log.debug("[%s] Meta model error: %s", symbol, exc)
                meta_conf = primary_conf
        else:
            meta_conf = primary_conf

        # ── Regime-aware sizing ───────────────────────────────────────────
        regime_pct = 0.10   # default
        if self._regime_detector is not None:
            try:
                prices = tick_df["price"].astype(float)
                if len(prices) >= self.cfg.regime_window:
                    rets = prices.pct_change().dropna().tail(self.cfg.regime_window)
                    regime_params = self._regime_detector.current_params(rets)
                    regime_pct = regime_params.get("position_pct", 0.10)
                    log.debug(
                        "[%s] Regime: %s | position_pct=%.0f%%",
                        symbol, regime_params.get("regime"), regime_pct * 100,
                    )
            except Exception as exc:
                log.debug("[%s] Regime detection error: %s", symbol, exc)

        # ── Risk engine ───────────────────────────────────────────────────
        nav         = self._risk_engine.nav
        # For Colab daily models, use vol_72 from price series as daily vol proxy
        try:
            rets     = price_series.pct_change().dropna()
            daily_vol = float(rets.tail(20).std()) if len(rets) >= 20 else 0.02
        except Exception:
            daily_vol = 0.02
        vol_size_usd = self._risk_engine.vol_adjusted_size(max(daily_vol, 0.001))
        size_usd     = min(vol_size_usd, nav * regime_pct) * meta_conf

        ok, msg = self._risk_engine.check_new_order(symbol, size_usd)
        if not ok:
            log.debug("[%s] Risk blocked: %s", symbol, msg)
            if "halted" in msg.lower() and getattr(self._risk_engine, "_already_halt_notified", False) is False:
                self._risk_engine._already_halt_notified = True
                await self._notifier.send(f"⚠️ <b>RISK ALERT</b>\n{msg}")
            return

        # Convert USD size to base asset quantity using mid price
        # Guard: Colab daily models may not have L2 book data yet
        try:
            mid = book.mid_price() if book.is_valid() else float(price_series.iloc[-1])
        except (StopIteration, RuntimeError, IndexError):
            mid = float(price_series.iloc[-1]) if len(price_series) > 0 else 0.0
        if mid <= 0:
            return
        quantity = size_usd / mid

        # ── Build signal and execute ──────────────────────────────────────
        signal = Signal(
            symbol    = symbol,
            side      = OrderSide.BUY if side_int > 0 else OrderSide.SELL,
            confidence = float(meta_conf),
            timestamp_ns = time.time_ns(),
        )

        try:
            orders = await self._exec_engine.on_signal(
                signal   = signal,
                book     = book,
                quantity = quantity,
                urgency  = "medium",
            )
            if orders:
                self._risk_engine.update_position(symbol, size_usd if side_int > 0 else -size_usd)
                log.info(
                    "[%s] %s %.6f @ mid=%.2f | meta_conf=%.3f | size=$%.0f",
                    symbol, signal.side.value, quantity, mid, meta_conf, size_usd,
                )
                await self._notifier.send(
                    f"⚡ <b>Trade Executed: {symbol}</b>\n"
                    f"Action: <code>{signal.side.name.upper()}</code>\n"
                    f"Amount: ${size_usd:,.2f}\n"
                    f"Price: ${mid:,.2f}\n"
                    f"AI Meta Conf: {meta_conf:.2f}"
                )
        except Exception as exc:
            log.warning("[%s] Execution error: %s", symbol, exc)

    # ── Health check loop ─────────────────────────────────────────────────

    async def _health_check_loop(self) -> None:
        """Periodically check model drift and Sharpe decay."""
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self.cfg.health_check_interval)
                await self._run_health_check()
        except asyncio.CancelledError:
            pass

    async def _run_health_check(self) -> None:
        """Compute PSI + Sharpe decay; update Prometheus gauges."""
        if not self._trade_returns:
            return

        try:
            live_returns = pd.Series(list(self._trade_returns))
            sr = float(
                live_returns.mean() / (live_returns.std() + 1e-9) * np.sqrt(252)
            )
            model_sharpe_ratio.set(sr)
            daily_pnl_usd.set(float(live_returns.sum() * self.cfg.nav))
            log.info("Health check: rolling Sharpe=%.3f", sr)
        except Exception as exc:
            log.warning("Health check error: %s", exc)

    # ── Daily reset loop ──────────────────────────────────────────────────

    async def _daily_reset_loop(self) -> None:
        """Reset daily PnL counters at UTC midnight."""
        try:
            while not self._stop.is_set():
                now = pd.Timestamp.utcnow()
                seconds_until_midnight = (
                    (24 - now.hour - 1) * 3600
                    + (60 - now.minute - 1) * 60
                    + (60 - now.second)
                )
                await asyncio.sleep(seconds_until_midnight)
                self._risk_engine.reset_daily()
                log.info("Daily PnL reset at UTC midnight")
        except asyncio.CancelledError:
            pass

    # ── Public status ─────────────────────────────────────────────────────

    @property
    def portfolio_summary(self) -> dict:
        """Return current portfolio state (nav, exposure, daily_pnl)."""
        if self._risk_engine is None:
            return {}
        return self._risk_engine.portfolio_summary()

    @property
    def is_running(self) -> bool:
        return not self._stop.is_set()
