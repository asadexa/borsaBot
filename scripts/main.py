"""BorsaBot main live trading process.

Wires all platform components into a single async event loop:

    ConnectBroker
        ↓
    StreamMarketData → OrderBook → FeedMonitor
        ↓
    TickNormalizer → EventBus (PUB)
        ↓
    FeatureBuilder  ← latest book + tick history
        ↓
    RegimeDetector  → adjust strategy params
        ↓
    PrimaryModel.predict_side()
        ↓
    MetaModel.should_trade()  ← confidence filter
        ↓
    RiskEngine.check_new_order()
        ↓
    ExecutionEngine.on_signal()
        ↓
    OrderManager FSM tracking
        ↓
    TimescaleDB / Redis / Prometheus metrics

Usage:
    python scripts/main.py
    python scripts/main.py --broker binance --symbols BTCUSDT ETHUSDT --paper
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


async def main(
    broker_name: str,
    symbols: list[str],
    paper: bool,
    model_dir: str,
    nav: float,
    user_settings: dict | None = None,
) -> None:
    """Main async entry point. Runs until SIGINT/SIGTERM."""

    # ── Setup logging ─────────────────────────────────────────────────────
    from borsabot.core.logging import configure_logging
    configure_logging(level="INFO")

    log.info("BorsaBot starting — broker=%s symbols=%s paper=%s", broker_name, symbols, paper)

    # ── Load config ───────────────────────────────────────────────────────
    from borsabot.config import settings

    # ── Start observability ───────────────────────────────────────────────
    from unittest.mock import AsyncMock, MagicMock

    def _make_mock_cache():
        c = MagicMock()
        c.set_tick     = AsyncMock()
        c.set_features = AsyncMock()
        c.stop         = AsyncMock()
        c.start        = AsyncMock()
        return c

    def _make_mock_server():
        srv = MagicMock()
        srv.stop = AsyncMock()
        return srv

    if not paper:
        try:
            from borsabot.monitoring.metrics import start_metrics_server
            from borsabot.monitoring.health import HealthServer
            start_metrics_server(port=8000)
            # Boş port bul (8080-8090)
            health_server = None
            for _port in range(8080, 8091):
                try:
                    hs = HealthServer(port=_port,
                                      redis_url=settings.redis_url,
                                      timescale_dsn=settings.timescale_dsn)
                    await hs.start()
                    health_server = hs
                    log.info("HealthServer port %d'de basladi", _port)
                    break
                except OSError:
                    continue
            if health_server is None:
                log.warning("HealthServer baslatilamadi (tum portlar dolu) — mock kullaniliyor")
                health_server = _make_mock_server()
        except Exception as exc:
            log.warning("Observability baslatma hatasi: %s — mock kullaniliyor", exc)
            health_server = _make_mock_server()
    else:
        health_server = _make_mock_server()
        log.info("Paper mode: Health server / Prometheus atlandi")

    # ── Connect storage ───────────────────────────────────────────────────
    from borsabot.storage.redis_cache import MarketCache
    from borsabot.storage.timescale import IntegrityWriter
    from borsabot.storage.parquet_lake import ParquetLake

    lake = ParquetLake()

    if paper:
        # Paper modda Redis/DB gerekmez
        cache  = _make_mock_cache()
        writer = None
        log.info("Paper mode: Redis/TimescaleDB atlandi")
    else:
        # Redis baglantisini dene — yoksa mock'a don
        try:
            cache = MarketCache(settings.redis_url)
            await cache.start()
            log.info("Redis baglandi: %s", settings.redis_url)
        except Exception as exc:
            log.warning("Redis yok (%s) — mock cache kullaniliyor", exc)
            cache = _make_mock_cache()

        # TimescaleDB baglantisini dene — yoksa None
        try:
            writer = IntegrityWriter(settings.timescale_dsn)
            await writer.start()
            log.info("TimescaleDB baglandi")
        except Exception as exc:
            log.warning("TimescaleDB yok (%s) — veri kalici depolanmayacak", exc)
            writer = None

    # ── Load broker ───────────────────────────────────────────────────────
    broker = _get_broker(broker_name, settings, paper)
    await broker.connect()
    log.info("Broker connected: %s (paper=%s)", broker_name, paper)

    # ── Create order book + monitoring per symbol ─────────────────────────
    from borsabot.market_data.order_book import OrderBook
    from borsabot.market_data.feed_monitor import FeedMonitor
    from borsabot.market_data.tick_normalizer import TickNormalizer

    books:   dict[str, OrderBook]  = {s: OrderBook(s) for s in symbols}
    monitor = FeedMonitor(max_gap_ms=500.0, stale_feed_ms=5_000.0)

    # ── Load models ───────────────────────────────────────────────────────
    from borsabot.features.builder import FeatureBuilder
    from borsabot.models.regime import RegimeDetector

    feature_builder = FeatureBuilder()
    regime_detector = _load_regime(model_dir)

    # Per-sembol model yükle
    primary_models: dict[str, object] = {}
    meta_models:    dict[str, object] = {}
    for sym in symbols:
        pm = _load_primary(model_dir, [sym])
        mm = _load_meta(model_dir, [sym])
        primary_models[sym] = pm
        meta_models[sym]    = mm
        mt5_sym = sym
        log.info("[%s] primary=%s  meta=%s",
                 mt5_sym,
                 "OK" if pm else "MISSING",
                 "OK" if mm else "MISSING")

    # ── Setup risk engine ─────────────────────────────────────────────────
    from borsabot.risk.engine import RiskEngine, RiskLimits

    risk = RiskEngine(
        limits=RiskLimits(
            max_position_usd=float(os.getenv("MAX_POSITION_USD", 50_000)),
            max_portfolio_pct=float(os.getenv("MAX_PORTFOLIO_PCT", 0.10)),
            max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT", 0.05)),
        ),
        nav=nav,
    )

    # ── Execution engine ──────────────────────────────────────────────────
    from borsabot.execution.engine import ExecutionEngine
    from borsabot.execution.order_fsm import OrderManager

    execution = ExecutionEngine(
        broker=broker, 
        default_twap_slices=1  # 0.01 altı geçersiz lot sorunu (10014) için eklendi
    )

    # ── Per-symbol tick history buffers (rolling 500 ticks) ───────────────
    tick_buffers: dict[str, deque] = {s: deque(maxlen=500) for s in symbols}
    seq_counter: dict[str, int] = {s: 0 for s in symbols}

    # ── State and Cooldown Management ─────────────────────────────────────
    user_settings = user_settings or {}
    RR_RATIO = user_settings.get("RISK_REWARD_RATIO", 2.0)
    SL_ATR_MULT = user_settings.get("SL_ATR_MULTIPLIER", 1.5)
    CONF_MIN = user_settings.get("CONFIDENCE_THRESHOLD", 0.60)
    ADX_MIN = user_settings.get("ADX_MINIMUM", 20.0)
    COOLDOWN_SEC = user_settings.get("COOLDOWN_HOURS", 6.0) * 3600

    last_eval_time: dict[str, float] = {s: 0.0 for s in symbols}
    last_signal_state: dict[str, int] = {s: 0 for s in symbols} # 0=FLAT, 1=BUY, -1=SELL
    last_trade_time: dict[str, float] = {s: 0.0 for s in symbols}

    # ── Signal handler for clean shutdown ─────────────────────────────────
    shutdown_event = asyncio.Event()

    def _shutdown(sig, frame):
        log.warning("Received signal %s — shutting down", sig)
        shutdown_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Background stale-feed watchdog ────────────────────────────────────
    async def _stale_feed_watchdog():
        while not shutdown_event.is_set():
            await asyncio.sleep(5.0)
            alerts = monitor.check_stale()
            for alert in alerts:
                log.error("STALE FEED: %s", alert)

    watchdog_task = asyncio.create_task(_stale_feed_watchdog())

    # ── Market data callback ──────────────────────────────────────────────
    async def on_tick(raw: dict) -> None:
        try:
            await _on_tick(raw)
        except Exception as e:
            log.error("ON_TICK CRASHED", exc_info=True)

    async def _on_tick(raw: dict) -> None:
        """Process a single raw tick from the broker stream."""
        sym = raw.get("symbol", raw.get("s", ""))
        if sym not in symbols:
            return

        seq_counter[sym] += 1
        ts = time.time_ns()

        # ── Feed health ───────────────────────────────────────────────────
        alerts = monitor.record(sym, ts, seq_counter[sym])
        for a in alerts:
            log.warning(a)

        # ── Normalize tick ────────────────────────────────────────────────
        if raw.get("_type") == "depth":
            books[sym].apply_delta(raw)
        else:
            if broker_name == "binance":
                event = TickNormalizer.from_binance_agg_trade(raw)
            elif broker_name == "mt5":
                event = TickNormalizer.from_mt5_tick(raw)
                bid, ask = raw.get("bid"), raw.get("ask")
                if bid and ask:
                    books[sym].apply_snapshot({"bids": [[bid, 1.0]], "asks": [[ask, 1.0]], "lastUpdateId": seq_counter[sym]})
            else:
                event = TickNormalizer.from_ib_tick(raw)
                bid, ask = raw.get("bid"), raw.get("ask")
                if bid and ask:
                    books[sym].apply_snapshot({"bids": [[bid, 1.0]], "asks": [[ask, 1.0]], "lastUpdateId": seq_counter[sym]})

            tick_data = {
                "time":  pd.Timestamp.now(),
                "price": event.payload["price"],
                "qty":   event.payload["qty"],
                "side":  event.payload.get("side", ""),
            }
            tick_buffers[sym].append(tick_data)

            # ── Cache latest tick ─────────────────────────────────────────
            await cache.set_tick(sym, event.payload)

            # ── Write to storage ──────────────────────────────────────────
            tick_row = {**event.payload, "symbol": sym, "sequence_id": seq_counter[sym]}
            lake.write(tick_row)
            if not paper:
                asyncio.create_task(writer.write_tick(tick_row))

        # ── Skip if no book or not enough ticks yet ───────────────────────
        book = books[sym]
        if not book.is_valid():
            return

        # ── Throttle evaluation (evaluate once every 60 seconds) ──────────
        current_time = time.time()
        if current_time - last_eval_time[sym] < 60.0:
            return
        last_eval_time[sym] = current_time

        # ── Fetch D1 Candles & Compute Features ───────────────────────────
        try:
            d1_df = await broker.get_historical_klines(sym, timeframe="1d", limit=60)
            if d1_df is None or len(d1_df) < 30:
                return
            
            # Merge current tick price into the last candle's close
            current_price = book.mid_price()
            d1_df.iloc[-1, d1_df.columns.get_loc('close')] = current_price
            
            # ColabFeaturePipeline
            from borsabot.models.colab_adapter import ColabFeaturePipeline
            primary_model = primary_models.get(sym)
            meta_model    = meta_models.get(sym)
            feat_cols = getattr(primary_model, "_feat_cols", ["fd_close", "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "vol_5", "vol_20", "rsi_14", "atr_14", "macd_hist", "bb_width", "stoch_k", "adx_14", "close_vwap"])
            features_series = ColabFeaturePipeline.compute(d1_df["close"], feat_cols=feat_cols)
            if features_series.empty:
                return
            # compute returns a single Series of features for the latest valid candle
            features_dict = features_series.to_dict()
        except Exception as exc:
            log.warning("D1 Pipeline failed for %s: %s", sym, exc)
            return

        # ── Cache features ────────────────────────────────────────────────
        await cache.set_features(sym, features_dict)

        # ── Regime detection ──────────────────────────────────────────────
        returns = d1_df["close"].pct_change().dropna()
        if regime_detector and len(returns) >= 30:
            if isinstance(regime_detector, dict):
                import numpy as np
                hmm = regime_detector["model"]
                mmap = regime_detector.get("mapping", {0:"low_vol", 1:"trending", 2:"high_vol"})
                obs = np.column_stack([returns.values, np.abs(returns.values), np.sign(returns.values)])
                state = int(hmm.predict(obs)[-1])
                regime = mmap.get(state, "low_vol")
                from borsabot.models.regime import REGIME_PARAMS
                rparams = REGIME_PARAMS.get(regime, REGIME_PARAMS["low_vol"])
            else:
                regime   = regime_detector.predict_regime(returns.values)
                rparams  = regime_detector.get_params(regime)
        else:
            regime = "?"
            rparams = {"position_pct": 0.05}

        # ── Primary model prediction ──────────────────────────────────────
        if primary_model is None:
            return

        try:
            # Colab adapter predicts from DataFrame/Series values
            X_input = features_series.values.reshape(1, -1)
            if isinstance(primary_model, dict):
                m_obj = primary_model["model"]
                RMAP  = primary_model.get("label_rmap", {0:-1,1:0,2:1})
                proba = m_obj.predict_proba(X_input)[0]
                pred  = int(proba.argmax())
                side_int  = RMAP.get(pred, 0)
                confidence  = float(proba.max())
            else:
                side_int, confidence = primary_model.predict_side(X_input)
            
            # Print evaluation so the user knows the AI is alive
            adx_current = float(features_dict.get("adx_14", 0.0))
            if time.time() - last_eval_time[sym] < 61.0: # Only print every ~minute to avoid spam
                log.info(f"[{sym}] AI Eval -> Side: {side_int}, Conf: {confidence:.3f}, ADX: {adx_current:.1f} (Settings: >{CONF_MIN:.2f} / >{ADX_MIN:.1f})")
                
        except Exception as exc:
            log.warning("PrimaryModel predict failed: %s", exc)
            return

        if side_int == 0:
            last_signal_state[sym] = 0 # reset state when flat
            return

        # ── Relaxed A/B Test (Demo Data Collection) ───────────────────────
        adx = float(features_dict.get("adx_14", 0.0))
        if confidence < CONF_MIN:
            return
        if adx != 0.0 and adx < ADX_MIN:
            return
            
        # ── Dynamic SL/TP Calculation (Rule set logic) ────────────────────
        atr_val = float(features_dict.get("atr_14", 0.0050))
        dist = atr_val * SL_ATR_MULT
        current_price = book.mid_price()
        
        if side_int == 1:
            sl_price = current_price - dist
            tp_price = current_price + (dist * RR_RATIO)
        else: # Sell
            sl_price = current_price + dist
            tp_price = current_price - (dist * RR_RATIO)
            
        # Log to file for analysis
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            csv_file = log_dir / "live_trades_datacollection.csv"
            write_header = not csv_file.exists()
            with open(csv_file, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("time_utc,symbol,side,price,conf,adx,spread_bps,regime,vol_20,dayofweek,order_usd,sl_price,tp_price\\n")
                now_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                spread_bps = (book.best_ask() - book.best_bid()) / book.mid_price() * 10000
                f.write(f"{now_str},{sym},{side_int},{book.mid_price():.5f},{confidence:.4f},"
                        f"{adx:.1f},{spread_bps:.1f},{regime},{features_dict.get('vol_20',0.0):.5f},"
                        f"{pd.Timestamp.utcnow().weekday()},{nav * rparams.get('position_pct', 0.05):.1f},"
                        f"{sl_price:.5f},{tp_price:.5f}\\n")
        except Exception as exc:
            log.error("CSV Log failed: %s", exc)

        # ── State Machine & Cooldown Prevention ───────────────────────────
        if side_int == last_signal_state[sym]:
            # Same directional signal already active/triggered previously
            return
            
        if current_time - last_trade_time[sym] < COOLDOWN_SEC:
            log.info("Trade blocked for %s due to %.1fh cooldown.", sym, COOLDOWN_SEC / 3600.0)
            return

        try:
            open_positions = await broker.get_positions()
            if open_positions.get(sym, 0.0) != 0.0:
                log.info("Trade blocked for %s: Open position exists.", sym)
                return
        except Exception as e:
            log.warning("Failed to verify open positions: %s", e)

        # Valid trade transition!
        last_signal_state[sym] = side_int
        last_trade_time[sym] = current_time


        # ── Meta model filter ─────────────────────────────────────────────
        if meta_model is not None:
            try:
                if isinstance(meta_model, dict):
                    import numpy as np
                    m_obj = meta_model["model"]
                    feat_cols = meta_model["feat_cols"]
                    
                    X_dict = dict(features_dict)
                    X_dict["primary_pred"] = float(side_int)
                    X_dict["primary_conf"] = float(confidence)
                    
                    X_arr = np.array([[X_dict.get(c, 0.0) for c in feat_cols]], dtype=float)
                    proba = m_obj.predict_proba(X_arr)[0]
                    
                    meta_conf = float(proba[1]) if len(proba) >= 2 else float(proba.max())
                    should = meta_conf >= 0.55
                else:
                    should, meta_conf = meta_model.should_trade(
                        features_series.to_frame().T, side_int, confidence
                    )
                    
                if not should:
                    log.debug("MetaModel FILTERED signal: conf=%.3f", meta_conf)
                    return
            except Exception as exc:
                log.warning("MetaModel predict failed: %s", exc)
                return

        # ── Position sizing ───────────────────────────────────────────────
        mid   = book.mid_price()
        order_usd = nav * rparams.get("position_pct", 0.05)
        qty   = order_usd / max(mid, 1e-9)
        
        # Scale to MT5 lot sizes if MT5 is active
        # XAUUSD 1 lot = 100 oz, EURUSD 1 lot = 100,000 base
        if "MT5" in str(type(broker)):
            if sym == "XAUUSD":
                qty = qty / 100.0
            else:
                qty = qty / 100000.0
            qty = max(0.01, round(qty, 2))
        else:
            # Crypto fallback or IB
            qty = round(qty, 5)

        # ── Risk check ────────────────────────────────────────────────────
        ok, reason = risk.check_new_order(sym, order_usd)
        if not ok:
            log.warning("Risk BLOCKED: %s", reason)
            return

        # ── Submit to execution engine ────────────────────────────────────
        from borsabot.core.events import Signal, OrderSide
        signal_evt = Signal(
            symbol=sym,
            side=OrderSide.BUY if side_int > 0 else OrderSide.SELL,
            confidence=confidence,
            sl=sl_price,
            tp=tp_price,
        )

        if paper:
            log.info(
                "PAPER SIGNAL: %s %s qty=%.4f conf=%.3f",
                signal_evt.side.value, sym, qty, confidence,
            )
        else:
            orders = await execution.on_signal(signal_evt, book, qty)
            for order in orders:
                log.info("Order: %s %s status=%s", sym, signal_evt.side.value, order.state)

    # ── Start market data streams ─────────────────────────────────────────
    log.info("Streaming market data for: %s", symbols)
    stream_tasks = [
        asyncio.create_task(
            broker.stream_market_data([sym], on_tick),
            name=f"stream_{sym}",
        )
        for sym in symbols
    ]

    # ── Wait for shutdown ─────────────────────────────────────────────────
    await shutdown_event.wait()

    # ── Graceful shutdown ─────────────────────────────────────────────────
    log.info("Shutting down...")
    for task in stream_tasks:
        task.cancel()
    watchdog_task.cancel()
    await asyncio.gather(*stream_tasks, watchdog_task, return_exceptions=True)

    lake.flush()
    await cache.stop()
    if writer is not None:
        await writer.stop()
    await broker.disconnect()
    await health_server.stop()

    log.info("BorsaBot stopped cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
# Broker factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_broker(name: str, settings, paper: bool):
    if name == "mock" or (paper and name not in ("mt5", "ib", "binance")):
        from borsabot.brokers.base import MockBrokerGateway
        return MockBrokerGateway()
    if name == "binance":
        from borsabot.brokers.binance_adapter import BinanceAdapter
        return BinanceAdapter(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            testnet=paper,
        )
    if name == "ib":
        from borsabot.brokers.ib_adapter import IBAdapter
        return IBAdapter(
            host=settings.ib_host,
            port=settings.ib_port,
            client_id=settings.ib_client_id,
        )
    if name == "mt5":
        from borsabot.brokers.mt5_adapter import MT5Adapter
        return MT5Adapter(
            account=settings.mt5_account,
            password=settings.mt5_password,
            server=settings.mt5_server,
        )
    raise ValueError(f"Unknown broker: {name!r}. Choose from: binance, ib, mt5, mock")


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders (graceful: return None if model files don't exist yet)
# ─────────────────────────────────────────────────────────────────────────────

def _load_regime(model_dir: str):
    path = Path(model_dir) / "regime.pkl"
    if not path.exists():
        log.warning("Regime model not found at %s — regime detection disabled", path)
        return None
    from borsabot.models.regime import RegimeDetector
    return RegimeDetector.load(path)


def _find_pkl(model_dir: str, *candidates: str):
    """Adaylar arasında ilk bulunan .pkl dosyasını döndür."""
    from pathlib import Path
    for name in candidates:
        p = Path(model_dir) / name
        if p.exists():
            return p
    return None


def _load_regime(model_dir: str):
    path = _find_pkl(model_dir, "regime.pkl")
    if path is None:
        log.warning("Regime model bulunamadı — rejim tespiti devre dışı")
        return None
    import pickle
    data = pickle.loads(path.read_bytes())
    # Colab formatı: {"model": hmm, "mapping": {...}}
    if isinstance(data, dict) and "model" in data:
        log.info("Regime model yüklendi (Colab formatı): %s", path.name)
        return data          # dict — main.py'de doğrudan kullanılır
    # Eski format: RegimeDetector nesnesi
    log.info("Regime model yüklendi (RegimeDetector): %s", path.name)
    return data


def _load_primary(model_dir: str, symbols: list[str] | None = None):
    # Önce sembol-eşleşmeli adaylar
    candidates = ["primary_model.pkl"]
    for sym in (symbols or []):
        safe = sym.replace("-", "_")
        candidates.insert(0, f"{safe}_primary.pkl")
    path = _find_pkl(model_dir, *candidates)

    # Bulunamazsa model_dir içindeki herhangi *_primary.pkl al
    if path is None:
        fallbacks = sorted(Path(model_dir).glob("*_primary.pkl"))
        path = fallbacks[0] if fallbacks else None

    if path is None:
        log.warning("Primary model bulunamadı — trading devre dışı")
        return None
    import pickle
    data = pickle.loads(path.read_bytes())
    log.info("PrimaryModel yüklendi: %s", path.name)
    return data


def _load_meta(model_dir: str, symbols: list[str] | None = None):
    candidates = ["meta_model.pkl"]
    for sym in (symbols or []):
        safe = sym.replace("-", "_")
        candidates.insert(0, f"{safe}_meta.pkl")
    path = _find_pkl(model_dir, *candidates)

    # Bulunamazsa herhangi *_meta.pkl al
    if path is None:
        fallbacks = sorted(Path(model_dir).glob("*_meta.pkl"))
        path = fallbacks[0] if fallbacks else None

    if path is None:
        log.warning("Meta model bulunamadı — meta-filtre devre dışı")
        return None
    import pickle
    data = pickle.loads(path.read_bytes())
    log.info("MetaModel yüklendi: %s", path.name)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BorsaBot — AI-Native Algorithmic Trading Platform",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--broker", default="mock",
        choices=["binance", "ib", "mt5", "mock"],
        help="Broker adapter to use (default: mock)",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT"],
        help="Trading symbols (default: BTCUSDT)",
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="Paper trading mode — signals logged but no real orders sent",
    )
    parser.add_argument(
        "--model-dir", default="models/",
        dest="model_dir",
        help="Directory containing trained model .pkl files (default: models/)",
    )
    parser.add_argument(
        "--nav", type=float, default=100_000.0,
        help="Net Asset Value in USD for position sizing (default: 100000)",
    )

    args = parser.parse_args()

    print(
        f"+------------------------------------------------------+\n"
        f"|        BorsaBot - AI-Native Trading Platform         |\n"
        f"|  broker={args.broker:<8}  symbols={','.join(args.symbols):<12}  paper={str(args.paper):<5}  |\n"
        f"+------------------------------------------------------+"
    )

    def run_interactive_wizard():
        print("="*50)
        print(" *** BORSABOT - TRADE RULESET (INTERACTIVE SETUP) ***")
        print("="*50)
        
        settings = {
            "RISK_REWARD_RATIO": 2.0,
            "SL_ATR_MULTIPLIER": 1.5,
            "CONFIDENCE_THRESHOLD": 0.60,
            "ADX_MINIMUM": 20.0,
            "COOLDOWN_HOURS": 6.0
        }
        
        print("\n[Current Ruleset Defaults]")
        for k, v in settings.items():
            print(f"  {k} = {v}")
            
        print("\nPress [ENTER] to start bot with these rules, or type 'edit' to change them:")
        try:
            choice = input("> ").strip().lower()
            if choice == 'edit':
                print("\nEnter new values (leave blank to keep current):")
                
                val = input(f"Risk/Reward Ratio [{settings['RISK_REWARD_RATIO']}]: ").strip()
                if val: settings['RISK_REWARD_RATIO'] = float(val)
                
                val = input(f"SL ATR Multiplier [{settings['SL_ATR_MULTIPLIER']}]: ").strip()
                if val: settings['SL_ATR_MULTIPLIER'] = float(val)
                
                val = input(f"Confidence Threshold [{settings['CONFIDENCE_THRESHOLD']}]: ").strip()
                if val: settings['CONFIDENCE_THRESHOLD'] = float(val)
                
                val = input(f"ADX Minimum [{settings['ADX_MINIMUM']}]: ").strip()
                if val: settings['ADX_MINIMUM'] = float(val)
                
                val = input(f"Cooldown Hours [{settings['COOLDOWN_HOURS']}]: ").strip()
                if val: settings['COOLDOWN_HOURS'] = float(val)
                
                print("\n*** New Ruleset Applied: ***")
                for k, v in settings.items():
                    print(f"  {k} = {v}")
        except Exception as e:
            print(f"Interactive setup skipped: {e}")

        print("\n*** Starting Engine... ***\n")
        return settings

    user_settings = run_interactive_wizard()

    asyncio.run(main(
        broker_name=args.broker,
        symbols=args.symbols,
        paper=args.paper,
        model_dir=args.model_dir,
        nav=args.nav,
        user_settings=user_settings,
    ))
