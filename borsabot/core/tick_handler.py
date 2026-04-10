"""
TickHandler — Broker-agnostic tick processing pipeline.

Responsibilities:
  1. Normalize raw tick (MT5 / Binance / IB)
  2. Execution pre-filters (spread, market hours)
  3. Regime detection (HMM) → strategy switch
  4. per-symbol Primary + Meta model predict
  5. Fixed-fraction risk sizing
  6. Submit to execution engine
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Spread limits per symbol (max bps) ────────────────────────────────────────
MAX_SPREAD_BPS: dict[str, float] = {
    "EURUSD": 5.0, "GBPUSD": 6.0, "USDJPY": 5.0, "USDCHF": 6.0,
    "XAUUSD": 20.0, "USOIL": 30.0, "BTCUSDT": 10.0,
    "default": 15.0,
}

# ── Canlı A/B Veri Toplama Modu (Relaxed) ───────────────────────────────────
# Amaç para kazanmak değil, demo'da gerçek piyasa davranışı ve edge testi.
# A Seti (Strict): Conf 0.65-0.70, ADX≥25, Sal-Cum
# B Seti (Relaxed): Conf 0.60-0.75, ADX≥20, Pazartesi dahil
# Burada Relaxed sınırları kullanılıp her şey CSV'ye loglanacak. Sonra analiz edeceğiz.
CONF_MIN = 0.60   # Genişletildi
CONF_MAX = 0.75   # Genişletildi
ADX_MIN  = 20.0   # Genişletildi
MAX_TRADES_PER_DAY = 10  # Daha fazla işlem verisi toplamak için arttırıldı


class TickHandler:
    """
    Stateful tick processor — one instance shared across all symbols.

    Usage:
        handler = TickHandler(broker_name, symbols, primary_models,
                              meta_models, regime_model, risk, execution,
                              cache, lake, writer, paper, nav)
        # pass handler.on_tick as callback to broker.stream_market_data
    """

    def __init__(
        self,
        broker_name: str,
        symbols: list[str],
        primary_models: dict,
        meta_models: dict,
        regime_model,
        risk,
        execution,
        cache,
        lake,
        writer,
        paper: bool,
        nav: float,
        tick_buffers: dict[str, deque],
        seq_counter: dict[str, int],
        monitor,
        books: dict,
    ) -> None:
        self.broker_name    = broker_name
        self.symbols        = set(symbols)
        self.primary_models = primary_models
        self.meta_models    = meta_models
        self.regime_model   = regime_model
        self.risk           = risk
        self.execution      = execution
        self.cache          = cache
        self.lake           = lake
        self.writer         = writer
        self.paper          = paper
        self.nav            = nav
        self.tick_buffers   = tick_buffers
        self.seq_counter    = seq_counter
        self.monitor        = monitor
        self.books          = books

        # Live state shared with dashboard
        self.live_state: dict = {
            s: {"price": 0.0, "spread_bps": 0.0, "regime": "?",
                "signal": "INIT", "conf": 0.0, "ticks": 0,
                "vol": 0.0, "last_order": "", "adx": 0.0}
            for s in symbols
        }
        # Günlük işlem sayıcısı (sniper modu)
        self._daily_trades: dict[str, int]  = {s: 0 for s in symbols}
        self._last_trade_day: dict[str, int] = {s: -1 for s in symbols}

    # ── Entry point ───────────────────────────────────────────────────────────

    async def on_tick(self, raw: dict) -> None:
        sym = raw.get("symbol", raw.get("s", ""))
        if sym not in self.symbols:
            return

        self.seq_counter[sym] += 1
        ts = time.time_ns()

        # Feed monitoring
        for alert in self.monitor.record(sym, ts, self.seq_counter[sym]):
            log.warning(alert)

        # ── 1. Normalize ──────────────────────────────────────────────────────
        if raw.get("_type") == "depth":
            self.books[sym].apply_delta(raw)
            return

        price, bid, ask, spread_pts = self._normalize(raw)
        if price <= 0:
            return

        tick_data = {
            "time":  pd.Timestamp.now(),
            "price": price, "qty": float(raw.get("qty", 1.0)),
            "side":  raw.get("side", "b"), "bid": bid, "ask": ask,
        }
        self.tick_buffers[sym].append(tick_data)
        self.live_state[sym]["price"] = price
        self.live_state[sym]["ticks"] = self.seq_counter[sym]

        await self.cache.set_tick(sym, tick_data)
        self.lake.write({**tick_data, "symbol": sym, "seq": self.seq_counter[sym]})
        if self.writer:
            asyncio.create_task(self.writer.write_tick(
                {**tick_data, "symbol": sym, "seq": self.seq_counter[sym]}))

        # ── 2. Execution filter: Spread ───────────────────────────────────────
        spread_bps = (spread_pts / price * 10_000) if price > 0 and spread_pts > 0 else 0.0
        self.live_state[sym]["spread_bps"] = spread_bps
        max_bps = MAX_SPREAD_BPS.get(sym, MAX_SPREAD_BPS["default"])
        if spread_bps > max_bps:
            self.live_state[sym]["signal"] = "WIDE_SPREAD"
            return

        # ── 3. Execution filters ────────────────────────────────────────────────
        now_utc = pd.Timestamp.utcnow()

        # 3a. Hafta sonu kapatılış
        if now_utc.weekday() == 5:                          # Cumartesi
            return
        if now_utc.weekday() == 6 and now_utc.hour < 21:   # Pazar gece 21'e kadar
            return

        # 3b. Günlük işlem limiti: sniper modu (max 2/gün)
        today = now_utc.day
        if self._last_trade_day[sym] != today:
            self._daily_trades[sym]    = 0
            self._last_trade_day[sym]  = today
        if self._daily_trades[sym] >= MAX_TRADES_PER_DAY:
            self.live_state[sym]["signal"] = "DAILY_LIMIT"
            return

        # ── 4. Yeterli veri var mı? ───────────────────────────────────────────
        if len(self.tick_buffers[sym]) < 50:
            return

        tick_df = pd.DataFrame(list(self.tick_buffers[sym])).set_index("time")
        returns = tick_df["price"].pct_change().dropna()
        cur_vol = float(returns.tail(20).std()) if len(returns) >= 20 else 0.01
        self.live_state[sym]["vol"] = cur_vol

        # ── 5. Regime tespiti (HMM dict format) ──────────────────────────────
        regime = self._detect_regime(returns)
        self.live_state[sym]["regime"] = regime

        if regime == "high_vol":
            self.live_state[sym]["signal"] = "HIGH_VOL"
            return  # Yüksek volatilite → işlem yok

        pos_pct = 0.08 if regime == "trending" else 0.05

        # ── 6. Primary model predict ──────────────────────────────────────────
        pm = self.primary_models.get(sym)
        if pm is None:
            self.live_state[sym]["signal"] = "NO_MODEL"
            return

        # ADX Filtresi [Relaxed: >= 20.0]
        try:
            from borsabot.models.colab_adapter import ColabFeaturePipeline
            ps   = tick_df["price"].astype(float)
            feat = ColabFeaturePipeline.compute(ps, feat_cols=["adx_14"])
            adx  = float(feat.get("adx_14", feat.iloc[-1] if not feat.empty else 0))
        except Exception:
            adx = 0.0
        self.live_state[sym]["adx"] = adx
        if adx < ADX_MIN:
            self.live_state[sym]["signal"] = f"ADX_LOW({adx:.0f})"
            return

        # Regime → strateji: low_vol = mean-reversion (sinyal ters)
        if regime == "low_vol" and side_int != 0:
            side_int = -side_int

        if side_int == 0:
            self.live_state[sym]["signal"] = "FLAT"
            return

        # ── 7. Meta model filtresi ────────────────────────────────────────────
        mm = self.meta_models.get(sym)
        if mm is not None:
            confidence = self._meta_filter(mm, X, side_int, confidence,
                                           cur_vol, now_utc)
            if confidence < 0:
                self.live_state[sym]["signal"] = "META_BLOCKED"
                return

        # Kalibrasyon band filtresi [OOS: sadece 0.65-0.70 bandı kalibre]
        # 0.70+ güvenilmez (model aşırı iddialı, gerçek WR=%50-57)
        if not (CONF_MIN <= confidence <= CONF_MAX):
            self.live_state[sym]["signal"] = f"CONF_BAND({confidence:.2f})"
            log.debug("[%s] Conf %.3f kalibre band dışında [%.2f-%.2f] — atlandı",
                      sym, confidence, CONF_MIN, CONF_MAX)
            return

        # ── 8. Risk sizing (sabit %0.5 NAV, vol-scaled) ──────────────────────
        # İşlem başı max risk: NAV × 0.5% → $3000 hesapta $15
        risk_pct  = min(pos_pct, 0.005 / max(cur_vol * 100, 0.1))
        order_usd = max(self.nav * risk_pct, 10.0)
        qty       = order_usd / max(price, 1e-9)

        ok, reason = self.risk.check_new_order(sym, order_usd)
        if not ok:
            log.warning("[%s] Risk BLOCKED: %s", sym, reason)
            return

        # ── 9. Emir ──────────────────────────────────────────────────────────
        from borsabot.core.events import Signal, OrderSide
        side_name = "BUY" if side_int > 0 else "SELL"
        self.live_state[sym]["signal"] = side_name
        self.live_state[sym]["conf"]   = confidence
        self.live_state[sym]["last_order"] = f"{side_name} ${order_usd:.0f}"

        log.info("[%s] SIGNAL %s | regime=%-10s | conf=%.3f | $%.1f",
                 sym, side_name, regime, confidence, order_usd)

        self._daily_trades[sym] += 1  # Sniper sayıcı

        # ── A/B Test Live Data Logging ──────────────────────────────────────
        self._log_trade_context(
            sym, side_name, confidence, adx, spread_bps, regime, 
            cur_vol, now_utc.weekday(), order_usd, price
        )

        if not self.paper:
            try:
                signal_evt = Signal(
                    symbol=sym,
                    side=OrderSide.BUY if side_int > 0 else OrderSide.SELL,
                    confidence=confidence,
                )
                orders = await self.execution.on_signal(signal_evt, self.books[sym], qty)
                for order in orders:
                    log.info("[%s] Order: %s status=%s", sym, side_name, order.state)
            except Exception as exc:
                log.warning("[%s] Execution error: %s", sym, exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalize(self, raw: dict) -> tuple[float, float, float, float]:
        """Returns (price, bid, ask, spread_pts)."""
        if self.broker_name == "mt5":
            bid = float(raw.get("bid", raw.get("price", 0)))
            ask = float(raw.get("ask", raw.get("price", 0)))
            price = (bid + ask) / 2 if bid and ask else float(raw.get("price", 0))
            return price, bid, ask, ask - bid
        elif self.broker_name == "binance":
            from borsabot.market_data.tick_normalizer import TickNormalizer
            ev = TickNormalizer.from_binance_agg_trade(raw)
            p  = float(ev.payload["price"])
            return p, p, p, 0.0
        else:
            from borsabot.market_data.tick_normalizer import TickNormalizer
            ev = TickNormalizer.from_ib_tick(raw)
            p  = float(ev.payload["price"])
            return p, p, p, 0.0

    def _detect_regime(self, returns: pd.Series) -> str:
        """HMM dict-format regime detection. Returns: trending/low_vol/high_vol."""
        rd = self.regime_model
        if rd is None or len(returns) < 30:
            return "trending"
        try:
            hmm  = rd["model"]   if isinstance(rd, dict) else rd
            rmap = rd["mapping"] if isinstance(rd, dict) else {}
            obs  = np.column_stack([returns.values[-30:],
                                    np.abs(returns.values[-30:]),
                                    np.sign(returns.values[-30:])])
            state = int(hmm.predict(obs)[-1])
            return rmap.get(state, "trending")
        except Exception:
            return "trending"

    def _primary_predict(self, pm, tick_df: pd.DataFrame):
        """Returns (side_int, confidence, X) or (0, 0, None) on failure."""
        try:
            m_obj     = pm["model"]     if isinstance(pm, dict) else pm
            feat_cols = pm.get("feat_cols") if isinstance(pm, dict) else None
            RMAP      = (pm.get("label_rmap") or {0: -1, 1: 0, 2: 1}) \
                        if isinstance(pm, dict) else {0: -1, 1: 0, 2: 1}

            if feat_cols:
                from borsabot.models.colab_adapter import ColabFeaturePipeline
                ps = tick_df["price"].astype(float)
                feat_row = ColabFeaturePipeline.compute(ps, feat_cols=feat_cols)
                X = feat_row.values.reshape(1, -1)
            else:
                # Legacy FeatureBuilder (gerekirse)
                X = None
                return 0, 0.0, None

            proba      = m_obj.predict_proba(X)[0]
            pred       = int(proba.argmax())
            side_int   = RMAP.get(pred, 0)
            confidence = float(proba.max())
            return side_int, confidence, X
        except Exception as exc:
            log.warning("Primary predict error: %s", exc)
            return 0, 0.0, None

    def _meta_filter(self, mm, X, side_int, confidence, cur_vol, now_utc) -> float:
        """Apply meta model. Returns confidence if passed, -1 if blocked."""
        try:
            mm_obj = mm["model"] if isinstance(mm, dict) else mm
            # Genişletilmiş meta features: + side, conf, vol, hour
            X_meta = np.hstack([X, [[float(side_int), confidence,
                                      cur_vol, now_utc.hour / 24.0]]])
            m_prob = float(mm_obj.predict_proba(X_meta)[0][1])
            if m_prob < 0.55:
                log.debug("Meta blocked: p=%.3f", m_prob)
                return -1.0
            return m_prob
        except Exception as exc:
            log.debug("Meta error: %s — pass-through", exc)
            return confidence  # meta hata verirse orijinal confidence kullan

    def _log_trade_context(self, sym: str, side: str, conf: float, adx: float, 
                           spread: float, regime: str, vol: float, 
                           weekday: int, order_usd: float, price: float) -> None:
        """Her execution sinyalini (strict veya relaxed) veri analizi için logla."""
        try:
            from pathlib import Path
            import os
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            csv_file = log_dir / "live_trades_datacollection.csv"
            
            write_header = not csv_file.exists()
            with open(csv_file, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("time_utc,symbol,side,price,conf,adx,spread_bps,regime,vol_20,dayofweek,order_usd\\n")
                
                now_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{now_str},{sym},{side},{price:.5f},{conf:.4f},{adx:.1f},{spread:.1f},"
                        f"{regime},{vol:.5f},{weekday},{order_usd:.1f}\\n")
        except Exception as e:
            log.error(f"Failed to write trade log: {e}")
