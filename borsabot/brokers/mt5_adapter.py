"""MetaTrader 5 adapter for retail forex broker integration."""

from __future__ import annotations

import logging
import time
import uuid
from typing import AsyncIterator

from borsabot.brokers.base import BrokerGateway
from borsabot.core.events import OrderRequest, OrderResponse, OrderSide, OrderStatus, OrderType

log = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    log.warning("MetaTrader5 not installed — MT5Adapter will not function.")


class MT5Adapter(BrokerGateway):
    """
    MetaTrader 5 broker adapter.

    Supports all MT5-compatible forex / CFD brokers.
    Requires MT5 desktop app running on the same machine (Windows only).
    """

    name = "metatrader5"

    def __init__(self, account: int, password: str, server: str) -> None:
        self._account = account
        self._password = password
        self._server = server
        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self) -> None:
        if not _MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package not installed.")

        # MT5 kurulum yolunu bul
        import glob, os
        _candidates = [
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
        ]
        _roaming = glob.glob(
            r"C:\Users\*\AppData\Roaming\MetaQuotes\Terminal\*\terminal64.exe"
        )
        mt5_path = next(
            (p for p in (_candidates + _roaming) if os.path.exists(p)), None
        )

        # initialize() + login tek seferde (IPC bağlantısı için gerekli)
        init_kwargs: dict = {
            "login":    self._account,
            "password": self._password,
            "server":   self._server,
        }
        if mt5_path:
            init_kwargs["path"] = mt5_path

        if not mt5.initialize(**init_kwargs):
            # MT5 yeni açıldıysa IPC oturması için retry
            import time as _time
            for attempt in range(1, 6):
                log.warning("MT5 initialize başarısız (deneme %d/5) — 3 sn bekleniyor... hata: %s",
                            attempt, mt5.last_error())
                _time.sleep(3)
                if mt5.initialize(**init_kwargs):
                    break
            else:
                raise ConnectionError(
                    f"MT5 initialize() 5 denemede başarısız: {mt5.last_error()}\n"
                    f"MT5 masaüstü uygulamasının açık ve giriş yapılmış olduğundan emin olun."
                )
        self._connected = True
        info = mt5.account_info()
        bal  = f"{info.balance:.2f} {info.currency}" if info else "?"
        log.info("MT5Adapter connected  account=%d  server=%s  balance=%s",
                 self._account, self._server, bal)

    async def disconnect(self) -> None:
        if _MT5_AVAILABLE:
            mt5.shutdown()
        self._connected = False
        log.info("MT5Adapter disconnected")

    @property
    def is_connected(self) -> bool:
        if not _MT5_AVAILABLE:
            return False
        info = mt5.account_info()
        return info is not None and self._connected

    # ── Order Management ──────────────────────────────────────────────────

    async def submit_order(self, req: OrderRequest) -> OrderResponse:
        if not _MT5_AVAILABLE:
            raise RuntimeError("MT5 not available")

        tick = mt5.symbol_info_tick(req.symbol)
        if tick is None:
            raise ValueError(f"Symbol {req.symbol!r} not found in MT5")

        order_type = (
            mt5.ORDER_TYPE_BUY if req.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL
        )
        price = tick.ask if req.side == OrderSide.BUY else tick.bid

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    req.symbol,
            "volume":    float(req.quantity),
            "type":      order_type,
            "price":     price,
            "deviation": 20,       # max slippage in points
            "magic":     20240101,
            "comment":   req.client_order_id or str(uuid.uuid4()),
            "type_time": mt5.ORDER_TIME_GTC,
        }
        
        sym_info = mt5.symbol_info(req.symbol)
        if sym_info and getattr(sym_info, "filling_mode", 0) == 1:
            request["type_filling"] = mt5.ORDER_FILLING_FOK
        elif sym_info and getattr(sym_info, "filling_mode", 0) == 2:
            request["type_filling"] = mt5.ORDER_FILLING_IOC
        else:
            request["type_filling"] = mt5.ORDER_FILLING_FOK

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"MT5 order_send failed: retcode={result.retcode if result else 'None'}, comment={result.comment if result else ''}, error={mt5.last_error()}")
            return OrderResponse(
                broker_order_id="",
                client_order_id=req.client_order_id,
                status=OrderStatus.REJECTED,
                filled_qty=0.0,
                avg_price=0.0,
            )

        return OrderResponse(
            broker_order_id=str(result.order),
            client_order_id=req.client_order_id,
            status=OrderStatus.FILLED,
            filled_qty=float(result.volume),
            avg_price=float(result.price),
            timestamp_ns=time.time_ns(),
        )

    async def cancel_order(self, broker_order_id: str) -> bool:
        if not _MT5_AVAILABLE:
            return False
        request = {
            "action":  mt5.TRADE_ACTION_REMOVE,
            "order":   int(broker_order_id),
        }
        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    async def get_order_status(self, broker_order_id: str) -> OrderResponse:
        orders = mt5.orders_get(ticket=int(broker_order_id))
        if not orders:
            history = mt5.history_orders_get(ticket=int(broker_order_id))
            if history:
                o = history[0]
                return OrderResponse(
                    broker_order_id=str(o.ticket),
                    client_order_id="",
                    status=OrderStatus.FILLED,
                    filled_qty=o.volume_current,
                    avg_price=o.price_current,
                )
        raise ValueError(f"MT5 order {broker_order_id} not found")

    async def get_positions(self) -> dict[str, float]:
        if not _MT5_AVAILABLE:
            return {}
        positions = mt5.positions_get()
        if positions is None:
            return {}
        result: dict[str, float] = {}
        for p in positions:
            direction = 1.0 if p.type == mt5.POSITION_TYPE_BUY else -1.0
            result[p.symbol] = p.volume * direction
        return result

    async def get_balance(self) -> dict[str, float]:
        if not _MT5_AVAILABLE:
            return {}
        info = mt5.account_info()
        if info is None:
            return {}
        return {
            "balance":  info.balance,
            "equity":   info.equity,
            "margin":   info.margin,
            "free_margin": info.margin_free,
        }

    async def get_historical_klines(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> "pd.DataFrame":
        import pandas as pd
        if not _MT5_AVAILABLE:
            return pd.DataFrame()
            
        tf_map = {
            "1m": mt5.TIMEFRAME_M1, "5m": mt5.TIMEFRAME_M5, "15m": mt5.TIMEFRAME_M15,
            "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4, "1d": mt5.TIMEFRAME_D1,
        }
        tf = tf_map.get(timeframe.lower(), mt5.TIMEFRAME_D1)
        
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, limit)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

    async def stream_market_data(
        self,
        symbols: list[str],
        on_tick=None,
    ) -> None:
        """
        MT5 tick polling — 50ms aralıkla tüm sembolleri sorgular.
        MT5 WebSocket push sunmaz; polling en stabil yöntemdir.
        """
        import asyncio

        if not _MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 paketi kurulu değil.")

        # Tüm semboller için Market Watch'a ekle / etkinleştir
        for sym in symbols:
            if not mt5.symbol_select(sym, True):
                log.warning("MT5: %s sembolü etkinleştirilemedi — atlanıyor", sym)

        log.info("MT5Adapter: tick polling başladı (%s) @ 50ms", symbols)
        last_ts: dict[str, int] = {s: 0 for s in symbols}

        while True:
            for sym in symbols:
                tick = mt5.symbol_info_tick(sym)
                if tick is None:
                    continue
                # Aynı tick'i iki kez gönderme
                if tick.time_msc == last_ts[sym]:
                    continue
                last_ts[sym] = tick.time_msc

                mid = (tick.bid + tick.ask) / 2
                spread = tick.ask - tick.bid

                payload = {
                    "symbol":  sym,
                    "s":       sym,
                    "bid":     tick.bid,
                    "ask":     tick.ask,
                    "price":   round(mid, 6),
                    "spread":  round(spread, 6),
                    "qty":     1.0,               # MT5 tick'te volume yoktur
                    "side":    "b",               # bid/ask'tan çıkarılamaz
                    "ts":      tick.time_msc * 1_000_000,  # ms → ns
                    "_flags":  tick.flags,
                }
                if on_tick is not None:
                    await on_tick(payload)

            await asyncio.sleep(0.05)   # 20 tick/saniye polling

