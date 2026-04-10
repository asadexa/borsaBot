"""
BorsaBot Terminal Dashboard
Çalıştır: python scripts/dashboard.py --broker mt5 --symbols EURUSD XAUUSD --model-dir models/
"""
import sys, io, os, time, asyncio, pickle, argparse
from pathlib import Path
from collections import deque

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.columns import Columns
from rich import box

console = Console()

# ── Renk yardımcıları ─────────────────────────────────────────────────────────
def _color_signal(sig: str) -> str:
    c = {"BUY": "bold green", "SELL": "bold red",
         "FLAT": "dim white", "HIGH_VOL": "yellow",
         "WIDE_SPREAD": "orange3", "META_BLOCKED": "magenta",
         "NO_MODEL": "red", "INIT": "dim", "SPREAD": "orange3"}.get(sig, "white")
    return f"[{c}]{sig}[/{c}]"

def _color_regime(r: str) -> str:
    c = {"trending": "bold cyan", "low_vol": "bold blue",
         "high_vol": "bold red", "?": "dim"}.get(r, "white")
    return f"[{c}]{r}[/{c}]"

def _color_pnl(v: float) -> str:
    c = "bold green" if v > 0 else ("bold red" if v < 0 else "dim")
    return f"[{c}]{v:+.2f}[/{c}]"


# ── MT5 bağlantısı ────────────────────────────────────────────────────────────
class MT5Live:
    def __init__(self, account, password, server):
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            mt5.initialize()
            mt5.login(account, password, server)
            self._ok = True
        except Exception:
            self._ok = False
            self._mt5 = None

    def account_info(self) -> dict:
        if not self._ok:
            return {}
        info = self._mt5.account_info()
        if info is None: return {}
        return {
            "balance":    info.balance,
            "equity":     info.equity,
            "margin":     info.margin,
            "free_margin":info.margin_free,
            "profit":     info.profit,
            "leverage":   info.leverage,
            "currency":   info.currency,
            "server":     info.server,
            "login":      info.login,
        }

    def positions(self) -> list[dict]:
        if not self._ok: return []
        pos = self._mt5.positions_get()
        if not pos: return []
        return [{"symbol": p.symbol,
                 "side":  "BUY" if p.type == 0 else "SELL",
                 "volume": p.volume,
                 "open_price": p.price_open,
                 "current":    p.price_current,
                 "profit":     p.profit,
                 "sl": p.sl, "tp": p.tp} for p in pos]

    def tick(self, sym: str) -> dict | None:
        if not self._ok: return None
        t = self._mt5.symbol_info_tick(sym)
        if t is None: return None
        return {"bid": t.bid, "ask": t.ask,
                "spread": t.ask - t.bid,
                "time": pd.Timestamp.fromtimestamp(t.time)}

    def model_signal(self, sym: str, primary_models: dict,
                     meta_models: dict, regime_model) -> dict:
        """Anlık fiyat üzerinde model sinyali hesapla."""
        result = {"signal": "?", "conf": 0.0, "regime": "?"}
        if not self._ok:
            return result
        try:
            # Son N günlük veri çek
            rates = self._mt5.copy_rates_from_pos(sym, self._mt5.TIMEFRAME_D1, 0, 60)
            if rates is None or len(rates) < 30:
                with open("logs/dash_debug.txt", "a") as f: f.write(f"[{sym}] Rates missing or < 30\\n")
                return result
            close = pd.Series([r[4] for r in rates])  # close

            # Basit feature seti (Colab pipeline)
            ret = close.pct_change().dropna()
            cur_vol = float(ret.tail(10).std())

            # Regime
            rd = regime_model
            if rd is not None and len(ret) >= 30:
                try:
                    hmm  = rd["model"]   if isinstance(rd, dict) else rd
                    rmap = rd["mapping"] if isinstance(rd, dict) else {}
                    obs  = np.column_stack([ret.values[-30:],
                                            np.abs(ret.values[-30:]),
                                            np.sign(ret.values[-30:])])
                    state  = int(hmm.predict(obs)[-1])
                    result["regime"] = rmap.get(state, "?")
                except Exception:
                    pass

            # Use natively provided abstract wrapper for inference if available
            pm = primary_models.get(sym)
            if pm is None:
                with open("logs/dash_debug.txt", "a") as f: f.write(f"[{sym}] PM is None\\n")
                return result
                
            if isinstance(pm, dict):
                # Fallback for old dictionary format
                from borsabot.models.colab_adapter import ColabFeaturePipeline
                feat_cols = pm.get("feat_cols", ["fd_close", "adx_14", "rsi_14", "vol_20", "atr_14"])
                feats = ColabFeaturePipeline.compute(close, feat_cols=feat_cols)
                if feats.empty:
                    with open("logs/dash_debug.txt", "a") as f: f.write(f"[{sym}] feats empty\\n")
                    return result
                
                # Check for array vs dataframe issues
                try:
                    X_input = feats.values.reshape(1, -1)
                    m_obj = pm["model"]
                    RMAP  = pm.get("label_rmap", {0:-1,1:0,2:1})
                    proba = m_obj.predict_proba(X_input)[0]
                    pred  = int(proba.argmax())
                    side  = RMAP.get(pred, 0)
                    conf  = float(proba.max())
                except Exception as eval_exc:
                    with open("logs/dash_debug.txt", "a") as f: f.write(f"[{sym}] Eval crash: {eval_exc}\\n")
                    return result
            else:
                # Modern format (ColabPrimaryAdapter wrapper object)
                try:
                    side, conf = pm.predict_from_prices(close)
                except Exception as eval_exc:
                    with open("logs/dash_debug.txt", "a") as f: f.write(f"[{sym}] predict_from_prices crash: {eval_exc}\\n")
                    return result

            if result["regime"] == "low_vol" and side != 0:
                side = -side   # mean-reversion

            result["signal"] = {1:"BUY", -1:"SELL", 0:"FLAT"}.get(side, "?")
            result["conf"]   = conf
        except Exception as e:
            result["signal"] = f"ERR"
        return result


# ── Dashboard render ──────────────────────────────────────────────────────────
def build_dashboard(mt5: MT5Live, symbols: list[str],
                    primary_models, meta_models, regime_model,
                    tick_history: dict[str, deque]) -> Layout:
    acct  = mt5.account_info()
    poss  = mt5.positions()
    now   = pd.Timestamp.now(tz="Europe/Istanbul")

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )

    # ── Header ────────────────────────────────────────────────────────────────
    bal  = acct.get("balance", 0)
    eq   = acct.get("equity", 0)
    pnl  = acct.get("profit", 0)
    pnl_color = "green" if pnl >= 0 else "red"
    header_txt = (
        f"[bold cyan]BorsaBot MT5 Dashboard[/bold cyan]  "
        f"[dim]|[/dim]  "
        f"Hesap: [yellow]{acct.get('login','?')}[/yellow]  "
        f"[dim]|[/dim]  "
        f"Bakiye: [bold]{bal:,.2f} {acct.get('currency','')}[/bold]  "
        f"[dim]|[/dim]  "
        f"Equity: [bold]{eq:,.2f}[/bold]  "
        f"[dim]|[/dim]  "
        f"P&L: [{pnl_color}]{pnl:+.2f}[/{pnl_color}]  "
        f"[dim]|[/dim]  "
        f"[dim]{now.strftime('%H:%M:%S %Z')}[/dim]"
    )
    layout["header"].update(Panel(header_txt, style="on grey15"))

    # ── Sol: Canlı Fiyatlar + Sinyal ──────────────────────────────────────────
    price_tbl = Table(title="Canlı Fiyatlar & Model Sinyalleri",
                      box=box.SIMPLE_HEAVY, style="cyan")
    price_tbl.add_column("Sembol",   style="bold white", width=10)
    price_tbl.add_column("Bid",      style="white",      width=10)
    price_tbl.add_column("Ask",      style="white",      width=10)
    price_tbl.add_column("Spread",   style="dim",        width=9)
    price_tbl.add_column("Rejim",    width=12)
    price_tbl.add_column("Sinyal",   width=14)
    price_tbl.add_column("Güven",    width=8)

    for sym in symbols:
        tick = mt5.tick(sym)
        sig  = mt5.model_signal(sym, primary_models, meta_models, regime_model)
        if tick:
            bid  = tick["bid"]
            ask  = tick["ask"]
            spr  = tick["spread"]
            spr_bps = spr / ((bid+ask)/2) * 10_000 if (bid+ask) > 0 else 0
            spr_txt = f"{spr_bps:.1f} bps"
            spr_col = "orange3" if spr_bps > 8 else "green"
        else:
            bid = ask = spr = 0.0
            spr_txt = "-"
            spr_col = "dim"

        price_tbl.add_row(
            sym,
            f"{bid:.5f}" if bid else "-",
            f"{ask:.5f}" if ask else "-",
            f"[{spr_col}]{spr_txt}[/{spr_col}]",
            _color_regime(sig.get("regime", "?")),
            _color_signal(sig.get("signal", "?")),
            f"{sig.get('conf',0):.2f}",
        )

    layout["left"].update(Panel(price_tbl, title="[bold]Piyasa[/bold]"))

    # ── Sağ: Açık Pozisyonlar ─────────────────────────────────────────────────
    pos_tbl = Table(title="Açık Pozisyonlar", box=box.SIMPLE_HEAVY, style="yellow")
    pos_tbl.add_column("Sembol", width=10)
    pos_tbl.add_column("Yön",    width=6)
    pos_tbl.add_column("Lot",    width=7)
    pos_tbl.add_column("Açılış", width=10)
    pos_tbl.add_column("Şimdiki",width=10)
    pos_tbl.add_column("P&L",    width=10)
    pos_tbl.add_column("SL",     width=8)
    pos_tbl.add_column("TP",     width=8)

    if poss:
        for p in poss:
            pnl_c = "green" if p["profit"] >= 0 else "red"
            side_c= "bold green" if p["side"] == "BUY" else "bold red"
            pos_tbl.add_row(
                p["symbol"],
                f"[{side_c}]{p['side']}[/{side_c}]",
                f"{p['volume']:.2f}",
                f"{p['open_price']:.5f}",
                f"{p['current']:.5f}",
                f"[{pnl_c}]{p['profit']:+.2f}[/{pnl_c}]",
                f"{p['sl']:.5f}" if p["sl"] else "-",
                f"{p['tp']:.5f}" if p["tp"] else "-",
            )
    else:
        pos_tbl.add_row("[dim]Açık pozisyon yok[/dim]",
                        "", "", "", "", "", "", "")

    # Kurallar özeti
    rules_txt = Text()
    rules_txt.append("\n📋 Aktif Kurallar\n", style="bold yellow")
    rules_txt.append("  • Spread > limit → ATLA\n", style="dim")
    rules_txt.append("  • high_vol rejim → İŞLEM YOK\n", style="dim")
    rules_txt.append("  • low_vol rejim  → Mean-Reversion\n", style="dim")
    rules_txt.append("  • trending rejim → Trend-Follow\n", style="dim")
    rules_txt.append("  • Meta conf < 0.55 → ATLA\n", style="dim")
    rules_txt.append("  • Risk/işlem: %0.5 NAV\n", style="dim")

    layout["right"].split_column(
        Layout(Panel(pos_tbl, title="[bold]Pozisyonlar[/bold]")),
        Layout(Panel(rules_txt, title="[bold]Rule Set[/bold]"), size=11),
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    footer = (
        f"[dim]Kaldıraç: 1:{acct.get('leverage','?')}  |  "
        f"Margin: {acct.get('margin',0):.2f}  |  "
        f"Free Margin: {acct.get('free_margin',0):.2f}  |  "
        f"Sunucu: {acct.get('server','?')}  |  "
        f"[bold]Ctrl+C[/bold] → çıkış[/dim]"
    )
    layout["footer"].update(Panel(footer, style="on grey15"))

    return layout


# ── Main ──────────────────────────────────────────────────────────────────────
async def run(args):
    from borsabot.config import settings

    # MT5 bağlan
    mt5 = MT5Live(settings.mt5_account, settings.mt5_password, settings.mt5_server)

    # Modelleri yükle
    def load_pkl(path: Path):
        if path.exists():
            return pickle.loads(path.read_bytes())
        return None

    model_dir = Path(args.model_dir)
    primary_models, meta_models = {}, {}
    for sym in args.symbols:
        safe = sym.replace("-","_")
        pm = load_pkl(model_dir / f"{safe}_primary.pkl")
        mm = load_pkl(model_dir / f"{safe}_meta.pkl")
        primary_models[sym] = pm
        meta_models[sym]    = mm
        log.info("[%s] primary=%s meta=%s", sym,
                 "OK" if pm else "MISSING", "OK" if mm else "MISSING")

    # MT5 regime
    regime_path = model_dir / "mt5_regime.pkl"
    if not regime_path.exists():
        regime_path = model_dir / "regime.pkl"
    regime_model = load_pkl(regime_path)

    tick_history = {s: deque(maxlen=200) for s in args.symbols}

    console.print("[bold cyan]BorsaBot Dashboard başlatılıyor...[/bold cyan]")
    console.print(f"Semboller: [yellow]{args.symbols}[/yellow]")
    console.print("Çıkmak için [bold]Ctrl+C[/bold]")
    await asyncio.sleep(1)

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            try:
                layout = build_dashboard(mt5, args.symbols,
                                         primary_models, meta_models,
                                         regime_model, tick_history)
                live.update(layout)
                await asyncio.sleep(1.0)
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Dashboard error: {e}[/red]")
                await asyncio.sleep(2.0)


import logging
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BorsaBot Terminal Dashboard")
    parser.add_argument("--broker",    default="mt5")
    parser.add_argument("--symbols",   nargs="+", default=["EURUSD", "XAUUSD"])
    parser.add_argument("--model-dir", default="models/")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        console.print("\n[bold cyan]Dashboard kapatıldı.[/bold cyan]")
