#!/usr/bin/env python
"""BorsaBot Live Trading Runner.

Connects to a broker (real or mock), loads trained models, and runs
the full trading pipeline in a live async event loop.

Usage (mock / paper trading — no keys required):
    python scripts/run_live.py --symbols BTCUSDT ETHUSDT --mock

Usage (Binance testnet):
    python scripts/run_live.py --symbols BTCUSDT --testnet

Usage (Binance production — EXTREME CAUTION):
    python scripts/run_live.py --symbols BTCUSDT --no-testnet

Options:
    --symbols       Symbols to trade (default: BTCUSDT ETHUSDT)
    --mock          Use MockBrokerGateway (no network, no real orders)
    --testnet       Use Binance testnet (requires BINANCE_* env vars)
    --no-testnet    Use Binance production (requires BINANCE_* env vars)
    --nav           Starting NAV in USD (default: 100000)
    --max-pos-usd   Max single position in USD (default: 10000)
    --duration      Runtime in seconds (0 = run forever, default: 0)
    --log-level     Logging level (default: INFO)

Environment variables (required for --testnet / --no-testnet):
    BINANCE_API_KEY     Binance API key
    BINANCE_API_SECRET  Binance API secret
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1254 encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Make sure the project root is on sys.path ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BorsaBot Live Trading Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"],
        help="Symbols to trade (default: BTCUSDT ETHUSDT)",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--mock", action="store_true", default=True,
        help="Use mock broker (paper trading, no real orders) [DEFAULT]",
    )
    mode.add_argument(
        "--testnet", dest="testnet", action="store_true",
        help="Use Binance testnet (requires BINANCE_* env vars)",
    )
    mode.add_argument(
        "--no-testnet", dest="testnet", action="store_false",
        help="Use Binance production — USE WITH EXTREME CAUTION",
    )
    p.add_argument("--nav",        type=float, default=100_000.0, help="Starting NAV in USD")
    p.add_argument("--max-pos-usd", type=float, default=10_000.0,  help="Max single position USD")
    p.add_argument("--max-dd-pct",  type=float, default=0.05,      help="Max daily drawdown (fraction)")
    p.add_argument("--meta-thr",    type=float, default=0.55,      help="Meta-model threshold")
    p.add_argument("--duration",    type=int,   default=0,         help="Run for N seconds (0 = forever)")
    p.add_argument("--metrics-port", type=int,  default=8000,      help="Prometheus /metrics port (default: 8000, 0 = disabled)")
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return p


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def make_broker(args: argparse.Namespace):
    """Build the appropriate BrokerGateway based on CLI args."""
    if args.mock:
        from borsabot.brokers.base import MockBrokerGateway
        return MockBrokerGateway()

    # Binance (testnet or production)
    from borsabot.config import settings
    api_key    = settings.binance_api_key
    api_secret = settings.binance_api_secret
    if not api_key or not api_secret:
        print(
            "ERROR: BINANCE_API_KEY and BINANCE_API_SECRET must be set in the environment or .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    from borsabot.brokers.binance_adapter import BinanceAdapter
    return BinanceAdapter(
        api_key    = api_key,
        api_secret = api_secret,
        testnet    = args.testnet,
    )


async def _run_with_timeout(trader, symbols: list[str], duration: int) -> None:
    """Start trader; optionally stop it after `duration` seconds."""
    task = asyncio.create_task(trader.run(symbols=symbols))

    if duration > 0:
        await asyncio.sleep(duration)
        trader.stop()
        await task
    else:
        await task


async def main() -> int:
    parser = build_arg_parser()
    args   = parser.parse_args()

    # Reconcile the mutually exclusive group defaults
    if args.testnet is not None:
        args.mock = False
    elif not args.mock:
        args.mock = True

    setup_logging(args.log_level)
    log = logging.getLogger("borsabot.run_live")

    # ── Print banner ───────────────────────────────────────────────────────
    mode_label = (
        "MOCK (paper)" if args.mock
        else ("BINANCE TESTNET" if args.testnet else "⚠️  BINANCE PRODUCTION ⚠️")
    )
    print("=" * 62)
    print(f"  BorsaBot Live Trader")
    print(f"  Mode:    {mode_label}")
    print(f"  Symbols: {', '.join(args.symbols)}")
    print(f"  NAV:    ${args.nav:,.0f}")
    print(f"  Max pos: ${args.max_pos_usd:,.0f}  |  Max DD: {args.max_dd_pct*100:.0f}%")
    if args.duration:
        print(f"  Runtime: {args.duration}s")
    else:
        print(f"  Runtime: until Ctrl-C")
    print("=" * 62)

    # ── Build components ───────────────────────────────────────────────────
    broker = make_broker(args)

    from borsabot.config import settings
    from borsabot.core.trader import LiveTrader, TraderConfig
    cfg = TraderConfig(
        nav              = args.nav,
        max_position_usd = args.max_pos_usd,
        max_drawdown_pct = args.max_dd_pct,
        meta_threshold   = args.meta_thr,
        db_dsn           = settings.timescale_dsn,
    )
    trader = LiveTrader.from_config(config=cfg, broker=broker)

    # ── Prometheus HTTP server ───────────────────────────────────
    metrics_port = args.metrics_port
    if metrics_port > 0:
        try:
            from prometheus_client import start_http_server
            start_http_server(metrics_port)
            log.info("Prometheus metrics server started on :%d/metrics", metrics_port)
        except Exception as exc:
            log.warning("Could not start Prometheus server: %s", exc)

    # ── Graceful Ctrl-C handling ───────────────────────────────────────────
    loop = asyncio.get_event_loop()

    def _handle_sigint():
        log.info("SIGINT received — initiating graceful shutdown …")
        trader.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_sigint)
        except NotImplementedError:
            # Windows does not support add_signal_handler for SIGTERM
            pass

    # ── Run ───────────────────────────────────────────────────────────────
    try:
        await _run_with_timeout(trader, args.symbols, args.duration)
    except (KeyboardInterrupt, asyncio.CancelledError):
        trader.stop()

    # ── Final summary ──────────────────────────────────────────────────────
    summary = trader.portfolio_summary
    if summary:
        print("\n" + "-" * 54)
        print("  Final Portfolio Summary")
        print("-" * 54)
        print(f"  NAV:           ${summary.get('nav', 0):>12,.2f}")
        print(f"  Total exposure:${summary.get('total_exposure', 0):>12,.2f}")
        print(f"  Exposure %:     {summary.get('exposure_pct', 0):>10.1f}%")
        print(f"  Daily PnL:     ${summary.get('daily_pnl', 0):>12,.2f}")
        if summary.get("halted"):
            print("  ** Trading was HALTED (drawdown limit reached) **")
        positions = summary.get("positions", {})
        if positions:
            print("  Open Positions:")
            for sym, usd in positions.items():
                print(f"    {sym:<12} ${usd:>10,.2f}")
        print("-" * 54)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
