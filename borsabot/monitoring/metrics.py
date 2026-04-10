"""Prometheus metrics registry.

All metrics are defined here as module-level singletons.
Import individual metrics from application code:

    from borsabot.monitoring.metrics import order_latency_ms, orders_submitted
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# ── Order metrics ─────────────────────────────────────────────────────────────
orders_submitted = Counter(
    "borsabot_orders_submitted_total",
    "Total orders submitted to broker",
    ["broker", "symbol"],
)

orders_filled = Counter(
    "borsabot_orders_filled_total",
    "Total orders filled",
    ["broker", "symbol"],
)

orders_rejected = Counter(
    "borsabot_orders_rejected_total",
    "Total orders rejected",
    ["broker", "symbol"],
)

order_latency_ms = Histogram(
    "borsabot_order_latency_ms",
    "Order round-trip latency in milliseconds",
    ["broker"],
    buckets=[5, 10, 25, 50, 100, 200, 500, 1000, 2000],
)

# ── Market data metrics ───────────────────────────────────────────────────────
market_feed_lag_ms = Gauge(
    "borsabot_market_feed_lag_ms",
    "Time since last market data event (ms)",
    ["symbol"],
)

feed_gap_count = Counter(
    "borsabot_feed_gap_total",
    "Number of market data feed gaps detected",
    ["symbol"],
)

# ── Position & PnL metrics ────────────────────────────────────────────────────
position_size_usd = Gauge(
    "borsabot_position_size_usd",
    "Current absolute position size in USD",
    ["symbol"],
)

daily_pnl_usd = Gauge(
    "borsabot_daily_pnl_usd",
    "Realized daily PnL in USD",
)

# ── Model metrics ─────────────────────────────────────────────────────────────
model_sharpe_ratio = Gauge(
    "borsabot_model_sharpe_ratio",
    "Rolling 30-day live Sharpe ratio",
)

model_psi_max = Gauge(
    "borsabot_model_psi_max",
    "Maximum PSI (feature drift) across all features",
)

# ── System metrics ────────────────────────────────────────────────────────────
event_processing_latency_ms = Histogram(
    "borsabot_event_processing_latency_ms",
    "Latency from market data event to feature computation",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 25, 50],
)


def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus HTTP scrape endpoint."""
    start_http_server(port)
