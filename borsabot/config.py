"""Application configuration via pydantic-settings (reads from .env)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Binance ───────────────────────────────────────────────────
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True

    # ── Interactive Brokers ───────────────────────────────────────
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1

    # ── MetaTrader 5 ─────────────────────────────────────────────
    mt5_account: int = 0
    mt5_password: str = ""
    mt5_server: str = ""

    # ── TimescaleDB ───────────────────────────────────────────────
    timescale_dsn: str = "postgresql://borsabot:borsabot@localhost:5432/borsabot"

    # ── Redis ─────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── ZeroMQ ────────────────────────────────────────────────────
    zmq_pub_addr: str = "tcp://127.0.0.1:5555"
    zmq_sub_addr: str = "tcp://127.0.0.1:5555"

    # ── Notifications ─────────────────────────────────────────────
    discord_webhook_url: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Trading ───────────────────────────────────────────────────
    default_symbols: str = "BTCUSDT,ETHUSDT"
    nav_usd: float = 100_000.0
    log_level: str = "INFO"

    @property
    def symbols(self) -> list[str]:
        return [s.strip() for s in self.default_symbols.split(",") if s.strip()]


# Singleton — import from anywhere with: from borsabot.config import settings
settings = Settings()
