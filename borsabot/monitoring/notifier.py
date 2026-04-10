"""Notifier service for sending alerts to Telegram and Discord."""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore

log = logging.getLogger(__name__)


class Notifier:
    """Async notification service for Telegram and Discord."""

    _session: ClassVar[aiohttp.ClientSession | None] = None

    def __init__(self, telegram_token: str = "", telegram_chat_id: str = "", discord_webhook: str = "") -> None:
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_webhook = discord_webhook
        
        # Determine active channels
        self.use_telegram = bool(self.telegram_token and self.telegram_chat_id)
        self.use_discord = bool(self.discord_webhook)

    async def _get_session(self) -> aiohttp.ClientSession:
        if Notifier._session is None or Notifier._session.closed:
            Notifier._session = aiohttp.ClientSession()
        return Notifier._session

    async def send(self, message: str) -> None:
        """Send a message to all configured notification channels."""
        if not aiohttp:
            log.warning("aiohttp not installed, cannot send notification.")
            return

        tasks = []
        if self.use_telegram:
            tasks.append(self._send_telegram(message))
        if self.use_discord:
            tasks.append(self._send_discord(message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_telegram(self, text: str) -> None:
        try:
            session = await self._get_session()
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }
            async with session.post(url, json=payload, timeout=5.0) as resp:
                if resp.status >= 400:
                    err = await resp.text()
                    log.error("Telegram notification failed: %s", err)
        except Exception as exc:
            log.error("Telegram error: %s", exc)

    async def _send_discord(self, text: str) -> None:
        try:
            session = await self._get_session()
            payload = {"content": text}
            async with session.post(self.discord_webhook, json=payload, timeout=5.0) as resp:
                if resp.status >= 400:
                    err = await resp.text()
                    log.error("Discord notification failed: %s", err)
        except Exception as exc:
            log.error("Discord error: %s", exc)

    async def close(self) -> None:
        """Close global session if exists."""
        if Notifier._session and not Notifier._session.closed:
            await Notifier._session.close()
