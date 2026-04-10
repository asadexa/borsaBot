"""
BorsaBot Telegram Bildirimleri
Kurulum:
  1. Telegram'da @BotFather'a /newbot yaz, token al
  2. @userinfobot'a mesaj at, chat_id al
  3. .env'e TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID ekle
  4. Test: python scripts/telegram_notify.py --test

Kullanım (bot içinden):
  from scripts.telegram_notify import TelegramNotifier
  notifier = TelegramNotifier()
  await notifier.send_signal('EURUSD', 'BUY', conf=0.67, adx=28.5, price=1.0850)
"""
import os, asyncio, argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import aiohttp
    _AIOHTTP_OK = True
except ImportError:
    _AIOHTTP_OK = False

# .env yükle
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


class TelegramNotifier:
    """Async Telegram bildirici — bot içinden kullanılır."""

    def __init__(self, token: str = TOKEN, chat_id: str = CHAT_ID):
        self.token   = token
        self.chat_id = chat_id
        self._ok     = bool(token and chat_id and "your_" not in token)

    async def _send(self, text: str) -> bool:
        if not self._ok or not _AIOHTTP_OK:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    return r.status == 200
        except Exception:
            return False

    async def send_signal(self, symbol: str, side: str, conf: float,
                          adx: float, price: float, regime: str = "?",
                          order_usd: float = 0.0) -> None:
        emoji = "🟢" if side == "BUY" else "🔴"
        msg = (
            f"{emoji} <b>BorsaBot Sinyal</b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"📌 Sembol : <b>{symbol}</b>\n"
            f"📊 Yön    : <b>{side}</b>\n"
            f"💰 Fiyat  : {price:.5f}\n"
            f"🎯 Güven  : {conf*100:.1f}%\n"
            f"📈 ADX    : {adx:.1f}\n"
            f"🌊 Rejim  : {regime}\n"
            f"💵 Miktar : ${order_usd:.0f}\n"
            f"━━━━━━━━━━━━━━\n"
            f"⚠️ Bu bir bildirimdir, onay gerekmez."
        )
        await self._send(msg)

    async def send_order_filled(self, symbol: str, side: str,
                                 qty: float, price: float, pnl: float | None = None) -> None:
        emoji = "✅"
        pnl_str = f"\n💰 PnL: {pnl:+.2f}$" if pnl is not None else ""
        msg = (
            f"{emoji} <b>Emir Gerçekleşti</b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"📌 Sembol : {symbol}\n"
            f"📊 Yön    : {side}\n"
            f"📦 Miktar : {qty:.4f}\n"
            f"💲 Fiyat  : {price:.5f}"
            f"{pnl_str}\n"
            f"━━━━━━━━━━━━━━"
        )
        await self._send(msg)

    async def send_risk_alert(self, reason: str, drawdown_pct: float) -> None:
        msg = (
            f"⛔ <b>Risk Uyarısı</b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"📉 Drawdown: {drawdown_pct*100:.1f}%\n"
            f"❌ Sebep: {reason}\n"
            f"━━━━━━━━━━━━━━\n"
            f"Bot işlemleri durdurdu!"
        )
        await self._send(msg)

    async def send_startup(self, symbols: list, nav: float, broker: str) -> None:
        msg = (
            f"🤖 <b>BorsaBot Başladı</b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"🏦 Broker  : {broker}\n"
            f"💼 Semboller: {', '.join(symbols)}\n"
            f"💵 NAV    : ${nav:,.0f}\n"
            f"━━━━━━━━━━━━━━\n"
            f"Filtreler: ADX≥25 | Conf 0.65-0.70 | Paz dışı"
        )
        await self._send(msg)

    async def send_daily_summary(self, trades: int, pnl: float,
                                  win_rate: float, equity: float) -> None:
        emoji = "📈" if pnl >= 0 else "📉"
        msg = (
            f"{emoji} <b>Günlük Özet</b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"📊 İşlem  : {trades}\n"
            f"💰 PnL    : {pnl:+.2f}$\n"
            f"🎯 Win R. : {win_rate*100:.0f}%\n"
            f"💼 Equity : ${equity:,.2f}\n"
            f"━━━━━━━━━━━━━━"
        )
        await self._send(msg)


async def _test():
    """Bağlantı testi."""
    if not TOKEN or "your_" in TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN eksik!")
        print("\nKurulum adımları:")
        print("  1. Telegram'da @BotFather → /newbot → token kopyala")
        print("  2. @userinfobot'a mesaj at → Chat ID al")
        print("  3. .env dosyasına ekle:")
        print("     TELEGRAM_BOT_TOKEN=1234567890:AAFxxxx...")
        print("     TELEGRAM_CHAT_ID=987654321")
        return

    notifier = TelegramNotifier()
    print(f"Token: {TOKEN[:8]}... | Chat: {CHAT_ID}")
    print("Test mesajı gönderiliyor...")
    ok = await notifier._send(
        "🤖 <b>BorsaBot Test</b>\n"
        "━━━━━━━━━━━━━━\n"
        "✅ Telegram bağlantısı başarılı!\n"
        "Bot hazır."
    )
    print("✅ Gönderildi!" if ok else "❌ Gönderilemedi — token/chat_id kontrol et")

    if ok:
        print("\nÖrnek sinyal bildirimi test ediliyor...")
        await notifier.send_signal(
            "EURUSD", "BUY", conf=0.67, adx=27.3,
            price=1.08512, regime="trending", order_usd=15.0
        )
        print("✅ Sinyal bildirimi gönderildi!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        asyncio.run(_test())
    else:
        print("Kullanım: python scripts/telegram_notify.py --test")
