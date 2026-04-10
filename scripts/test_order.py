import asyncio
import io
import sys
import logging
from borsabot.config import settings
from borsabot.brokers.binance_adapter import BinanceAdapter
from borsabot.core.events import OrderRequest, OrderSide, OrderType

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


logging.basicConfig(level=logging.INFO)

async def main():
    print("Binance Testnet Test Emri Gönderiliyor...")
    
    # Adapter oluştur
    adapter = BinanceAdapter(
        api_key=settings.binance_api_key,
        api_secret=settings.binance_api_secret,
        testnet=True
    )
    
    # Bağlan
    await adapter.connect()
    print("Bağlantı başarılı.")
    
    # Bakiye kontrolü
    balance = await adapter.get_balance()
    print(f"Mevcut Bakiye: {balance}")

    # Küçük bir Market Alım Emri (0.001 BTC)
    req = OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.001
    )
    
    print(f"Emir Gönderiliyor: {req.quantity} {req.symbol} {req.side.name}")
    resp = await adapter.submit_order(req)
    print(f"Emir Sonucu: {resp}")
    
    await adapter.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
