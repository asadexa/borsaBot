import asyncio
import io
import sys
import logging
from borsabot.config import settings
from borsabot.brokers.binance_adapter import BinanceAdapter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO)

async def main():
    print("Gerçek (Production) Binance Hesabı Kontrol Ediliyor...")
    
    # Adapter oluştur (testnet=False ile gerçek ağa bağlanıyoruz)
    adapter = BinanceAdapter(
        api_key=settings.binance_api_key,
        api_secret=settings.binance_api_secret,
        testnet=False
    )
    
    # Bağlan
    await adapter.connect()
    
    # Bakiye kontrolü
    try:
        balance = await adapter.get_balance()
        print("\n--- MEVCUT BAKIYELER ---")
        for asset, amount in balance.items():
            if amount > 0:
                print(f"{asset}: {amount}")
        print("------------------------\n")
        print("Bağlantı BAŞARILI. Gerçek bakiye okundu.")
        
    except Exception as e:
        print(f"\nHATA: Bakiye okunamadı veya yetki yok. Detaylar: {e}")
    finally:
        await adapter.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
