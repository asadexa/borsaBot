import MetaTrader5 as mt5
from datetime import datetime, timedelta

if not mt5.initialize():
    print("MT5 başlatılamadı.")
    exit()

print("Mevcut Açık Pozisyonlar:")
positions = mt5.positions_get()
if positions:
    for p in positions:
        print(f"- {p.symbol}: {'Alış' if p.type == mt5.POSITION_TYPE_BUY else 'Satış'} | Hacim: {p.volume} | Kar/Zarar: {p.profit} USD")
else:
    print("  Hiç açık pozisyon yok.")

print("\nBugünün Kapalı İşlemleri:")
today = datetime.now()
start = today.replace(hour=0, minute=0, second=0, microsecond=0)
deals = mt5.history_deals_get(start, today)
if deals:
    for d in deals:
        if d.magic == 20240101:  # Bot'un magic numarası main.py'de 20240101 olarak belirlenmiş
            print(f"- {d.symbol}: Kar/Zarar: {d.profit} USD")
else:
    print("  Bugün bot tarafından gerçekleştirilmiş geçmiş işlem görünmüyor.")

mt5.shutdown()
