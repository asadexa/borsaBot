import sys
import time
import MetaTrader5 as mt5

if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    sys.exit()

symbols = ["EURUSD", "XAUUSD"]
for sym in symbols:
    mt5.symbol_select(sym, True)

last_times = {s: 0 for s in symbols}

print("Listening for ticks...")
for i in range(10):
    for sym in symbols:
        tick = mt5.symbol_info_tick(sym)
        if tick and tick.time_msc != last_times[sym]:
            print(f"New Tick {sym}: time={tick.time_msc}, bid={tick.bid}, ask={tick.ask}")
            last_times[sym] = tick.time_msc
    time.sleep(1)

mt5.shutdown()
