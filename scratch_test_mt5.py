import sys
import MetaTrader5 as mt5

if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    sys.exit()

symbols = ["EURUSD", "XAUUSD"]
for sym in symbols:
    selected = mt5.symbol_select(sym, True)
    print(f"Selected {sym}: {selected}")
    tick = mt5.symbol_info_tick(sym)
    if tick:
        print(f"Tick {sym}: time={tick.time}, bid={tick.bid}, ask={tick.ask}")
    else:
        print(f"Tick {sym}: None")

mt5.shutdown()
