import MetaTrader5 as mt5
mt5.initialize()
info = mt5.symbol_info("EURUSD")
print(f"Filling mode bits: {info.filling_mode}")
if info.filling_mode & mt5.SYMBOL_FILLING_FOK: print("FOK supported")
if info.filling_mode & mt5.SYMBOL_FILLING_IOC: print("IOC supported")
mt5.shutdown()
