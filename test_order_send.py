import os
from dotenv import load_dotenv
import MetaTrader5 as mt5

load_dotenv()
account = int(os.getenv("MT5_ACCOUNT", 0))
password = os.getenv("MT5_PASSWORD", "")
server = os.getenv("MT5_SERVER", "")

if not mt5.initialize(login=account, password=password, server=server):
    print("MT5 başlatılamadı.")
    exit()

symbol = "EURUSD"
mt5.symbol_select(symbol, True)
info = mt5.symbol_info_tick(symbol)

request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": 0.01,
    "type": mt5.ORDER_TYPE_BUY,
    "price": info.ask,
    "deviation": 20,
    "magic": 20240101,
    "comment": "Test Order",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

result = mt5.order_send(request)
print(f"Result IOC: {result}")
mt5.shutdown()
