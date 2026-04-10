import MetaTrader5 as mt5

if not mt5.initialize():
    print("MT5 başlatılamadı.")
    exit()

term_info = mt5.terminal_info()
if term_info!=None:
    print(f"Terminal build: {term_info.build}")
    print(f"Trade allowed: {term_info.trade_allowed}")
    print(f"Trade api allowed: {term_info.trade_api}")
else:
    print("Failed to get terminal info")

account_info = mt5.account_info()
if account_info!=None:
    print(f"Account trade allowed: {account_info.trade_allowed}")
    print(f"Account trade expert: {account_info.trade_expert}")
else:
    print("Failed to get account info")

mt5.shutdown()
