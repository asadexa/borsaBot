import MetaTrader5 as mt5

if not mt5.initialize():
    print("MT5 başlatılamadı.")
    exit()

term_info = mt5.terminal_info()
if term_info!=None:
    print(f"Terminal build: {term_info.build}")
    print(f"Trade allowed by Terminal: {term_info.trade_allowed}")

account_info = mt5.account_info()
if account_info!=None:
    print(f"Trade allowed by Account: {account_info.trade_allowed}")
    print(f"Trade allowed by Account Expert: {account_info.trade_expert}")
    print(f"Margin mode: {account_info.margin_mode}")
    print(f"Server: {account_info.server}")
    print(f"Login: {account_info.login}")
else:
    print("Failed to get account info")

mt5.shutdown()
