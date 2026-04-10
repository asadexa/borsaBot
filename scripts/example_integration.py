"""
Example Integration Snippet 
Shows how to inject the LiveLogger into your actual MT5 Bot.
"""

from borsabot.core.events import OrderSide
from scripts.logger import LiveLogger
import time

class MyEngine:
    def __init__(self):
        # 1. Initialize Logger
        self.live_logger = LiveLogger()
        self.active_trade_map = {}  # Map broker_id to our trade_id

    async def execute_trade(self, symbol: str, side: int, current_price: float, conf: float, adx: float, vol: float):
        """
        Triggered when your model says "BUY"
        """
        # ... Your existing code to send MT5 order ...
        position_size = 0.05
        # broker_order_id = mt5.order_send(...)
        broker_order_id = "MT5-12345" 
        
        # 2. Log Trade Open immediately after execution
        dir_str = "BUY" if side == 1 else "SELL"
        
        trade_id = self.live_logger.log_open(
            symbol=symbol,
            direction=dir_str,
            entry_price=current_price,
            position_size=position_size,
            confidence=conf,
            adx=adx,
            vol=vol,
            spread=0.8
        )
        
        # Keep track so we can close it later
        self.active_trade_map[broker_order_id] = trade_id
        
        
    async def on_trade_closed(self, broker_order_id: str, exit_price: float, profit: float):
        """
        Triggered when MT5 reports a position was closed (via StopLoss/TakeProfit or Manual close)
        """
        our_trade_id = self.active_trade_map.get(broker_order_id)
        if not our_trade_id:
            return  # Unknown trade
            
        # 3. Log Trade Close
        self.live_logger.log_close(
            trade_id=our_trade_id,
            exit_price=exit_price,
            profit_loss=profit
        )
        
        del self.active_trade_map[broker_order_id]

