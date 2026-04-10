import json
import csv
import logging
from pathlib import Path
from datetime import datetime
import uuid

log = logging.getLogger(__name__)

class LiveLogger:
    """
    Real-Time Trade Logger for Live / Forward Testing.
    Stateful logging: 
      - Trade Open  -> stored in active_trades.json
      - Trade Close -> matched, merged, and flushed to live_trades.csv
    """
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_file = self.log_dir / "active_trades.json"
        self.csv_file = self.log_dir / "live_trades.csv"
        
        self.csv_headers = [
            # Core
            "trade_id", "symbol", "direction", "position_size",
            # Timestamps
            "timestamp_open", "timestamp_close", "holding_time_min",
            # Prices & PnL
            "entry_price", "exit_price", "profit_loss", "profit_loss_pct",
            # Feature Snapshot
            "confidence_score", "adx", "volatility", "spread", "session", "day_of_week"
        ]
        
        # Ensure CSV has headers if it doesn't exist
        if not self.csv_file.exists():
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)
                
    def _load_active(self) -> dict:
        if self.active_file.exists():
            try:
                return json.loads(self.active_file.read_text("utf-8"))
            except Exception as e:
                log.error(f"Error reading active_trades.json: {e}")
        return {}

    def _save_active(self, data: dict):
        self.active_file.write_text(json.dumps(data, indent=2), "utf-8")

    def _get_session(self, dt: datetime) -> str:
        """Simple UTC session identifier"""
        hour = dt.hour
        if 0 <= hour < 8: return "Asia"
        elif 8 <= hour < 16: return "London"
        else: return "NewYork"

    def log_open(self, symbol: str, direction: str, entry_price: float, position_size: float,
                 confidence: float, adx: float, vol: float, spread: float, timestamp: datetime = None) -> str:
        """
        Logs a newly opened trade. Returns a unique trade_id.
        """
        ts = timestamp or datetime.utcnow()
        trade_id = str(uuid.uuid4())
        
        trade_data = {
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "position_size": position_size,
            "timestamp_open": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "entry_price": entry_price,
            "confidence_score": confidence,
            "adx": adx,
            "volatility": vol,
            "spread": spread,
            "session": self._get_session(ts),
            "day_of_week": ts.strftime("%A")
        }
        
        active_trades = self._load_active()
        active_trades[trade_id] = trade_data
        self._save_active(active_trades)
        
        log.info(f"Trade Opened & Logged [ID: {trade_id[:8]}] - {direction} {symbol} @ {entry_price}")
        return trade_id

    def log_close(self, trade_id: str, exit_price: float, profit_loss: float, timestamp: datetime = None) -> bool:
        """
        Matches an open trade by trade_id, adds closing metrics, and flushes to CSV.
        """
        active_trades = self._load_active()
        
        if trade_id not in active_trades:
            log.warning(f"Trade {trade_id} not found in active records!")
            return False
            
        trade = active_trades.pop(trade_id)
        ts_close = timestamp or datetime.utcnow()
        ts_open = datetime.strptime(trade["timestamp_open"], "%Y-%m-%d %H:%M:%S")
        
        # Calculate derived metrics
        holding_time_min = round((ts_close - ts_open).total_seconds() / 60.0, 2)
        direction_mult = 1 if trade["direction"].upper() == "BUY" else -1
        
        # Approximation of PnL % based on un-leveraged price movement
        profit_loss_pct = ((exit_price - trade["entry_price"]) / trade["entry_price"]) * direction_mult * 100.0
        
        trade.update({
            "timestamp_close": ts_close.strftime("%Y-%m-%d %H:%M:%S"),
            "holding_time_min": holding_time_min,
            "exit_price": exit_price,
            "profit_loss": profit_loss,
            "profit_loss_pct": round(profit_loss_pct, 4)
        })
        
        # Format for CSV
        row = [trade.get(col, "") for col in self.csv_headers]
        
        # Append to CSV
        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        # Update JSON State
        self._save_active(active_trades)
        
        log.info(f"Trade Closed & Flushed to CSV [ID: {trade_id[:8]}] - PnL: ${profit_loss:.2f}")
        return True
