import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Styling for dark terminal/presentation vibe
plt.style.use('dark_background')

class AnalyticsEngine:
    """Reads live trade logs and computes real performance & statistical edges."""
    
    def __init__(self, log_dir: str = "logs"):
        self.csv_file = Path(log_dir) / "live_trades.csv"
        
    def load_data(self) -> pd.DataFrame:
        if not self.csv_file.exists():
            print(f"❌ Log file not found at {self.csv_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.csv_file)
        if df.empty:
            print("⚠️ Log file is empty. Running live bot first is required.")
            return df
            
        df['timestamp_open'] = pd.to_datetime(df['timestamp_open'])
        df['timestamp_close'] = pd.to_datetime(df['timestamp_close'])
        df = df.sort_values('timestamp_close').reset_index(drop=True)
        return df

    def run_analysis(self):
        df = self.load_data()
        if df.empty: return

        print("\n" + "="*50)
        print("🚀 LIVE TRADING PERFORMANCE ANALYTICS")
        print("="*50)
        
        # 1. Core Metrics
        total_trades = len(df)
        wins = df[df['profit_loss'] > 0]
        losses = df[df['profit_loss'] <= 0]
        
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = wins['profit_loss'].mean() if not wins.empty else 0
        avg_loss = losses['profit_loss'].mean() if not losses.empty else 0
        profit_factor = abs(wins['profit_loss'].sum() / losses['profit_loss'].sum()) if losses['profit_loss'].sum() != 0 else np.inf
        
        realized_rr = avg_win / abs(avg_loss) if avg_loss != 0 else np.inf
        breakeven_wr = (1 / (1 + realized_rr)) * 100 if realized_rr != np.inf else 0
        
        # Drawdown
        df['cumulative_pnl'] = df['profit_loss'].cumsum()
        df['peak'] = df['cumulative_pnl'].cummax()
        df['drawdown'] = df['cumulative_pnl'] - df['peak']
        max_drawdown = df['drawdown'].min()
        
        print("\n📊 CORE METRICS")
        print(f"Total Trades : {total_trades}")
        print(f"Win Rate     : {win_rate:.1f}% (Gereken: {breakeven_wr:.1f}%)")
        print(f"Avg Win      : ${avg_win:.2f}")
        print(f"Avg Loss     : ${avg_loss:.2f}")
        print(f"Realized R/R : {realized_rr:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown : ${abs(max_drawdown):.2f}")
        
        # 2. Edge Breakdown Analysis
        print("\n🔍 EDGE BREAKDOWN ANALYSIS")
        
        # Confidence Breakdown
        df['conf_bucket'] = pd.cut(df['confidence_score'], bins=[0.0, 0.60, 0.65, 0.70, 1.0], labels=['<0.60', '0.60-0.65', '0.65-0.70', '0.70+'])
        print("\n--- By Confidence Bucket ---")
        print(df.groupby('conf_bucket')['profit_loss'].agg(['count', 'sum', 'mean']).round(2).fillna("-"))
        
        # ADX Breakdown
        df['adx_bucket'] = pd.cut(df['adx'], bins=[0, 20, 25, 30, 100], labels=['<20', '20-25', '25-30', '30+'])
        print("\n--- By ADX Range ---")
        print(df.groupby('adx_bucket')['profit_loss'].agg(['count', 'sum', 'mean']).round(2).fillna("-"))
        
        # Session Breakdown
        print("\n--- By Session ---")
        print(df.groupby('session')['profit_loss'].agg(['count', 'sum', 'mean']).round(2).fillna("-"))
        
        # 3. Statistical Significance (BONUS)
        print("\n⚖️ STATISTICAL SIGNIFICANCE (IS IT LUCK?)")
        if total_trades < 50:
            print(f"⚠️ Minimum Sample Warning: You only have {total_trades} trades.")
            print("Mathematical t-tests require at least 50 live trades to reliably separate edge from luck.")
        else:
            pnl_series = df['profit_loss_pct']
            mean_ret = pnl_series.mean()
            std_ret = pnl_series.std()
            
            if std_ret == 0:
                print("StdDev is zero. Need varied outcomes.")
            else:
                # T-stat compares mean to 0 to see if it's statistically positive
                t_stat = mean_ret / (std_ret / np.sqrt(total_trades))
                pseudo_sharpe = (mean_ret / std_ret) * np.sqrt(total_trades) 
                
                print(f"T-Statistic   : {t_stat:.2f}")
                print(f"Pseudo-Sharpe : {pseudo_sharpe:.2f}")
                
                if t_stat > 2.0 and mean_ret > 0:
                    print("✅ SIGNIFICANT EDGE: The probability of this PnL coming from pure luck is < 5%. Keep trading.")
                elif t_stat < -2.0:
                    print("❌ NEGATIVE EDGE: Your system is statistically losing money. Consider inverting the signals or pausing.")
                else:
                    print("⚖️ INCONCLUSIVE: Cannot reject the hypothesis that your results are random noise. Need more data.")

        # 4. Visualizations
        if total_trades > 0:
            self._plot_metrics(df)

    def _plot_metrics(self, df):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Live Trading Analytics Dashboard", fontsize=16, fontweight='bold')
        
        # Top-Left: Equity Curve
        axes[0, 0].plot(df.index, df['cumulative_pnl'], color='#00ffcc', linewidth=2)
        axes[0, 0].set_title('Cumulative P&L (Equity Curve)')
        axes[0, 0].set_ylabel('$ PnL')
        axes[0, 0].fill_between(df.index, df['cumulative_pnl'], 0, color='#00ffcc', alpha=0.1)
        axes[0, 0].axhline(0, color='white', linestyle='--', alpha=0.3)
        
        # Top-Right: Drawdown
        axes[0, 1].fill_between(df.index, df['drawdown'], 0, color='#ff3366', alpha=0.4)
        axes[0, 1].plot(df.index, df['drawdown'], color='#ff3366', linewidth=1)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('$ Drawdown')
        
        # Bottom-Left: Returns Histogram
        axes[1, 0].hist(df['profit_loss_pct'], bins=20, color='#ffcc00', edgecolor='black')
        axes[1, 0].set_title('Distribution of Trade Returns (%)')
        axes[1, 0].set_xlabel('Return %')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(df['profit_loss_pct'].mean(), color='white', linestyle='dashed', linewidth=2, label='Mean Return')
        axes[1, 0].legend()
        
        # Bottom-Right: PnL by Confidence Bucket
        conf_summary = df.groupby('conf_bucket', observed=False)['profit_loss'].sum()
        conf_summary.plot(kind='bar', ax=axes[1, 1], color=['#bbbbbb', '#ff9900', '#00ffcc', '#00ccff'])
        axes[1, 1].set_title('Total PnL by Confidence Bucket')
        axes[1, 1].set_ylabel('$ PnL')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        out_path = Path("logs") / "performance_dashboard.png"
        plt.savefig(out_path, dpi=150)
        print(f"\n🖼️ Saved visual dashboard to -> {out_path}")
        
        # Pop up window for terminal users
        print("Displaying charts... (Close the window to exit script)")
        plt.show()

if __name__ == "__main__":
    engine = AnalyticsEngine()
    engine.run_analysis()
