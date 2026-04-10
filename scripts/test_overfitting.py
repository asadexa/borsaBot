"""
Overfitting Testi: Kuralları hem 2023-2024 (Validation) hem 2025-2026 (OOS) döneminde sına.
Eğer bir kural OOS'ta çalışıp Validation'da batıyorsa -> OVERFIT'tir, çöpe atılır.
"""
import sys, io, pickle, warnings
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from scripts.analyze_edge import features_and_context

def evaluate_period(ctx: pd.DataFrame, pm, mm, name: str):
    print(f"\n{'='*60}")
    print(f"  Dönem: {name} ({ctx.index[0].date()} -> {ctx.index[-1].date()})")
    print(f"{'='*60}")
    
    RMAP = pm.get("label_rmap", {0:-1,1:0,2:1})
    m = pm["model"]
    feat_cols = pm.get("feat_cols", [])
    model_cols = [c for c in feat_cols if c in ctx.columns]
    
    records = []
    # DataFrame üzerinde vectorised hesaplama hız için daha iyi ama for loop ile yapalım
    # predict_proba'yı toplu yapalım:
    X_all = ctx[model_cols].values
    probas = m.predict_proba(X_all)
    preds = probas.argmax(axis=1)
    confs = probas.max(axis=1)
    
    for i, date in enumerate(ctx.index):
        pred = preds[i]
        conf = confs[i]
        side = RMAP.get(pred, 0)
        
        if side == 0:
            continue
            
        fwd5  = float(ctx.loc[date, "ret_5d_fwd"])
        adx   = float(ctx.loc[date, "adx_14"])
        day   = int(ctx.loc[date, "dayofweek"])
        
        actual_ret = fwd5 * side
        pnl = actual_ret - (3.0 / 10_000) # 3 bps komisyon/spread
        
        records.append({
            "date": date,
            "side": side,
            "conf": conf,
            "adx": adx,
            "dayofweek": day,
            "pnl": pnl,
            "win": pnl > 0
        })
        
    df = pd.DataFrame(records)
    if df.empty:
        print("  İşlem yok!")
        return df
        
    def stat(mask, label):
        sub = df[mask]
        n = len(sub)
        if n == 0:
            print(f"  {label:<25} : n=  0")
            return
        wr = sub['win'].mean() * 100
        avg_pnl = sub['pnl'].mean() * 100
        pf = sub[sub['pnl']>0]['pnl'].sum() / abs(sub[sub['pnl']<0]['pnl'].sum() + 1e-9)
        flag = "✅" if wr > 52 and avg_pnl > 0 else "❌"
        print(f"  {flag} {label:<23} | n={n:3d} | WR={wr:4.1f}% | AvgPnL={avg_pnl:+.3f}% | PF={pf:.2f}")

    stat(pd.Series([True]*len(df), index=df.index), "1. Tümü (Baseline)")
    stat(df['dayofweek'] != 0, "2. Pazartesi Hariç")
    stat(df['adx'] >= 25, "3. Sadece ADX >= 25")
    
    # Conf filter tests (Overfitting test!)
    print("\n  --- Confidence Kalibrasyon Testi ---")
    stat((df['conf'] >= 0.55), "Conf >= 0.55")
    stat((df['conf'] >= 0.60), "Conf >= 0.60")
    stat((df['conf'] >= 0.65) & (df['conf'] <= 0.70), "Conf 0.65-0.70 (Eski)")
    stat((df['conf'] >= 0.70), "Conf >= 0.70")
    
    print("\n  --- Sniper Kombinasyonları ---")
    stat((df['adx'] >= 25) & (df['dayofweek'] != 0), "ADX>=25 + Paz.Haric")
    stat((df['adx'] >= 20) & (df['conf'] >= 0.60), "ADX>=20 + Conf>=0.60")
    
    return df

def main():
    print("Veri indiriliyor...")
    tk = yf.Ticker("EURUSD=X")
    df = tk.history(start="2022-10-01", end="2026-03-20", interval="1d", auto_adjust=True)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    
    ctx = features_and_context(df)
    
    pm = pickle.loads(Path("models/EURUSD_primary.pkl").read_bytes())
    
    # Dönemleri ayır (2023-2024 = Validasyon, 2025-2026 = OOS/Test)
    ctx_val = ctx.loc["2023-01-01":"2024-12-31"]
    ctx_oos = ctx.loc["2025-01-01":"2026-03-20"]
    
    evaluate_period(ctx_val, pm, None, "2023-2024 (Validation/Kalibrasyon)")
    evaluate_period(ctx_oos, pm, None, "2025-2026 (Out-of-Sample)")
    
if __name__ == "__main__":
    main()
