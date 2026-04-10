"""
EURUSD Edge Analizi — Kazanan vs Kaybeden İşlem Koşulları
Çalıştır: python scripts/analyze_edge.py
"""
import sys, io, pickle, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OOS_START  = "2025-01-01"
OOS_END    = "2026-03-20"
FEE_BPS    = 3.0


def download(yf_sym, start, end):
    tk = yf.Ticker(yf_sym)
    df = tk.history(start=start, end=end, interval="1d", auto_adjust=True)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["open","high","low","close","volume"]].sort_index().dropna(subset=["close"])


def features_and_context(df: pd.DataFrame) -> pd.DataFrame:
    """Hem model feature'ları hem analiz bağlamı."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    f = pd.DataFrame(index=df.index)
    ret = c.pct_change()

    # Model feature'ları (EURUSD_primary.pkl feat_cols ile birebir eşleşmeli)
    f["fd_close"]  = ret.rolling(5).mean()           # frac-diff proxy
    f["ret_1"]     = ret.shift(1)
    f["ret_2"]     = ret.shift(2)
    f["ret_3"]     = ret.shift(3)
    f["ret_5"]     = ret.shift(5)
    f["ret_10"]    = ret.shift(10)
    f["vol_5"]     = ret.rolling(5).std()
    f["vol_20"]    = ret.rolling(20).std()
    f["rsi_14"]    = _rsi(c, 14)
    f["atr_14"]    = (h - l).ewm(com=13, min_periods=14).mean() / (c + 1e-9)
    f["macd_hist"] = (c.ewm(span=12).mean() - c.ewm(span=26).mean() -
                      (c.ewm(span=12).mean()-c.ewm(span=26).mean()).ewm(span=9).mean()) / (c+1e-9)
    f["bb_width"]  = 2 * c.rolling(20).std() / (c.rolling(20).mean() + 1e-9)
    f["stoch_k"]   = (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min() + 1e-9)
    f["adx_14"]    = _adx(h, l, c, 14)
    f["close_vwap"]= (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-9)

    # ── Yön filtresi (ADX tek başına yetmez) ──────────────────────────────
    ema50 = c.ewm(span=50, min_periods=50).mean()
    f["above_ema50"] = (c > ema50).astype(int)           # Fiyat EMA50 üstünde mi?
    # Son 5 barın slope'u (lineer regresyon eğimi, normalize)
    slopes = []
    for i in range(len(c)):
        if i < 4:
            slopes.append(np.nan)
        else:
            y = c.iloc[i-4:i+1].values
            x = np.arange(5)
            slope = np.polyfit(x, y, 1)[0] / (y.mean() + 1e-9)
            slopes.append(slope)
    f["slope_5d"] = slopes                               # + = yukarı trend
    f["trend_align"] = (                                 # ADX + yön uyumu
        (f["adx_14"] > 25) &
        (
            ((f["slope_5d"] > 0) & (f["above_ema50"] == 1)) |
            ((f["slope_5d"] < 0) & (f["above_ema50"] == 0))
        )
    ).astype(int)

    # ── Analiz bağlamı ─────────────────────────────────────────────────────
    f["month"]      = df.index.month
    f["dayofweek"]  = df.index.dayofweek   # 0=Pazartesi, 4=Cuma
    f["quarter"]    = df.index.quarter
    f["is_trend"]   = (f["adx_14"] > 25).astype(int)
    f["vol_regime"] = pd.qcut(f["vol_20"].dropna().reindex(f.index),
                               q=3, labels=["low","mid","high"]).astype(str)
    f["ret_5d_fwd"] = ret.shift(-5)

    return f.dropna()



def _rsi(close, n):
    d = close.diff()
    g = d.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(com=n-1, min_periods=n).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))


def _adx(h, l, c, n):
    up    = h.diff().clip(lower=0)
    down  = (-l.diff()).clip(lower=0)
    tr    = (h - l).ewm(com=n-1, min_periods=n).mean()
    dmp   = up.ewm(com=n-1, min_periods=n).mean() / (tr + 1e-9)
    dmn   = down.ewm(com=n-1, min_periods=n).mean() / (tr + 1e-9)
    dx    = (dmp - dmn).abs() / (dmp + dmn + 1e-9)
    return dx.ewm(com=n-1, min_periods=n).mean() * 100


def run_trades(df: pd.DataFrame, ctx: pd.DataFrame, pm, mm) -> pd.DataFrame:
    """Her işlemi kaydet: koşullar + sonuç."""
    RMAP = pm.get("label_rmap", {0:-1,1:0,2:1}) if isinstance(pm,dict) else {0:-1,1:0,2:1}
    m    = pm["model"] if isinstance(pm,dict) else pm
    feat_cols = (pm.get("feat_cols") or []) if isinstance(pm,dict) else []
    # Sadece model'in beklediği ve ctx'te mevcut olan kolonlar
    model_cols = [c for c in feat_cols if c in ctx.columns] if feat_cols else \
                 [c for c in ctx.columns if c not in [
                     "month","quarter","dayofweek","is_trend","vol_regime",
                     "ret_5d_fwd","trend_align","slope_5d","above_ema50"
                 ]]
    if not model_cols:
        print(f"  HATA: model_cols boş! feat_cols={feat_cols[:5]}")
        return pd.DataFrame()
    records = []
    for date in ctx.loc[OOS_START:OOS_END].index:
        row = ctx.loc[date, model_cols]
        X   = row.values.reshape(1, -1)
        try:
            proba = m.predict_proba(X)[0]
            pred  = int(proba.argmax())
            side  = RMAP.get(pred, 0)
            conf  = float(proba.max())
        except Exception:
            continue

        if side == 0:
            continue

        # Meta filter
        if mm is not None:
            try:
                mm_obj = mm["model"] if isinstance(mm,dict) else mm
                vol    = float(ctx.loc[date, "vol_20"])
                X_meta = np.hstack([X, [[float(side), conf, vol, 0.65]]])
                meta_p = float(mm_obj.predict_proba(X_meta)[0][1])
                if meta_p < 0.55:
                    continue
                conf = meta_p
            except Exception:
                pass

        adx        = float(ctx.loc[date, "adx_14"])
        vol20      = float(ctx.loc[date, "vol_20"])
        rsi        = float(ctx.loc[date, "rsi_14"])
        month      = int(ctx.loc[date, "month"])
        dayofweek  = int(ctx.loc[date, "dayofweek"])
        is_tr      = bool(ctx.loc[date, "is_trend"])
        vr         = str(ctx.loc[date, "vol_regime"])
        fwd5       = float(ctx.loc[date, "ret_5d_fwd"])
        ta         = int(ctx.loc[date, "trend_align"])
        slope      = float(ctx.loc[date, "slope_5d"])
        above_ema  = int(ctx.loc[date, "above_ema50"])

        actual_ret = fwd5 * side
        cost = FEE_BPS / 10_000
        pnl  = actual_ret - cost

        records.append({
            "date":       date,
            "side":       "BUY" if side > 0 else "SELL",
            "conf":       round(conf, 3),
            "adx":        round(adx, 1),
            "vol_20":     round(vol20, 5),
            "rsi_14":     round(rsi, 1),
            "month":      month,
            "dayofweek":  dayofweek,
            "is_trend":   is_tr,
            "trend_align":ta,         # ADX>25 + yön uyumu
            "slope_5d":   round(slope, 6),
            "above_ema50":above_ema,
            "vol_regime": vr,
            "fwd_ret5":   round(fwd5, 5),
            "actual_pnl": round(pnl, 5),
            "win":        pnl > 0,
        })

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


def analyze(trades: pd.DataFrame, sym: str):
    if trades.empty:
        print(f"[{sym}] İşlem bulunamadı.")
        return

    wins   = trades[trades["win"]]
    losses = trades[~trades["win"]]
    n = len(trades)
    print(f"\n{'='*65}")
    print(f"  {sym} — Edge Analizi ({n} işlem | "
          f"Win: {len(wins)} ({len(wins)/n*100:.0f}%) | "
          f"Loss: {len(losses)} ({len(losses)/n*100:.0f}%))")
    print(f"{'='*65}")

    # ── Kazanan vs Kaybeden: Ortalama Koşullar ────────────────────────
    compare_cols = ["conf", "adx", "vol_20", "rsi_14"]
    print(f"\n  {'Koşul':<15} {'Kazanan':>10} {'Kaybeden':>10} {'Fark':>10}")
    print(f"  {'-'*50}")
    for col in compare_cols:
        wm = wins[col].mean()   if len(wins)   > 0 else 0
        lm = losses[col].mean() if len(losses) > 0 else 0
        diff = wm - lm
        flag = "✓" if (col in ["conf","adx"] and diff > 0) else \
               "✓" if (col == "vol_20" and diff < 0) else " "
        print(f"  {flag} {col:<14} {wm:>10.4f} {lm:>10.4f} {diff:>+10.4f}")

    # ── Aylık performans ──────────────────────────────────────────────
    print(f"\n  Aylık Performans:")
    for m, grp in trades.groupby("month"):
        wr = grp["win"].mean()
        avg = grp["actual_pnl"].mean() * 100
        bar = "█" * int(wr * 20)
        flag = "✓" if wr > 0.5 else "✗"
        print(f"  {flag} Ay {m:2d}: {bar:<20} WR={wr*100:.0f}% | Avg PnL={avg:+.4f}%")

    # ── Trend vs No-Trend ─────────────────────────────────────────────
    trend_trades    = trades[trades["is_trend"]]
    notrend_trades  = trades[~trades["is_trend"]]
    print(f"\n  Trend (ADX>25) : {len(trend_trades)} işlem | "
          f"WR={trend_trades['win'].mean()*100:.0f}% | "
          f"AvgPnL={trend_trades['actual_pnl'].mean()*100:+.4f}%")
    print(f"  Trend'siz      : {len(notrend_trades)} işlem | "
          f"WR={notrend_trades['win'].mean()*100:.0f}% | "
          f"AvgPnL={notrend_trades['actual_pnl'].mean()*100:+.4f}%")

    # ── Vol rejim ─────────────────────────────────────────────────────
    print(f"\n  Volatilite Rejimi:")
    for vr, grp in trades.groupby("vol_regime"):
        wr  = grp["win"].mean()
        avg = grp["actual_pnl"].mean() * 100
        flag = "✓" if wr > 0.5 else "✗"
        print(f"  {flag} {vr:<8}: {len(grp):3d} işlem | WR={wr*100:.0f}% | "
              f"AvgPnL={avg:+.4f}%")

    # ── Yön Filtresi Analizi (ADX tek başına yetmez) ──────────────────
    print(f"\n  Yön Filtresi Analizi (ADX + Eğim + EMA)")
    for name, mask in [
        ("Tümü               ", pd.Series([True]*len(trades), index=trades.index)),
        ("ADX≥25             ", trades["adx"]>=25),
        ("ADX≥25+Slope+EMA   ", trades["trend_align"]==1),
        ("Slope>0+AboveEMA   ", (trades["slope_5d"]>0)&(trades["above_ema50"]==1)),
    ]:
        sub = trades[mask] if isinstance(mask, pd.Series) and mask.sum()>0 else pd.DataFrame()
        if len(sub) > 2:
            wr  = sub["win"].mean()
            avg = sub["actual_pnl"].mean()*100
            flag= "✅" if wr > 0.5 else "  "
            print(f"  {flag} {name}: n={len(sub):3d} | WR={wr*100:.0f}% | PnL={avg:+.4f}%")

    # ── Calibration Check ─────────────────────────────────────────────
    print(f"\n  Model Kalibrasyon Kontrolü (Conf → Gerçek WR?)")
    print(f"  {'Conf aralığı':<20} {'Beklenen WR':>12} {'Gerçek WR':>12} {'Fark':>10} {'Kalibre mi?'}")
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01]
    labels = ["0.50-0.55","0.55-0.60","0.60-0.65","0.65-0.70","0.70-0.75","0.75+"]
    for i, label in enumerate(labels):
        sub = trades[(trades["conf"] >= bins[i]) & (trades["conf"] < bins[i+1])]
        if len(sub) < 3:
            continue
        expected = (bins[i] + bins[i+1]) / 2
        actual_wr = sub["win"].mean()
        diff = actual_wr - expected
        calib = "✅ İyi" if abs(diff) < 0.10 else ("⬆️ Aşırı güvenli" if diff > 0 else "⬇️ Aşırı iddialı")
        print(f"  {label:<20} {expected*100:>10.0f}% {actual_wr*100:>10.0f}% {diff*100:>+9.0f}%  {calib}")
    print(f"  Not: diff>0 → model alçakgönüllü (iyi), diff<0 → model abartıyor (tehlikeli)")

    # ── Day-of-Week Analizi (session proxy) ───────────────────────────
    if "dayofweek" in trades.columns:
        print(f"\n  Haftanın Günü Analizi (Session Proxy — veriyle doğrulama):")
        days = {0:"Pazartesi",1:"Salı",2:"Çarşamba",3:"Perşembe",4:"Cuma"}
        for d, grp in trades.groupby("dayofweek"):
            wr  = grp["win"].mean()
            avg = grp["actual_pnl"].mean()*100
            bar = "█" * int(wr*20)
            flag= "✅" if wr > 0.5 else "  "
            print(f"  {flag} {days.get(d,'?'):<12}: n={len(grp):3d} "
                  f"| WR={wr*100:.0f}% | AvgPnL={avg:+.4f}%")
        best_days = [d for d,grp in trades.groupby("dayofweek")
                     if len(grp)>2 and grp["win"].mean()>0.5]
        if best_days:
            print(f"  → Kazandıran günler: {[days[d] for d in best_days]}")
            print(f"  → Session için bu günlerde işlem açmayı öneririm")
        else:
            print(f"  → Hiçbir gün tutarlı edge yok — günlük data için session filtresi işe yaramaz")
            print(f"     (Tick verisiyle intraday saate bakılmalı)")

    # ── Sniper kombinasyon (güncellendi: ADX + yön + conf) ────────────

    # ── Grafik ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{sym} — Edge Analizi | Kazanan vs Kaybeden Koşullar",
                 fontsize=13, fontweight="bold")

    colors = {"Kazanan": "#2ECC71", "Kaybeden": "#E74C3C"}

    def hist_compare(ax, col, xlabel, bins=15):
        if len(wins) > 0:
            ax.hist(wins[col].dropna(), bins=bins, alpha=0.7,
                    label=f"Kazanan (n={len(wins)})", color="#2ECC71", density=True)
        if len(losses) > 0:
            ax.hist(losses[col].dropna(), bins=bins, alpha=0.7,
                    label=f"Kaybeden (n={len(losses)})", color="#E74C3C", density=True)
        ax.axvline(wins[col].mean()   if len(wins)>0   else 0, color="#27AE60", ls="--", lw=2)
        ax.axvline(losses[col].mean() if len(losses)>0 else 0, color="#C0392B", ls="--", lw=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Yoğunluk")
        ax.legend(fontsize=8)
        ax.set_title(xlabel)

    hist_compare(axes[0,0], "conf",   "Model Güven Skoru")
    hist_compare(axes[0,1], "adx",    "ADX (Trend Gücü)")
    hist_compare(axes[0,2], "vol_20", "Volatilite (20g)")
    hist_compare(axes[1,0], "rsi_14", "RSI(14)")

    # Aylık win rate
    monthly = trades.groupby("month").agg(wr=("win","mean"), n=("win","count"))
    ax = axes[1,1]
    colors_m = ["#2ECC71" if r > 0.5 else "#E74C3C" for r in monthly["wr"]]
    ax.bar(monthly.index, monthly["wr"]*100, color=colors_m, edgecolor="white")
    ax.axhline(50, color="black", ls="--", lw=1)
    ax.set_xlabel("Ay")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Aylık Win Rate")
    ax.set_ylim(0, 100)

    # Sniper filtre etkisi
    ax = axes[1,2]
    filters = {
        "Tümü":          trades,
        "ADX≥20":        trades[trades["adx"]>=20],
        "ADX≥25":        trades[trades["adx"]>=25],
        "Conf≥0.60":     trades[trades["conf"]>=0.60],
        "ADX≥25+C≥0.60": trades[(trades["adx"]>=25)&(trades["conf"]>=0.60)],
    }
    fnames, fwr, fn = [], [], []
    for name, sub in filters.items():
        if len(sub) > 0:
            fnames.append(name)
            fwr.append(sub["win"].mean()*100)
            fn.append(len(sub))

    fc = ["#2ECC71" if w > 50 else "#E74C3C" for w in fwr]
    bars = ax.bar(range(len(fnames)), fwr, color=fc, edgecolor="white")
    ax.axhline(50, color="black", ls="--", lw=1)
    for i, (bar, count) in enumerate(zip(bars, fn)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"n={count}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(fnames)))
    ax.set_xticklabels(fnames, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Filtre Kombinasyonları Etkisi")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out = Path("results") / f"edge_analysis_{sym}.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafik → {out}")

    return trades


def main():
    print("=" * 65)
    print("  BorsaBot — EURUSD Edge Analizi")
    print(f"  OOS: {OOS_START} → {OOS_END}")
    print("=" * 65)

    model_dir = Path("models")
    pm = pickle.loads((model_dir / "EURUSD_primary.pkl").read_bytes())
    mm = pickle.loads((model_dir / "EURUSD_meta.pkl").read_bytes()) \
         if (model_dir / "EURUSD_meta.pkl").exists() else None

    print("\nEURUSD=X verisi indiriliyor...")
    # Eğitim sonu dahil, feature hesabı için önceki veriye bakıyoruz
    df = download("EURUSD=X", start="2024-10-01", end=OOS_END)
    print(f"{len(df)} bar: {df.index[0].date()} → {df.index[-1].date()}")

    ctx = features_and_context(df)
    print(f"{len(ctx)} satır feature hesaplandı. Sinyaller simüle ediliyor...")

    trades = run_trades(df, ctx, pm, mm)
    print(f"{len(trades)} sinyal üretildi.")

    analyze(trades, "EURUSD")

    # CSV çıktısı
    if not trades.empty:
        out_csv = Path("results") / "eurusd_trades_analysis.csv"
        trades.to_csv(out_csv)
        print(f"\n  CSV  → {out_csv}")

    print("\n" + "=" * 65)
    print("  Analiz tamamlandı!")
    print("  Grafik ve CSV: results/ klasöründe")
    print("=" * 65)


if __name__ == "__main__":
    main()
