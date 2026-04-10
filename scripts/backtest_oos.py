"""
BorsaBot — Walk-Forward Out-of-Sample Backtest
Overfitting'den kaçınma yöntemleri:
  1. Model 2018-2024'te eğitildi → test 2025'te (tamamen OOS)
  2. Embargo: eğitim/test sınırında 5 gün boşluk
  3. Triple Barrier: gelecek bilgisi kullanılmaz (sıralı işlenir)
  4. PSR (Probabilistic Sharpe Ratio) raporlanır
  5. Walk-Forward: Çoklu OOS penceresi → tek yol değil

Çalıştır:
  python scripts/backtest_oos.py --symbols EURUSD XAUUSD --model-dir models/
"""
import sys, io, argparse, pickle, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── MT5 sembol → Yahoo Finance eşlemesi ──────────────────────────────────────
MT5_TO_YF = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAUUSD": "GC=F",
    "USOIL":  "CL=F",
    "BTCUSDT":"BTC-USD",
    "ETHUSDT":"ETH-USD",
    "BTC_USD":"BTC-USD",
    "ETH_USD":"ETH-USD",
}

# ── Sabitler ──────────────────────────────────────────────────────────────────
TRAIN_END   = "2024-12-31"   # Bu tarihten sonrası OOS
OOS_START   = "2025-01-01"
OOS_END     = "2026-03-20"   # Bugün
EMBARGO_DAYS = 5             # Eğitim/test sınırında boşluk
FEE_BPS     = 3.0            # Forex spread + komisyon
MAX_SPREAD_BPS = {"EURUSD": 5, "XAUUSD": 20, "default": 15}
META_THRESHOLD = 0.55

# Walk-Forward pencereleri (aylık)
WF_TRAIN_MONTHS = 12  # Her pencere: 12 ay eğitim bağlamı
WF_TEST_MONTHS  = 3   # Her pencere: 3 ay test


def download(yf_sym: str, start: str, end: str) -> pd.DataFrame:
    tk = yf.Ticker(yf_sym)
    df = tk.history(start=start, end=end, interval="1d", auto_adjust=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[~df.index.duplicated()].sort_index().dropna(subset=["close"])


def compute_features(close: pd.Series) -> pd.DataFrame:
    """Colab ile aynı feature pipeline (ColabFeaturePipeline olmadan)."""
    f = pd.DataFrame(index=close.index)
    ret = close.pct_change()

    # Frac-diff (d=0.4, basitleştirilmiş)
    f["fd_close"] = close.pct_change().rolling(5).mean()

    for lag in [1, 2, 3, 5, 10]:
        f[f"ret_{lag}"] = ret.shift(lag)
    f["vol_5"]  = ret.rolling(5).std()
    f["vol_20"] = ret.rolling(20).std()

    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    hl  = close.rolling(14).max() - close.rolling(14).min()
    f["atr_14"] = hl / (close + 1e-9)

    ema12 = close.ewm(span=12, min_periods=12).mean()
    ema26 = close.ewm(span=26, min_periods=26).mean()
    f["macd_hist"] = (ema12 - ema26 - (ema12 - ema26).ewm(span=9).mean()) / (close + 1e-9)

    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    f["bb_width"] = 2 * std20 / (ma20 + 1e-9)

    low14  = close.rolling(14).min()
    high14 = close.rolling(14).max()
    f["stoch_k"] = (close - low14) / (high14 - low14 + 1e-9)

    up   = close.diff().clip(lower=0)
    down = (-close.diff()).clip(lower=0)
    tr14 = hl.ewm(com=13, min_periods=14).mean()
    dmp  = up.ewm(com=13, min_periods=14).mean() / (tr14 + 1e-9)
    dmn  = down.ewm(com=13, min_periods=14).mean() / (tr14 + 1e-9)
    dx   = (dmp - dmn).abs() / (dmp + dmn + 1e-9)
    f["adx_14"] = dx.ewm(com=13, min_periods=14).mean()

    f["close_vwap"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9)
    return f.dropna()


def predict_signal(pm, mm, feat_row: np.ndarray, cur_vol: float, hour_norm: float):
    """Primary + Meta predict. Returns (side_int, confidence)."""
    RMAP = pm.get("label_rmap", {0:-1, 1:0, 2:1}) if isinstance(pm, dict) else {0:-1,1:0,2:1}
    m    = pm["model"] if isinstance(pm, dict) else pm
    X    = feat_row.reshape(1, -1)

    proba  = m.predict_proba(X)[0]
    pred   = int(proba.argmax())
    side   = RMAP.get(pred, 0)
    conf   = float(proba.max())

    if mm is not None:
        try:
            mm_obj = mm["model"] if isinstance(mm, dict) else mm
            X_meta = np.hstack([X, [[float(side), conf, cur_vol, hour_norm]]])
            p_correct = float(mm_obj.predict_proba(X_meta)[0][1])
            if p_correct < META_THRESHOLD:
                return 0, 0.0
            conf = p_correct
        except Exception:
            pass

    return side, conf


def simulate(signals, closes, fee_bps, risk_frac=0.005):
    """
    Basit pozisyon simülasyonu.
    risk_frac  = işlem başı NAV yüzdesi (%0.5)
    """
    cap = 100_000.0
    pos = 0
    equity = [cap]
    rets   = []
    cost   = fee_bps / 10_000

    for i in range(1, len(signals)):
        pp, pn = closes[i-1], closes[i]
        # Mevcut pozisyon getirisi
        if pos != 0 and pp > 0:
            cap *= (1 + (pn - pp) / pp * pos)

        # Sinyal değişti → emir
        new_pos = int(signals[i])
        if new_pos != pos:
            cap *= (1 - abs(new_pos - pos) * cost)
            pos = new_pos

        equity.append(cap)
        rets.append(cap / equity[-2] - 1 if equity[-2] > 0 else 0.0)

    eq = np.array(equity)
    r  = np.array(rets)
    sr = r.mean() / (r.std() + 1e-9) * np.sqrt(252)
    rm = np.maximum.accumulate(eq)
    mdd = float(((eq - rm) / (rm + 1e-9)).min())
    wr  = float((r > 0).mean()) if len(r) > 0 else 0.0
    return {"sharpe": float(sr), "mdd": mdd, "equity": eq,
            "returns": r, "winrate": wr}


def walk_forward(close: pd.Series, pm, mm, mt5_sym: str):
    """
    Walk-Forward OOS test:
      - Her WF_TEST_MONTHS uzunluğunda OOS penceresi
      - Embargo: pencere başında EMBARGO_DAYS gün atla
      - Regime: HMM yok, vol-based basit rejim
    """
    feat_df = compute_features(close)
    feat_cols = [c for c in feat_df.columns if c in (
        pm.get("feat_cols", feat_df.columns.tolist()) if isinstance(pm,dict) else feat_df.columns.tolist()
    )]
    feat_df = feat_df[feat_cols].dropna()
    close   = close.reindex(feat_df.index).dropna()

    results = []
    oos_start = pd.Timestamp(OOS_START)
    oos_end   = pd.Timestamp(OOS_END)

    current = oos_start
    while current < oos_end:
        window_end = current + pd.DateOffset(months=WF_TEST_MONTHS)

        # Embargo: pencere başını atla
        emb_start = current + pd.Timedelta(days=EMBARGO_DAYS)
        test_idx  = feat_df.loc[emb_start:window_end].index
        if len(test_idx) < 10:
            current = window_end
            continue

        signals = []
        closes_w = []

        for date in test_idx:
            if date not in feat_df.index:
                continue
            X_row   = feat_df.loc[date].values
            cur_vol = float(feat_df["vol_20"].loc[:date].tail(1).values[0]) \
                      if "vol_20" in feat_df.columns else 0.01
            side, conf = predict_signal(pm, mm, X_row, cur_vol, 12/24.0)
            signals.append(side)
            closes_w.append(float(close.loc[date]))

        if len(signals) < 2:
            current = window_end
            continue

        res = simulate(np.array(signals), np.array(closes_w), FEE_BPS)
        res["period"] = f"{emb_start.strftime('%Y-%m')} → {window_end.strftime('%Y-%m')}"
        res["n_bars"] = len(signals)
        res["n_trades"] = int(np.sum(np.diff(signals) != 0))
        results.append(res)

        current = window_end

    return results


def psr(sharpes: np.ndarray) -> float:
    """Probabilistic Sharpe Ratio."""
    mu  = sharpes.mean()
    std = sharpes.std() + 1e-9
    return float(norm.cdf(mu / std))


def report(sym: str, all_results: list[dict]):
    if not all_results:
        print(f"\n[{sym}] Yeterli OOS verisi yok.")
        return

    sharpes = np.array([r["sharpe"] for r in all_results])
    mdds    = np.array([r["mdd"]    for r in all_results])
    wrs     = np.array([r["winrate"] for r in all_results])
    PSR     = psr(sharpes)

    print(f"\n{'='*60}")
    print(f"  {sym} — Walk-Forward OOS Backtest Sonuçları")
    print(f"  ({len(all_results)} pencere | {sum(r['n_bars'] for r in all_results)} gün)")
    print(f"{'='*60}")
    print(f"  {'Pencere':<30} {'Sharpe':>7} {'MDD':>8} {'WinRate':>8} {'İşlem':>7}")
    print(f"  {'-'*60}")
    for r in all_results:
        flag = "✓" if r["sharpe"] > 0 else "✗"
        print(f"  {flag} {r['period']:<28} {r['sharpe']:+7.2f} "
              f"{r['mdd']*100:7.1f}% {r['winrate']*100:7.1f}% {r['n_trades']:>7}")
    print(f"  {'─'*60}")
    print(f"  {'ORTALAMA':<30} {sharpes.mean():+7.2f} "
          f"{mdds.mean()*100:7.1f}% {wrs.mean()*100:7.1f}%")
    print(f"\n  PSR   : {PSR:.3f}  {'✅ Kabul (>0.55)' if PSR>0.55 else '⚠️ Zayıf (<0.55)'}")
    print(f"  Sharpe: {sharpes.mean():+.3f} ± {sharpes.std():.3f}")
    print(f"  Max MDD: {mdds.min()*100:.1f}%")

    # Grafik
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"{sym} | Walk-Forward OOS Backtest\n"
                 f"PSR={PSR:.2f} | Sharpe={sharpes.mean():+.2f}±{sharpes.std():.2f} | Max MDD={mdds.min()*100:.1f}%",
                 fontsize=12, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])   # Equity (tam genişlik)
    ax2 = fig.add_subplot(gs[1, 0])   # Sharpe histogram
    ax3 = fig.add_subplot(gs[1, 1])   # MDD bar

    # Birleşik equity curve (pencereler arka arkaya)
    eq_all, base = [], 100_000.0
    tick_labels, tick_positions = [], []
    offset = 0
    for r in all_results:
        eq      = r["equity"] / r["equity"][0] * base
        base    = float(eq[-1])
        pos_start = offset
        eq_all.extend(eq.tolist())
        tick_positions.append(pos_start)
        tick_labels.append(r["period"].split(" →")[0])
        offset += len(eq)

    eq_all_arr = np.array(eq_all)
    color_arr = ["#2ECC71" if v >= eq_all_arr[max(0,i-1)] else "#E74C3C"
                 for i, v in enumerate(eq_all_arr)]
    ax1.plot(eq_all_arr, color="#3498DB", lw=1.5, alpha=0.9, label="OOS Equity")
    rm = np.maximum.accumulate(eq_all_arr)
    ax1.fill_between(range(len(eq_all_arr)),
                     eq_all_arr, rm, alpha=0.2, color="red", label="Drawdown")
    ax1.axhline(100_000, color="gray", ls="--", lw=0.8)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax1.set_title("Birleşik OOS Equity Curve", fontsize=10)
    ax1.set_ylabel("Portföy Değeri ($)")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Sharpe histogram
    colors_sh = ["#2ECC71" if s > 0 else "#E74C3C" for s in sharpes]
    ax2.bar(range(len(sharpes)), sharpes, color=colors_sh, edgecolor="white")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axhline(sharpes.mean(), color="blue", ls="--", lw=1.2,
                label=f"Ort: {sharpes.mean():+.2f}")
    ax2.set_title("Pencere Başına Sharpe", fontsize=10)
    ax2.set_xlabel("Pencere")
    ax2.set_ylabel("Sharpe (Yıllık)")
    ax2.legend()

    # MDD bar
    colors_mdd = ["#E74C3C" if m < -0.05 else "#F39C12" for m in mdds]
    ax3.bar(range(len(mdds)), mdds * 100, color=colors_mdd, edgecolor="white")
    ax3.axhline(-5, color="red", ls="--", lw=1, label="Risk limiti: -5%")
    ax3.set_title("Pencere Başına Max Drawdown", fontsize=10)
    ax3.set_xlabel("Pencere")
    ax3.set_ylabel("MDD (%)")
    ax3.legend()

    plt.tight_layout()
    out = Path("results") / f"oos_{sym}_wf.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Grafik  → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",   nargs="+", default=["EURUSD", "XAUUSD"])
    parser.add_argument("--model-dir", default="models/")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    def load(path):
        if path.exists():
            return pickle.loads(path.read_bytes())
        return None

    print("=" * 60)
    print("  BorsaBot OOS Walk-Forward Backtest")
    print(f"  OOS dönemi : {OOS_START} → {OOS_END}")
    print(f"  WF pencere : {WF_TEST_MONTHS} ay test | Embargo: {EMBARGO_DAYS} gün")
    print(f"  Risk/işlem : %0.5 NAV | Fee: {FEE_BPS} bps")
    print("=" * 60)

    for sym in args.symbols:
        safe = sym.replace("-", "_")
        pm   = load(model_dir / f"{safe}_primary.pkl")
        mm   = load(model_dir / f"{safe}_meta.pkl")

        if pm is None:
            print(f"\n[{sym}] Primary model bulunamadı — atlanıyor.")
            continue

        yf_sym = MT5_TO_YF.get(sym, sym)
        print(f"\n[{sym}] Yahoo Finance'den veri çekiliyor ({yf_sym})...")

        # Hem eğitim sonu hem OOS indir (feature hesabı için önceki veriye bakıyor)
        df = download(yf_sym,
                      start=(pd.Timestamp(TRAIN_END) - pd.DateOffset(months=3)).strftime("%Y-%m-%d"),
                      end=OOS_END)
        if df.empty or len(df) < 30:
            print(f"[{sym}] Yeterli veri yok.")
            continue

        close = df["close"]
        print(f"[{sym}] {len(df):,} bar | {df.index[0].date()} → {df.index[-1].date()}")
        print(f"[{sym}] Walk-Forward OOS başlatılıyor...")

        results = walk_forward(close, pm, mm, sym)
        report(sym, results)

    print("\n" + "=" * 60)
    print("  OOS Backtest tamamlandı!")
    print("  Grafikler: results/ klasöründe")
    print("=" * 60)


if __name__ == "__main__":
    main()
