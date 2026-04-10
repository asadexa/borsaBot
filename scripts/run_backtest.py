"""BorsaBot CPCV Backtest Runner.

Gerçek eğitilmiş Colab modelleriyle Combinatorial Purged Cross-Validation
(CPCV) backtest çalıştırır ve Sharpe / PSR / MaxDD / HitRate gibi
metrikleri raporlar.

Kullanım:
    python scripts/run_backtest.py --symbol BTC-USD --period 2y
    python scripts/run_backtest.py --symbol ETH-USD --n-groups 8 --n-test 2
    python scripts/run_backtest.py --symbol BTC-USD --save-results
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BorsaBot CPCV Backtest Runner")
    p.add_argument("--symbol",    default="BTC-USD",  help="yfinance symbol (default: BTC-USD)")
    p.add_argument("--period",    default="2y",       help="yfinance period (default: 2y)")
    p.add_argument("--interval",  default="1d",       help="yfinance interval (default: 1d)")
    p.add_argument("--n-groups",  type=int, default=6, help="CPCV groups (default: 6)")
    p.add_argument("--n-test",    type=int, default=2, help="CPCV test groups (default: 2)")
    p.add_argument("--embargo",   type=float, default=0.02, help="Embargo fraction (default: 0.02)")
    p.add_argument("--threshold", type=float, default=0.55, help="Meta threshold (default: 0.55)")
    p.add_argument("--tx-cost",   type=float, default=0.001, help="Transaction cost (default: 0.001)")
    p.add_argument("--save-results", action="store_true", help="Save results to CSV")
    return p.parse_args()


def download_data(symbol: str, period: str, interval: str):
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance")
        sys.exit(1)

    print(f"Downloading {symbol} ({period}, {interval}) ...")
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        print(f"ERROR: No data for {symbol}")
        sys.exit(1)
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    print(f"  {len(df)} bars  ({df.index[0].date()} - {df.index[-1].date()})")
    return df


def build_features(df):
    import pandas as pd
    from borsabot.models.colab_adapter import ColabFeaturePipeline

    prices = df["Close"]
    volume = df["Volume"]
    rows = []
    for i in range(25, len(df)):
        try:
            rows.append((df.index[i], ColabFeaturePipeline.compute(prices.iloc[:i+1], volume.iloc[:i+1])))
        except Exception:
            pass

    if not rows:
        print("ERROR: Feature building failed")
        sys.exit(1)

    features = pd.DataFrame({col: [r[1][col] for r in rows] for col in rows[0][1].index},
                             index=[r[0] for r in rows])
    print(f"  Features: {features.shape[0]} x {features.shape[1]}")
    return features


def compute_labels(df, features):
    import numpy as np
    close   = df["Close"].reindex(features.index)
    fwd_ret = close.shift(-1) / close - 1
    labels  = np.sign(fwd_ret).fillna(0).astype(int)
    return labels[:-1]    # drop last (no forward return)


def run_cpcv(features, labels, primary_path, meta_path, n_groups, n_test, embargo, threshold, tx_cost):
    import numpy as np
    import pandas as pd
    from borsabot.models.colab_adapter import ColabPrimaryAdapter, ColabMetaAdapter, ColabFeaturePipeline
    from borsabot.backtest.cpcv import CPCV

    feat = features.iloc[:-1]    # align with labels
    lab  = labels.reindex(feat.index)

    primary = ColabPrimaryAdapter.load(primary_path)
    meta    = ColabMetaAdapter.load(meta_path, threshold=threshold)

    print(f"\nCPCV: groups={n_groups}, test={n_test}, embargo={embargo:.1%}, samples={len(feat)}")

    # Build events DataFrame that CPCV expects (index=bar_time, t1=label_end_time)
    idx = feat.index
    t1  = list(idx[1:]) + [idx[-1]]
    events_df = pd.DataFrame({"t1": t1}, index=idx)

    cpcv = CPCV(n_groups=n_groups, n_test_groups=n_test, embargo_pct=embargo)
    results = []

    for split_n, (train_idx, test_idx) in enumerate(cpcv.split(feat, events_df), 1):
        X_test = feat.iloc[test_idx]
        y_test = lab.iloc[test_idx].values

        preds, meta_flags = [], []
        for i in range(len(X_test)):
            row = X_test.iloc[[i]]
            try:
                side, pconf = primary.predict_side(row)
            except Exception:
                side, pconf = 0, 0.5

            if side == 0:
                preds.append(0); meta_flags.append(False); continue

            try:
                do_trade, _ = meta.should_trade(row, side, pconf)
            except Exception:
                do_trade = True

            preds.append(side); meta_flags.append(do_trade)

        preds = np.array(preds)
        mask  = np.array(meta_flags) & (preds != 0)

        if len(preds) == 0 or not mask.any():
            print(f"  Split {split_n:2d}:  [no trades — skipping]")
            results.append({"split": split_n, "n_test": len(test_idx), "trades": 0,
                            "cov%": 0.0, "hit%": 0.0, "net_ret": 0.0,
                            "sharpe": 0.0, "max_dd%": 0.0})
            continue

        pos_ret = (preds * y_test).astype(float)
        fil_ret = np.where(mask, pos_ret, 0.0)
        n_tr    = int(mask.sum())
        net_ret = fil_ret.sum() - n_tr * tx_cost

        ret_tr  = fil_ret[mask] if mask.any() else np.array([0.0])
        mu, sig = ret_tr.mean(), ret_tr.std() or 1e-9
        sharpe  = mu / sig * np.sqrt(252) if len(ret_tr) > 1 else 0.0

        # MaxDD on traded-only equity curve (exclude non-traded zero bars)
        if mask.any() and len(ret_tr) > 1:
            eq_s = pd.Series(np.cumprod(1 + ret_tr))
            dd_s = eq_s / eq_s.cummax() - 1
            mdd  = float(dd_s.min())
        else:
            mdd  = 0.0
        mdd = max(mdd, -1.0)   # cap at -100%
        hit  = float((y_test[mask] == preds[mask]).mean()) if mask.any() else 0.0

        results.append({
            "split":   split_n,
            "n_test":  len(test_idx),
            "trades":  n_tr,
            "cov%":    round(mask.mean() * 100, 1),
            "hit%":    round(hit * 100, 1),
            "net_ret": round(net_ret * 100, 2),
            "sharpe":  round(sharpe, 3),
            "max_dd%": round(mdd * 100, 2),
        })
        print(f"  Split {split_n:2d}:  trades={n_tr:3d}  hit={hit*100:.0f}%  "
              f"sharpe={sharpe:+.2f}  dd={mdd*100:.1f}%")

    return pd.DataFrame(results)


def print_summary(df, symbol):
    import numpy as np
    from scipy.stats import norm

    print()
    print("=" * 58)
    print(f"  CPCV Summary  |  {symbol}")
    print("=" * 58)
    rows = [
        ("Paths",          str(len(df))),
        ("Total Trades",   str(df["trades"].sum())),
        ("Avg Coverage",   f"{df['cov%'].mean():.1f}%"),
        ("Avg Hit Rate",   f"{df['hit%'].mean():.1f}%"),
        ("Avg Net Return", f"{df['net_ret'].mean():.2f}%"),
        ("Median Sharpe",  f"{df['sharpe'].median():+.3f}"),
        ("Mean Sharpe",    f"{df['sharpe'].mean():+.3f}"),
        ("Sharpe >= 0",    f"{(df['sharpe'] >= 0).mean()*100:.0f}%"),
        ("Mean Max DD",    f"{df['max_dd%'].mean():.1f}%"),
        ("Worst DD",       f"{df['max_dd%'].min():.1f}%"),
    ]
    for k, v in rows:
        print(f"  {k:<22} {v:>12}")

    sr_mean = df["sharpe"].mean()
    sr_std  = df["sharpe"].std() or 1e-9
    psr     = norm.cdf(sr_mean / (sr_std / max(len(df)**0.5, 1)))
    print(f"  {'PSR':<22} {psr*100:>11.1f}%")
    print("=" * 58)

    verdict = "VIABLE" if sr_mean > 0.3 else ("MARGINAL" if sr_mean > 0 else "NEEDS WORK")
    print(f"\n  Verdict: {verdict}  (mean Sharpe = {sr_mean:+.3f})\n")


def main() -> None:
    args = parse_args()
    slug = args.symbol.replace("-", "_").replace("/", "_")
    models    = Path("models")
    primary_p = models / f"{slug}_primary.pkl"
    meta_p    = models / f"{slug}_meta.pkl"

    if not primary_p.exists():
        print(f"ERROR: Model not found: {primary_p}")
        print("       Run train_colab.ipynb first.")
        sys.exit(1)

    df       = download_data(args.symbol, args.period, args.interval)
    features = build_features(df)
    labels   = compute_labels(df, features)

    results = run_cpcv(
        features, labels,
        primary_path = str(primary_p),
        meta_path    = str(meta_p),
        n_groups     = args.n_groups,
        n_test       = args.n_test,
        embargo      = args.embargo,
        threshold    = args.threshold,
        tx_cost      = args.tx_cost,
    )

    print_summary(results, args.symbol)
    print("Per-Split Results:")
    print(results.to_string(index=False))

    if args.save_results:
        out = Path("results")
        out.mkdir(exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv = out / f"backtest_{slug}_{ts}.csv"
        results.to_csv(csv, index=False)
        print(f"\nSaved: {csv}")


if __name__ == "__main__":
    main()
