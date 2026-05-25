"""Skill diagnostics for the multi-asset vol-targeted trend portfolio.

Reads daily prediction parquets in artifacts/ and reports whether (a) the
per-asset vol forecasts have genuine skill and (b) the aggregated portfolio
earns a real, cost-robust, risk-adjusted edge.

Usage: python scripts/diagnose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / "artifacts"
PPY = 252


def _sharpe(rets: pd.Series) -> float:
    r = rets.dropna()
    sd = r.std(ddof=0)
    return float(r.mean() / sd * np.sqrt(PPY)) if sd > 0 else 0.0


def _max_dd(rets: pd.Series) -> float:
    eq = (1 + rets.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def _r2(pred: pd.Series, actual: pd.Series) -> float:
    ss_res = float(((pred - actual) ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _per_asset_table() -> pd.DataFrame:
    rows = []
    for p in sorted(ART_DIR.glob("daily_predictions_*.parquet")):
        name = p.stem.replace("daily_predictions_", "")
        if name == "portfolio":
            continue
        df = pd.read_parquet(p).sort_index()
        if df.empty or "pnl" not in df.columns:
            continue
        v = df[["vol_fcst", "realized_rv"]].dropna() if {"vol_fcst", "realized_rv"}.issubset(df.columns) else pd.DataFrame()
        nv = df[["rv_20d", "realized_rv"]].dropna() if {"rv_20d", "realized_rv"}.issubset(df.columns) else pd.DataFrame()
        rows.append({
            "asset": name,
            "n": int(len(df)),
            "sharpe": _sharpe(df["pnl"]),
            "bh_sharpe": _sharpe(df["bh_pnl"]) if "bh_pnl" in df.columns else float("nan"),
            "vol_r2": _r2(v["vol_fcst"], v["realized_rv"]) if len(v) > 30 else float("nan"),
            "naive_r2": _r2(nv["rv_20d"], nv["realized_rv"]) if len(nv) > 30 else float("nan"),
            "hit_rate": float(df["dir_correct"].mean()) if "dir_correct" in df.columns else float("nan"),
            "max_dd": _max_dd(df["pnl"]),
        })
    return pd.DataFrame(rows)


def _verdict(sharpe: float, sharpe_2x: float, vt_sharpe: float,
             vol_r2_mean: float, naive_r2_mean: float,
             dd: float, bh_dd: float) -> str:
    beats_naive = vol_r2_mean > naive_r2_mean or np.isnan(naive_r2_mean)
    if (sharpe >= 0.7 and sharpe > vt_sharpe and sharpe_2x > 0.5
            and beats_naive and dd > bh_dd):
        return "REAL EDGE - portfolio is skilful, diversified and cost-robust"
    if sharpe >= 0.4 and beats_naive:
        return "MARGINAL - diversification helps but the bar isn't cleared"
    return "NO EDGE - portfolio fails to convincingly clear the bar"


def diagnose_portfolio() -> None:
    path = ART_DIR / "daily_predictions_portfolio.parquet"
    if not path.exists():
        print("\nNo portfolio prediction file - run scripts/run_baseline.py first")
        return
    df = pd.read_parquet(path).sort_index()
    if df.empty or "pnl" not in df.columns:
        print("\nportfolio file is empty")
        return

    print(f"\n## PORTFOLIO  ({len(df)} days)")
    sharpe = _sharpe(df["pnl"])
    bh_sharpe = _sharpe(df["bh_pnl"]) if "bh_pnl" in df.columns else 0.0
    vt_sharpe = _sharpe(df["vt_bh_pnl"]) if "vt_bh_pnl" in df.columns else 0.0
    dd = _max_dd(df["pnl"])
    bh_dd = _max_dd(df["bh_pnl"]) if "bh_pnl" in df.columns else 0.0
    total_ret = (1 + df["pnl"].fillna(0)).prod() - 1
    bh_ret = (1 + df["bh_pnl"].fillna(0)).prod() - 1 if "bh_pnl" in df.columns else 0.0

    print(f"  Sharpe                  : {sharpe:+.2f}   "
          f"(equal-wt b&h {bh_sharpe:+.2f}, vol-tgt eq-wt b&h {vt_sharpe:+.2f})")
    print(f"  max drawdown            : {dd:.1%}   (equal-wt b&h {bh_dd:.1%})")
    print(f"  total return            : {total_ret:+.1%}   (equal-wt b&h {bh_ret:+.1%})")
    ann = df["pnl"].dropna().std(ddof=0) * np.sqrt(PPY)
    print(f"  annualized vol          : {ann:.1%}")

    table = _per_asset_table()
    if not table.empty:
        vol_r2_mean = float(table["vol_r2"].mean(skipna=True))
        naive_r2_mean = float(table["naive_r2"].mean(skipna=True))
        print(f"  mean per-asset vol R^2  : {vol_r2_mean:+.3f}   "
              f"(naive rv_20d mean {naive_r2_mean:+.3f})")
    else:
        vol_r2_mean = naive_r2_mean = float("nan")

    # Cost stress: reconstruct the aggregated cost stream from per-asset files
    # using the same equal-risk weighting as aggregate_portfolio.
    sharpe_2x = sharpe
    import json
    scale = 1.0
    mp = ART_DIR / "metrics_portfolio.json"
    if mp.exists():
        try:
            scale = float(json.loads(mp.read_text()).get("portfolio_scale", 1.0))
        except Exception:
            pass
    cost_frames = []
    for p in sorted(ART_DIR.glob("daily_predictions_*.parquet")):
        if p.stem.endswith("_portfolio"):
            continue
        d = pd.read_parquet(p)
        if "cost" in d.columns:
            cost_frames.append(d["cost"])
    if cost_frames:
        cdf = pd.concat(cost_frames, axis=1).reindex(df.index)
        w = cdf.notna().sum(axis=1).clip(lower=1)
        port_cost = cdf.fillna(0).sum(axis=1) / w * scale
        line = []
        for k in (1, 2, 4):
            stressed = df["pnl"] - (k - 1) * port_cost
            s = _sharpe(stressed)
            line.append(f"{k}x={s:+.2f}")
            if k == 2:
                sharpe_2x = s
        print(f"  Sharpe vs cost          : {'  '.join(line)}")

    print(f"  VERDICT: {_verdict(sharpe, sharpe_2x, vt_sharpe, vol_r2_mean, naive_r2_mean, dd, bh_dd)}")

    if not table.empty:
        print("\n## PER-ASSET BREAKDOWN")
        with pd.option_context("display.max_rows", None,
                               "display.float_format", "{:+.3f}".format):
            print(table.set_index("asset")
                  [["n", "sharpe", "bh_sharpe", "vol_r2", "naive_r2",
                    "hit_rate", "max_dd"]].to_string())


def main() -> None:
    if not ART_DIR.exists():
        print("No artifacts/ directory - run scripts/run_baseline.py first")
        sys.exit(1)
    print("=" * 72)
    print("MULTI-ASSET VOLATILITY-TARGETED TREND - PORTFOLIO DIAGNOSTICS")
    print("=" * 72)
    diagnose_portfolio()
    print("\n" + "=" * 72)
    print("Portfolio TSMOM's edge is risk-adjusted return and shallow drawdowns")
    print("from diversification - not necessarily beating equal-weight buy-hold")
    print("on Sharpe during bull runs.")
    print("=" * 72)


if __name__ == "__main__":
    main()
