"""Skill diagnostics for the volatility-targeted trend backtest.

Reads the daily prediction parquets in artifacts/ and reports whether the
volatility forecast has genuine skill and whether the strategy earns a real,
cost-robust, risk-adjusted edge. No re-training - runs in seconds.

Usage:
    python scripts/diagnose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / "artifacts"
PPY = 252  # trading periods per year


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


def _verdict(vol_r2: float, naive_r2: float, sharpe: float, sharpe_2x: float,
             vt_sharpe: float, dd: float, bh_dd: float) -> str:
    beats_naive = vol_r2 > naive_r2 or np.isnan(naive_r2)
    if (beats_naive and vol_r2 >= 0.30 and sharpe >= 0.8
            and sharpe >= vt_sharpe and sharpe_2x > 0.5 and dd > bh_dd):
        return "REAL EDGE - vol forecast is skilful and the strategy is robust"
    if beats_naive and vol_r2 >= 0.15 and sharpe >= 0.4:
        return "MARGINAL - some skill, not yet convincing"
    return "NO EDGE - vol forecast or strategy does not clear the bar"


def diagnose(label: str, path: Path) -> bool:
    if not path.exists():
        return False
    df = pd.read_parquet(path).sort_index()
    if not {"vol_fcst", "realized_rv", "pnl", "bh_pnl"}.issubset(df.columns):
        print(f"\n## {label}: missing expected columns, skipped")
        return True
    print(f"\n## {label}   ({len(df)} predictions)")

    # --- volatility-forecast skill -----------------------------------------
    v = df[["vol_fcst", "realized_rv"]].dropna()
    if len(v) < 30:
        print("  too few scored rows to judge vol skill")
        return True
    vol_r2 = _r2(v["vol_fcst"], v["realized_rv"])
    vol_corr = float(v["vol_fcst"].corr(v["realized_rv"]))
    naive_r2 = float("nan")
    if "rv_20d" in df.columns:
        nv = df[["rv_20d", "realized_rv"]].dropna()
        if len(nv) > 30:
            naive_r2 = _r2(nv["rv_20d"], nv["realized_rv"])
    print(f"  vol forecast R^2       : {vol_r2:+.3f}   "
          f"(naive rv_20d baseline {naive_r2:+.3f})")
    print(f"  vol forecast corr      : {vol_corr:+.3f}   (1.0 = perfect)")

    # --- strategy vs benchmarks --------------------------------------------
    sharpe = _sharpe(df["pnl"])
    bh_sharpe = _sharpe(df["bh_pnl"])
    vt_sharpe = _sharpe(df["vt_bh_pnl"]) if "vt_bh_pnl" in df.columns else 0.0
    dd, bh_dd = _max_dd(df["pnl"]), _max_dd(df["bh_pnl"])
    strat_ret = (1 + df["pnl"].fillna(0)).prod() - 1
    bh_ret = (1 + df["bh_pnl"].fillna(0)).prod() - 1
    print(f"  strategy Sharpe        : {sharpe:+.2f}   "
          f"(buy&hold {bh_sharpe:+.2f}, vol-targeted b&h {vt_sharpe:+.2f})")
    print(f"  max drawdown           : {dd:.1%}   (buy&hold {bh_dd:.1%})")
    print(f"  total return  strategy {strat_ret:+.1%}   buy&hold {bh_ret:+.1%}")

    # --- cost sensitivity ---------------------------------------------------
    sharpe_2x = sharpe
    if {"cost"}.issubset(df.columns):
        gross = df["pnl"] + df["cost"]
        line = []
        for k in (1, 2, 4):
            s = _sharpe(gross - k * df["cost"])
            line.append(f"{k}x={s:+.2f}")
            if k == 2:
                sharpe_2x = s
        print(f"  Sharpe vs cost         : {'  '.join(line)}")
    if "turnover" in df.columns:
        print(f"  avg turnover           : {df['turnover'].mean():.3f}")
    if "dir_correct" in df.columns:
        print(f"  trend hit rate         : {df['dir_correct'].mean():.3f}  "
              f"(secondary - expectancy matters more than hit rate)")

    print(f"  VERDICT: {_verdict(vol_r2, naive_r2, sharpe, sharpe_2x, vt_sharpe, dd, bh_dd)}")
    return True


def main() -> None:
    if not ART_DIR.exists():
        print("No artifacts/ directory - run scripts/run_baseline.py first")
        sys.exit(1)
    print("=" * 64)
    print("VOLATILITY-TARGETED TREND - BACKTEST DIAGNOSTICS")
    print("=" * 64)
    found = False
    for asset in ("gold", "silver"):
        if diagnose(f"{asset} daily",
                    ART_DIR / f"daily_predictions_{asset}.parquet"):
            found = True
    if not found:
        print("\nNo daily prediction files - run scripts/run_baseline.py first")
        return
    print("\n" + "=" * 64)
    print("A skilful vol forecast must beat the naive rv_20d baseline. The")
    print("strategy's edge is risk-adjusted return and drawdown control after")
    print("costs - not necessarily beating a long-only bull market on Sharpe.")
    print("=" * 64)


if __name__ == "__main__":
    main()
