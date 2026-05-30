"""Rebalance the combined signal from cached per-asset predictions.

The model + raw signals (trend, xsmom, value) live in cached parquets,
so changing the signal_weights / long_only / threshold gating doesn't
require any model retraining - we just recombine and recompute pnl.

Default behavior:
  1. Print a Sharpe sweep across a handful of (tsmom, xsmom, value)
     weight combinations so the choice of defaults is visible.
  2. Apply cfg.signal_weights to every cached parquet (overwrites the
     combined_signal / target_position / pnl columns) and report the
     portfolio metrics under the new combine.

Usage:
    python scripts/rebalance.py            # sweep + apply current cfg defaults
    python scripts/rebalance.py --dry-run  # sweep only, don't modify parquets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ART_DIR, ASSETS, RunConfig

HOLDING = 5  # cfg.backtest_horizon

SWEEP = [
    # (tsmom, xsmom, value, label)
    (0.5, 0.3, 0.2, "pre-2023-diag default"),
    (1.0, 0.0, 0.0, "tsmom only"),
    (0.0, 1.0, 0.0, "xsmom only"),
    (0.0, 0.0, 1.0, "value only"),
    (0.5, 0.5, 0.0, "tsmom+xsmom 50/50"),
    (0.4, 0.6, 0.0, "tsmom+xsmom 40/60"),
    (0.3, 0.7, 0.0, "tsmom+xsmom 30/70 - new default"),
    (0.2, 0.8, 0.0, "tsmom+xsmom 20/80"),
    (0.33, 0.34, 0.33, "equal-weight three"),
    (0.4, 0.4, 0.2, "balanced with value"),
]


def load_cached() -> dict[str, pd.DataFrame]:
    out = {}
    for name in ASSETS:
        p = ART_DIR / f"daily_predictions_{name}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        needed = {"trend_signal", "xsmom_signal", "value_signal", "vol_fcst",
                  "close"}
        if not needed.issubset(df.columns):
            continue
        out[name] = df
    return out


def rebuild_asset(df: pd.DataFrame, weights: tuple[float, float, float],
                  cfg: RunConfig) -> pd.DataFrame:
    wt, wx, wv = weights
    sig = (wt * df["trend_signal"].fillna(0)
           + wx * df["xsmom_signal"].fillna(0)
           + wv * df["value_signal"].fillna(0))
    if cfg.long_only:
        sig = sig.clip(lower=0.0)
    sig = sig.where(sig.abs() >= cfg.signal_threshold, 0.0)
    ratio = (cfg.target_vol / df["vol_fcst"]).clip(0, cfg.max_leverage)
    pos = (sig * ratio).clip(-cfg.max_leverage, cfg.max_leverage)
    book = pos.rolling(HOLDING, min_periods=1).mean()
    fwd1 = df["close"].pct_change(fill_method=None).shift(-1)
    turnover = pos.diff().abs().fillna(0)
    cost = turnover * (cfg.cost_bps / 1e4)
    pnl = book * fwd1 - cost
    out = df.copy()
    out["combined_signal"] = sig
    out["position"] = pos
    out["target_position"] = pos
    out["book"] = book
    out["fwd1"] = fwd1
    out["cost"] = cost
    out["pnl"] = pnl
    if "vt_bh_pnl" not in out.columns:
        rv60 = df["close"].pct_change(fill_method=None).rolling(60).std()
        ann_rv = rv60 * np.sqrt(252)
        vt = (cfg.target_vol / ann_rv).clip(0, cfg.max_leverage).shift(1)
        out["vt_bh_pnl"] = vt * fwd1
    if "bh_pnl" not in out.columns:
        out["bh_pnl"] = fwd1
    return out


def portfolio_pnl(per_asset: dict[str, pd.Series], scale: float) -> pd.Series:
    pdf = pd.DataFrame(per_asset).sort_index()
    w = pdf.notna().sum(axis=1).clip(lower=1)
    return pdf.fillna(0).sum(axis=1) / w * scale


def sharpe(r: pd.Series) -> float:
    s = r.dropna()
    if len(s) < 30 or s.std() == 0:
        return float("nan")
    return (s.mean() / s.std()) * np.sqrt(252)


def max_dd(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def yearly_sharpe(p: pd.Series, year: int) -> float:
    return sharpe(p[p.index.year == year])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="sweep only, do not modify cached parquets")
    args = ap.parse_args()

    cfg = RunConfig()
    data = load_cached()
    if not data:
        print("no cached parquets with raw signals - run run_baseline.py first")
        return
    print(f"loaded {len(data)} assets")
    print(f"current cfg.signal_weights: {cfg.signal_weights}  "
          f"long_only={cfg.long_only}  threshold={cfg.signal_threshold}")

    # --- sweep ---
    print("\n" + "=" * 88)
    print("WEIGHT SWEEP (portfolio Sharpe at portfolio_scale={:.1f})"
          .format(cfg.portfolio_scale))
    print("=" * 88)
    print(f"{'tsmom':>6}{'xsmom':>7}{'value':>7}  {'full_sharpe':>12}"
          f"  {'2023':>7}  {'2022':>7}  {'max_dd':>8}  label")
    for wt, wx, wv, label in SWEEP:
        pnls = {n: rebuild_asset(df, (wt, wx, wv), cfg)["pnl"]
                for n, df in data.items()}
        port = portfolio_pnl(pnls, cfg.portfolio_scale)
        print(f"{wt:>6.2f}{wx:>7.2f}{wv:>7.2f}  "
              f"{sharpe(port):>+12.3f}  "
              f"{yearly_sharpe(port, 2023):>+7.2f}  "
              f"{yearly_sharpe(port, 2022):>+7.2f}  "
              f"{max_dd(port)*100:>+7.1f}%  {label}")

    # --- apply current cfg weights ---
    target = (cfg.signal_weights["tsmom"], cfg.signal_weights["xsmom"],
              cfg.signal_weights["value"])
    print("\n" + "=" * 88)
    print(f"APPLYING {target} ({'DRY RUN' if args.dry_run else 'WRITING TO DISK'})")
    print("=" * 88)
    pnls = {}
    for name, df in data.items():
        new_df = rebuild_asset(df, target, cfg)
        pnls[name] = new_df["pnl"]
        if not args.dry_run:
            p = ART_DIR / f"daily_predictions_{name}.parquet"
            new_df.to_parquet(p)
    port = portfolio_pnl(pnls, cfg.portfolio_scale)
    print(f"portfolio Sharpe   : {sharpe(port):+.3f}")
    print(f"portfolio max DD   : {max_dd(port)*100:+.1f}%")
    print(f"2023 Sharpe        : {yearly_sharpe(port, 2023):+.2f}")
    print(f"2022 Sharpe        : {yearly_sharpe(port, 2022):+.2f}")

    # per-asset shift
    print("\nPER-ASSET Sharpe (new combine, full period):")
    rows = []
    for n, p in pnls.items():
        rows.append({"asset": n, "sharpe": sharpe(p),
                     "n_active": int((p.abs() > 0).sum())})
    print(pd.DataFrame(rows).set_index("asset")
          .sort_values("sharpe", ascending=False).round(3).to_string())

    if args.dry_run:
        print("\n[dry run] no parquets modified - re-run without --dry-run to apply")
    else:
        print("\nCached parquets updated. Re-run scripts/diagnose.py for the full report.")


if __name__ == "__main__":
    main()
