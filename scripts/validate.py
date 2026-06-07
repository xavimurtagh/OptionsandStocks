"""Honest single-config out-of-sample scoreboard for the vol-targeted strategy.

Runs purged K-fold cross-validation per asset and reports the Deflated Sharpe
Ratio so a result can be judged against the number of configs tried. For a
multi-config lever comparison + PBO, use scripts/sweep.py.

Usage:
    python scripts/validate.py                 # gold silver spy
    python scripts/validate.py gold --fast     # 1 ensemble member, 5d horizon
    python scripts/validate.py --splits 6 --trials 20
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ASSETS, RunConfig
from src.data import load_all
from src.features import (build_daily_features, cross_sectional_momentum,
                          cross_sectional_value)
from src.research import fit_vol_fold, fold_eval, series_stats
from src.validation import run_purged_cv


def make_fit_predict(ticker: str):
    def fit_predict(train, test, cfg):
        fc = fit_vol_fold(train, test, cfg)
        if fc is None:
            return pd.Series(dtype=float)
        return fold_eval(test, fc, cfg, ticker)["pnl"]
    return fit_predict


def _flag(argv, flag, default):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def main(argv: list[str]) -> None:
    fast = "--fast" in argv
    n_splits = int(_flag(argv, "--splits", 5))
    n_trials = int(_flag(argv, "--trials", 10))
    names = [a for a in argv[1:] if not a.startswith("--")] or ["gold", "silver", "spy"]
    names = [n for n in names if n in ASSETS]

    cfg = RunConfig()
    if fast:
        cfg.n_ensemble = 1
        cfg.daily_horizons = [5]

    full_assets = {n: ASSETS[n] for n in cfg.universe}
    print(f"Loading data for {len(full_assets)} assets (xsmom/value context)...")
    data = load_all(cfg, full_assets)
    if data["fred"].empty:
        print("[WARN] FRED macro features unavailable this run (network) - "
              "real yields / DXY / VIX / GVZ are missing, which badly handicaps\n"
              "       gold/silver. Re-run when fred.stlouisfed.org is reachable; "
              "numbers below are degraded.")
    tickers = [a.ticker for a in full_assets.values()]
    data["xsmom"] = cross_sectional_momentum(data["prices"], tickers, cfg.xsmom_lookback)
    data["value"] = cross_sectional_value(data["prices"], tickers, cfg.value_lookback)

    print(f"\n{'asset':<8}{'OOS Sharpe':>12}{'fold mean±sd':>16}{'CAGR':>8}"
          f"{'maxDD':>8}{'Calmar':>8}{'DSR':>7}")
    print("-" * 67)
    for name in names:
        asset = ASSETS[name]
        feats = build_daily_features(data, asset, cfg)
        if feats.empty:
            print(f"{name:<8}  (no feature matrix)"); continue
        folds, combined = run_purged_cv(feats, cfg, make_fit_predict(asset.ticker),
                                        n_splits=n_splits)
        if combined.empty:
            print(f"{name:<8}  (insufficient data)"); continue
        st = series_stats(combined, n_trials=n_trials)
        fmean = folds["sharpe_ann"].mean() if not folds.empty else float("nan")
        fsd = folds["sharpe_ann"].std() if not folds.empty else float("nan")
        print(f"{name:<8}{st['sharpe']:>12.2f}{f'{fmean:.2f}±{fsd:.2f}':>16}"
              f"{st['cagr']:>8.1%}{st['maxdd']:>8.1%}{st['calmar']:>8.2f}"
              f"{st['dsr']:>7.2f}")


if __name__ == "__main__":
    main(sys.argv)
