"""Honest out-of-sample scoreboard for the vol-targeted trend strategy.

Unlike scripts/run_baseline.py (a single tradable walk-forward), this runs
purged K-fold cross-validation and reports the Deflated Sharpe Ratio so a result
can be judged against the number of configurations we've tried. Use it as the
gate for every later change: a change ships only if it improves OOS Calmar /
log-wealth AND keeps PBO < 0.5.

Usage:
    python scripts/validate.py                 # default subset, purged-KFold DSR
    python scripts/validate.py gold silver spy
    python scripts/validate.py --fast          # 1 ensemble member, 5d horizon only
    python scripts/validate.py gold --pbo       # add a signal-weight sweep + PBO
    python scripts/validate.py --splits 6 --trials 20
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import combine_and_size, evaluate
from src.config import ASSETS, RunConfig
from src.data import load_all
from src.features import (build_daily_features, cross_sectional_momentum,
                          cross_sectional_value)
from src.model import predict_vol_multi_horizon, train_vol_multi_horizon
from src.validation import (deflated_sharpe_ratio, probability_backtest_overfitting,
                            purged_kfold_indices, run_purged_cv, sharpe_stats)


def _fit_vol(train: pd.DataFrame, test: pd.DataFrame, cfg: RunConfig):
    """Train the vol ensemble on the (purged) train fold and forecast on test.
    The signal combine is post-hoc, so this is shared across weight configs."""
    rv_col = f"fwd_rv_{cfg.backtest_horizon}d"
    tr = train.dropna(subset=[rv_col])
    if len(tr) < 252:
        return None
    models = train_vol_multi_horizon(tr, cfg.daily_horizons,
                                     n_models=cfg.n_ensemble, device=cfg.device)
    if not models:
        return None
    fc = predict_vol_multi_horizon(models, test)
    return fc if "vol_fcst" in fc.columns else None


def _pnl_from_forecast(test: pd.DataFrame, fc: pd.DataFrame, cfg: RunConfig,
                       ticker: str) -> pd.Series:
    out = combine_and_size(test, fc, cfg)
    rv_col, ret_col = f"fwd_rv_{cfg.backtest_horizon}d", f"fwd_ret_{cfg.backtest_horizon}d"
    out["target_ret"] = test[ret_col]
    out["realized_rv"] = test[rv_col]
    if "rv_20d" in test.columns:
        out["rv_20d"] = test["rv_20d"]
    _, enriched = evaluate(out, cfg, holding=cfg.backtest_horizon, ticker=ticker)
    return enriched["pnl"]


def make_fit_predict(ticker: str):
    def fit_predict(train, test, cfg):
        fc = _fit_vol(train, test, cfg)
        if fc is None:
            return pd.Series(dtype=float)
        return _pnl_from_forecast(test, fc, cfg, ticker)
    return fit_predict


def _calmar(combined: pd.Series) -> tuple[float, float, float]:
    """Terminal log-wealth, CAGR and max drawdown of a per-period return series."""
    r = combined.dropna()
    if len(r) < 2:
        return 0.0, 0.0, 0.0
    equity = (1 + r).cumprod()
    log_wealth = float(np.log(equity.iloc[-1])) if equity.iloc[-1] > 0 else float("-inf")
    dd = float((equity / equity.cummax() - 1).min())
    years = len(r) / 252
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else 0.0
    return log_wealth, cagr, dd


def pbo_weight_sweep(df: pd.DataFrame, cfg: RunConfig, ticker: str,
                     n_splits: int) -> dict:
    """Train the vol model once per fold, then evaluate several signal-weight
    configs on the same forecasts -> a returns matrix for CSCV PBO."""
    configs = {
        "tsmom": {"tsmom": 1.0, "xsmom": 0.0, "value": 0.0},
        "xsmom": {"tsmom": 0.0, "xsmom": 1.0, "value": 0.0},
        "value": {"tsmom": 0.0, "xsmom": 0.0, "value": 1.0},
        "blend_30_70": {"tsmom": 0.3, "xsmom": 0.7, "value": 0.0},
        "blend_50_50": {"tsmom": 0.5, "xsmom": 0.5, "value": 0.0},
        "equal": {"tsmom": 0.34, "xsmom": 0.33, "value": 0.33},
    }
    embargo = max(cfg.daily_horizons) + 1
    splits = purged_kfold_indices(len(df), n_splits, embargo)
    per_config: dict[str, list] = {k: [] for k in configs}
    for tr_idx, te_idx in splits:
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        fc = _fit_vol(train, test, cfg)
        if fc is None:
            continue
        for name, weights in configs.items():
            c = RunConfig(**{**cfg.__dict__, "signal_weights": weights})
            per_config[name].append(_pnl_from_forecast(test, fc, c, ticker))
    matrix = pd.DataFrame({k: pd.concat(v).sort_index() for k, v in per_config.items()
                           if v})
    matrix = matrix[~matrix.index.duplicated(keep="last")]
    return probability_backtest_overfitting(matrix, n_splits=10)


def main(argv: list[str]) -> None:
    fast = "--fast" in argv
    do_pbo = "--pbo" in argv
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
              "       gold/silver. Re-run when fred.stlouisfed.org is reachable "
              "to seed the cache; numbers below are degraded.")
    tickers = [a.ticker for a in full_assets.values()]
    data["xsmom"] = cross_sectional_momentum(data["prices"], tickers, cfg.xsmom_lookback)
    data["value"] = cross_sectional_value(data["prices"], tickers, cfg.value_lookback)

    print(f"\n{'asset':<8}{'OOS Sharpe':>12}{'fold mean±sd':>16}{'CAGR':>8}"
          f"{'maxDD':>8}{'DSR':>8}{'PBO':>8}")
    print("-" * 70)
    for name in names:
        asset = ASSETS[name]
        feats = build_daily_features(data, asset, cfg)
        if feats.empty:
            print(f"{name:<8}  (no feature matrix)")
            continue
        folds, combined = run_purged_cv(feats, cfg, make_fit_predict(asset.ticker),
                                        n_splits=n_splits)
        if combined.empty:
            print(f"{name:<8}  (insufficient data)")
            continue
        st = sharpe_stats(combined)
        dsr = deflated_sharpe_ratio(st["sharpe_per"], n_trials=n_trials,
                                    n_obs=st["n"], skew=st["skew"], kurt=st["kurt"])
        _, cagr, dd = _calmar(combined)
        fmean = folds["sharpe_ann"].mean() if not folds.empty else float("nan")
        fsd = folds["sharpe_ann"].std() if not folds.empty else float("nan")
        pbo = pbo_weight_sweep(feats, cfg, asset.ticker, n_splits)["pbo"] if do_pbo else float("nan")
        print(f"{name:<8}{st['sharpe_ann']:>12.2f}{f'{fmean:.2f}±{fsd:.2f}':>16}"
              f"{cagr:>8.1%}{dd:>8.1%}{dsr:>8.2f}{pbo:>8.2f}")


def _flag(argv: list[str], flag: str, default):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


if __name__ == "__main__":
    main(sys.argv)
