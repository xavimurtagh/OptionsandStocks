"""Research utilities: cross-validated strategy evaluation and lever sweeps.

The expensive step (training the vol-forecast ensemble) is independent of the
position-sizing levers (signal weights, long/short, threshold, target vol),
which are all applied *after* the forecast. So we train once per fold and score
the whole grid of configs on the same forecasts, then judge them with the
out-of-sample tools in validation.py (Deflated Sharpe, PBO). Shared by
scripts/validate.py and scripts/sweep.py.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .backtest import combine_and_size, evaluate
from .config import RunConfig
from .model import predict_vol_multi_horizon, train_vol_multi_horizon
from .validation import (deflated_sharpe_ratio, probability_backtest_overfitting,
                         purged_kfold_indices, sharpe_stats)


def fit_vol_fold(train: pd.DataFrame, test: pd.DataFrame, cfg: RunConfig):
    """Train the vol ensemble on the (purged) train fold; forecast on test."""
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


def fold_eval(test: pd.DataFrame, fc: pd.DataFrame, cfg: RunConfig,
              ticker: str) -> pd.DataFrame:
    """Combine + size + mark-to-market one fold; returns the enriched frame
    (carrying pnl, bh_pnl, vt_bh_pnl)."""
    out = combine_and_size(test, fc, cfg)
    ret_col = f"fwd_ret_{cfg.backtest_horizon}d"
    rv_col = f"fwd_rv_{cfg.backtest_horizon}d"
    out["target_ret"] = test[ret_col]
    out["realized_rv"] = test[rv_col]
    if "rv_20d" in test.columns:
        out["rv_20d"] = test["rv_20d"]
    _, enriched = evaluate(out, cfg, holding=cfg.backtest_horizon, ticker=ticker)
    return enriched


def series_stats(returns: pd.Series, n_trials: int = 1) -> dict:
    """Return-objective stats for one OOS return series."""
    st = sharpe_stats(returns)
    r = returns.dropna()
    if len(r) < 2:
        return {"sharpe": 0.0, "cagr": 0.0, "maxdd": 0.0, "calmar": 0.0,
                "logwealth": 0.0, "dsr": 0.0, "n": len(r)}
    eq = (1 + r).cumprod()
    final = float(eq.iloc[-1])
    logwealth = float(np.log(final)) if final > 0 else float("-inf")
    maxdd = float((eq / eq.cummax() - 1).min())
    years = len(r) / 252
    cagr = final ** (1 / years) - 1 if years > 0 and final > 0 else 0.0
    denom = abs(maxdd) if maxdd < 0 else 1e-9   # never-down -> very high Calmar
    calmar = cagr / denom
    dsr = deflated_sharpe_ratio(st["sharpe_per"], n_trials=n_trials, n_obs=st["n"],
                                skew=st["skew"], kurt=st["kurt"])
    return {"sharpe": st["sharpe_ann"], "cagr": float(cagr), "maxdd": maxdd,
            "calmar": float(calmar), "logwealth": logwealth, "dsr": dsr, "n": st["n"]}


# Named signal-weight presets for the sweep.
WEIGHT_PRESETS = {
    "tsmom": {"tsmom": 1.0, "xsmom": 0.0, "value": 0.0},
    "xsmom": {"tsmom": 0.0, "xsmom": 1.0, "value": 0.0},
    "blend_30_70": {"tsmom": 0.3, "xsmom": 0.7, "value": 0.0},
    "blend_50_50": {"tsmom": 0.5, "xsmom": 0.5, "value": 0.0},
}

DEFAULT_GRID = {
    "weights": list(WEIGHT_PRESETS),
    "long_only": [True, False],
    "signal_threshold": [0.0, 0.2],
    "target_vol": [0.12, 0.20],
}


def expand_grid(base: RunConfig, grid: dict) -> list[tuple[str, RunConfig]]:
    """Cartesian product of the grid into named RunConfig variants."""
    out = []
    for wname in grid["weights"]:
        for lo in grid["long_only"]:
            for thr in grid["signal_threshold"]:
                for tv in grid["target_vol"]:
                    cfg = replace(base, signal_weights=dict(WEIGHT_PRESETS[wname]),
                                  long_only=lo, signal_threshold=thr, target_vol=tv)
                    name = (f"{wname}|{'L' if lo else 'LS'}|"
                            f"thr{thr:g}|tv{tv:g}")
                    out.append((name, cfg))
    return out


def lever_sweep(feats: pd.DataFrame, base_cfg: RunConfig, ticker: str,
                grid: dict | None = None, n_splits: int = 5
                ) -> tuple[pd.DataFrame, float]:
    """Train once per purged fold, score every grid config on the same
    forecasts. Returns (ranked summary incl. B&H / vt-B&H references, sweep PBO).
    """
    grid = grid or DEFAULT_GRID
    cfgs = expand_grid(base_cfg, grid)
    embargo = max(base_cfg.daily_horizons) + 1
    splits = purged_kfold_indices(len(feats), n_splits, embargo)

    pnls: dict[str, list] = {name: [] for name, _ in cfgs}
    bh, vtbh = [], []
    for tr_idx, te_idx in splits:
        train, test = feats.iloc[tr_idx], feats.iloc[te_idx]
        fc = fit_vol_fold(train, test, base_cfg)
        if fc is None:
            continue
        ref = fold_eval(test, fc, base_cfg, ticker)  # references at base config
        bh.append(ref["bh_pnl"])
        vtbh.append(ref["vt_bh_pnl"])
        for name, c in cfgs:
            pnls[name].append(fold_eval(test, fc, c, ticker)["pnl"])

    def _stitch(parts):
        if not parts:
            return pd.Series(dtype=float)
        s = pd.concat(parts).sort_index()
        return s[~s.index.duplicated(keep="last")]

    matrix = pd.DataFrame({n: _stitch(v) for n, v in pnls.items() if v})
    n_trials = max(len(cfgs), 1)
    rows = {n: series_stats(matrix[n], n_trials=n_trials) for n in matrix.columns}
    # Benchmarks (not deflated - they were not selected).
    rows["[B&H]"] = series_stats(_stitch(bh), n_trials=1)
    rows["[vt-B&H]"] = series_stats(_stitch(vtbh), n_trials=1)

    summary = pd.DataFrame(rows).T
    summary = summary.sort_values("cagr", ascending=False)
    pbo = probability_backtest_overfitting(matrix, n_splits=10)["pbo"]
    return summary, pbo
