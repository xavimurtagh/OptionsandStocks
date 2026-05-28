from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RunConfig
from .model import predict_vol_multi_horizon, train_vol_multi_horizon


def walk_forward_vol_daily(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """Walk-forward backtest of the volatility-targeted trend strategy.

    Each fold trains a vol-forecast ensemble on the embargoed past, predicts
    forward realized vol on the test window, and sizes a trend-following
    position by target_vol / forecast_vol.
    """
    df = df.sort_index().copy()
    start_test = df.index.min() + pd.Timedelta(days=365 * cfg.train_min_years)
    test_dates = df.index[df.index >= start_test]
    if len(test_dates) == 0:
        return pd.DataFrame()

    h = cfg.backtest_horizon
    rv_col, ret_col = f"fwd_rv_{h}d", f"fwd_ret_{h}d"
    labeled = df.dropna(subset=[rv_col])
    # Embargo by row count so the longest forward-vol window cannot peek into
    # the test period (calendar-day embargo would undercount trading days).
    embargo = max(cfg.daily_horizons) + 1

    preds = []
    step = cfg.step_days
    for i in range(0, len(test_dates), step):
        t0 = test_dates[i]
        t1 = test_dates[min(i + step, len(test_dates) - 1)]
        prior = labeled[labeled.index < t0]
        train = prior.iloc[:-embargo] if len(prior) > embargo else prior.iloc[:0]
        test = labeled.loc[t0:t1]
        if len(train) < 252 or test.empty:
            continue
        models = train_vol_multi_horizon(train, cfg.daily_horizons,
                                         n_models=cfg.n_ensemble,
                                         device=cfg.device)
        if not models:
            continue
        fc = predict_vol_multi_horizon(models, test)
        if "vol_fcst" not in fc.columns:
            continue
        out = pd.DataFrame(index=test.index)
        out["close"] = test["close"]
        out["trend_signal"] = test["trend_signal"]
        for sig_col in ("xsmom_signal", "value_signal"):
            if sig_col in test.columns:
                out[sig_col] = test[sig_col]
        out["target_ret"] = test[ret_col]
        out["realized_rv"] = test[rv_col]
        if "rv_20d" in test.columns:
            out["rv_20d"] = test["rv_20d"]  # naive vol-forecast baseline
        out["vol_fcst"] = fc["vol_fcst"]
        for hh in cfg.daily_horizons:
            col = f"vol_fcst_{hh}"
            if col in fc.columns:
                out[col] = fc[col]
        # Weighted-combine TSMOM + XSMOM + value (each already in [-1, 1]).
        w = cfg.signal_weights
        combined = w.get("tsmom", 1.0) * test["trend_signal"].fillna(0)
        if "xsmom_signal" in test.columns:
            combined = combined + w.get("xsmom", 0.0) * test["xsmom_signal"].fillna(0)
        if "value_signal" in test.columns:
            combined = combined + w.get("value", 0.0) * test["value_signal"].fillna(0)
        if cfg.long_only:
            combined = combined.clip(lower=0.0)
        combined = combined.where(combined.abs() >= cfg.signal_threshold, 0.0)
        out["combined_signal"] = combined
        ratio = (cfg.target_vol / out["vol_fcst"]).clip(0, cfg.max_leverage)
        out["position"] = combined * ratio
        preds.append(out)
        print(f"[wf-vol] {t0.date()}->{t1.date()} n_train={len(train)} "
              f"n_test={len(test)} mean_vol_fcst={out['vol_fcst'].mean():.3f}")
    if not preds:
        return pd.DataFrame()
    result = pd.concat(preds).sort_index()
    # Adjacent folds share a boundary day via inclusive `.loc[t0:t1]`; the later
    # fold's prediction is the one trained on more data, so keep that.
    return result[~result.index.duplicated(keep="last")]


def _stats(rets: pd.Series, periods_per_year: int = 252) -> dict:
    rets = rets.dropna()
    if len(rets) == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0,
                "final_equity": 1.0, "cagr": 0.0}
    sd = rets.std(ddof=0)
    sharpe = rets.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0
    ds = rets[rets < 0].std(ddof=0)
    sortino = rets.mean() / ds * np.sqrt(periods_per_year) if ds and ds > 0 else 0.0
    equity = (1 + rets).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    years = len(rets) / periods_per_year
    final = float(equity.iloc[-1])
    cagr = final ** (1 / years) - 1 if years > 0 and final > 0 else 0.0
    return {"sharpe": float(sharpe), "sortino": float(sortino),
            "max_dd": float(dd), "final_equity": final, "cagr": float(cagr)}


def evaluate(pred: pd.DataFrame, cfg: RunConfig,
             holding: int) -> tuple[dict, pd.DataFrame]:
    """Mark-to-market the volatility-targeted trend strategy with costs."""
    if pred is None or pred.empty:
        return {"empty": True}, pd.DataFrame()
    pred = pred.sort_index().copy()
    pred = pred[pred["close"].notna()]
    if pred.empty or "position" not in pred.columns:
        return {"empty": True}, pd.DataFrame()

    pred["target_position"] = pred["position"].clip(-cfg.max_leverage,
                                                    cfg.max_leverage)
    pred["book"] = pred["target_position"].rolling(holding, min_periods=1).mean()
    pred["fwd1"] = pred["close"].pct_change(fill_method=None).shift(-1)

    turnover = pred["book"].diff().abs()
    turnover.iloc[0] = abs(pred["book"].iloc[0])
    pred["turnover"] = turnover
    pred["cost"] = turnover * (cfg.cost_bps / 1e4)

    pred["pnl"] = pred["book"] * pred["fwd1"] - pred["cost"]
    pred["bh_pnl"] = pred["fwd1"]
    pred["equity"] = (1 + pred["pnl"].fillna(0)).cumprod()
    pred["bh_equity"] = (1 + pred["bh_pnl"].fillna(0)).cumprod()

    metrics = {
        "strategy": _stats(pred["pnl"]),
        "benchmark": _stats(pred["bh_pnl"]),
        "n_predictions": int(len(pred)),
        "n_active": int((pred["target_position"] != 0).sum()),
        "avg_turnover": float(pred["turnover"].mean()),
    }

    # Volatility-targeted buy-hold: isolates the trend timing's contribution
    # (always long, sized the same way the strategy is).
    if "vol_fcst" in pred.columns:
        vt = (cfg.target_vol / pred["vol_fcst"]).clip(0, cfg.max_leverage)
        pred["vt_bh_pnl"] = vt * pred["fwd1"]
        pred["vt_bh_equity"] = (1 + pred["vt_bh_pnl"].fillna(0)).cumprod()
        metrics["vt_benchmark"] = _stats(pred["vt_bh_pnl"])

    # Vol-forecast skill.
    if {"vol_fcst", "realized_rv"}.issubset(pred.columns):
        v = pred[["vol_fcst", "realized_rv"]].dropna()
        if len(v) > 30:
            ss_res = float(((v["vol_fcst"] - v["realized_rv"]) ** 2).sum())
            ss_tot = float(((v["realized_rv"] - v["realized_rv"].mean()) ** 2).sum())
            metrics["vol_r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            metrics["vol_corr"] = float(v["vol_fcst"].corr(v["realized_rv"]))
            if "rv_20d" in pred.columns:
                nv = pred[["rv_20d", "realized_rv"]].dropna()
                if len(nv) > 30:
                    nss = float(((nv["rv_20d"] - nv["realized_rv"]) ** 2).sum())
                    nst = float(((nv["realized_rv"] - nv["realized_rv"].mean()) ** 2).sum())
                    metrics["vol_r2_naive"] = 1.0 - nss / nst if nst > 0 else 0.0

    # Directional hit rate (secondary - expectancy matters more).
    if "trend_signal" in pred.columns and "target_ret" in pred.columns:
        dir_pred = np.sign(pred["trend_signal"])
        pred["dir_correct"] = (dir_pred == np.sign(pred["target_ret"])).astype(float)
        traded = pred[pred["target_position"] != 0]
        metrics["hit_rate"] = (float(traded["dir_correct"].mean())
                               if len(traded) else 0.0)
    return metrics, pred


def aggregate_portfolio(per_asset: dict[str, pd.DataFrame],
                        cfg: RunConfig) -> tuple[dict, pd.DataFrame]:
    """Equal-risk-weighted portfolio over the per-asset backtest predictions.

    Each asset's per-asset book is already sized to cfg.target_vol; the
    portfolio is the equal-weighted mean of per-asset PnLs, scaled by
    cfg.portfolio_scale to lift the diversified vol back toward a typical
    CTA risk budget. Buy-hold benchmarks use the same diversification.
    """
    if not per_asset:
        return {"empty": True}, pd.DataFrame()

    pnls, bh_pnls, vt_bh_pnls = {}, {}, {}
    for name, df in per_asset.items():
        if df is None or df.empty:
            continue
        # Older cached parquets may carry duplicate index labels from the
        # inclusive walk-forward slicing; collapse them defensively.
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]
        if "pnl" in df.columns:
            pnls[name] = df["pnl"]
        if "bh_pnl" in df.columns:
            bh_pnls[name] = df["bh_pnl"]
        if "vt_bh_pnl" in df.columns:
            vt_bh_pnls[name] = df["vt_bh_pnl"]

    if not pnls:
        return {"empty": True}, pd.DataFrame()

    pnl_df = pd.DataFrame(pnls).sort_index()
    # Equal-weight across whichever assets have a return today; absent assets
    # contribute zero exposure rather than dragging the average down.
    weights = pnl_df.notna().sum(axis=1).clip(lower=1)
    portfolio_pnl = pnl_df.fillna(0).sum(axis=1) / weights * cfg.portfolio_scale

    bh_df = pd.DataFrame(bh_pnls).sort_index().reindex(pnl_df.index)
    bh_w = bh_df.notna().sum(axis=1).clip(lower=1)
    portfolio_bh = bh_df.fillna(0).sum(axis=1) / bh_w

    out = pd.DataFrame({"pnl": portfolio_pnl, "bh_pnl": portfolio_bh})
    out["equity"] = (1 + out["pnl"].fillna(0)).cumprod()
    out["bh_equity"] = (1 + out["bh_pnl"].fillna(0)).cumprod()

    if vt_bh_pnls:
        vt_df = pd.DataFrame(vt_bh_pnls).sort_index().reindex(pnl_df.index)
        vt_w = vt_df.notna().sum(axis=1).clip(lower=1)
        out["vt_bh_pnl"] = vt_df.fillna(0).sum(axis=1) / vt_w
        out["vt_bh_equity"] = (1 + out["vt_bh_pnl"].fillna(0)).cumprod()

    # Per-asset contribution columns for attribution / drill-down.
    for name in pnl_df.columns:
        out[f"pnl_{name}"] = pnl_df[name] / weights * cfg.portfolio_scale

    metrics = {
        "strategy": _stats(out["pnl"]),
        "benchmark": _stats(out["bh_pnl"]),
        "n_predictions": int(len(out)),
        "n_assets": int(pnl_df.shape[1]),
        "portfolio_scale": float(cfg.portfolio_scale),
    }
    if "vt_bh_pnl" in out.columns:
        metrics["vt_benchmark"] = _stats(out["vt_bh_pnl"])

    # Per-asset attribution table.
    per_asset_stats = {}
    for name in pnl_df.columns:
        s = pnl_df[name].dropna()
        per_asset_stats[name] = {
            "sharpe": _stats(s)["sharpe"],
            "contribution": float(s.sum() / cfg.portfolio_scale
                                  if cfg.portfolio_scale else s.sum()),
            "n_predictions": int(len(s)),
        }
    metrics["per_asset"] = per_asset_stats

    return metrics, out
