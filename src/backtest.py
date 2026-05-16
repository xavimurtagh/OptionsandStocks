from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .config import RunConfig
from .meta import MetaStrategy
from .model import predict_vol_multi_horizon, train_vol_multi_horizon


def kelly_size(prob_up: np.ndarray, confidence: np.ndarray,
               fraction: float, threshold: float = 0.0) -> np.ndarray:
    edge = 2.0 * prob_up - 1.0
    raw = np.clip(edge * confidence * fraction, -1.0, 1.0)
    return np.where(confidence < threshold, 0.0, raw)


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
        raise ValueError("Not enough history for walk-forward")

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
        out["target_ret"] = test[ret_col]
        out["realized_rv"] = test[rv_col]
        if "rv_20d" in test.columns:
            out["rv_20d"] = test["rv_20d"]  # naive vol-forecast baseline
        out["vol_fcst"] = fc["vol_fcst"]
        for hh in cfg.daily_horizons:
            col = f"vol_fcst_{hh}"
            if col in fc.columns:
                out[col] = fc[col]
        ratio = (cfg.target_vol / out["vol_fcst"]).clip(0, cfg.max_leverage)
        pos = out["trend_signal"] * ratio
        if cfg.vrp_filter and "opt_iv" in test.columns:
            extreme = (test["opt_iv"] / out["vol_fcst"] > 1.5).fillna(False)
            pos = pos.where(~extreme, pos * 0.5)
        out["position"] = pos
        preds.append(out)
        print(f"[wf-vol] {t0.date()}->{t1.date()} n_train={len(train)} "
              f"n_test={len(test)} mean_vol_fcst={out['vol_fcst'].mean():.3f}")
    return pd.concat(preds).sort_index() if preds else pd.DataFrame()


def walk_forward_intraday(df: pd.DataFrame, horizon, cfg: RunConfig) -> pd.DataFrame:
    df = df.sort_index().copy()
    labeled = df.dropna(subset=["target_ret"])
    if len(labeled) < horizon.train_min_bars + horizon.step_bars:
        print(f"[wf-intraday {horizon.label}] not enough bars: {len(labeled)}")
        return pd.DataFrame()

    preds = []
    n = len(labeled)
    embargo = horizon.forward_bars + 1
    for end in range(horizon.train_min_bars, n, horizon.step_bars):
        train = labeled.iloc[: end - embargo]
        test = labeled.iloc[end: min(end + horizon.step_bars, n)]
        if len(train) < horizon.train_min_bars or test.empty:
            continue
        strat = MetaStrategy([horizon.label], "target_up", "target_ret",
                             "weight", n_ensemble=cfg.n_ensemble,
                             device=cfg.device).fit(train)
        if not strat.ok:
            continue
        out = strat.predict(test)
        if out.empty:
            continue
        out["target_ret"] = test["target_ret"]
        out["close"] = test["close"]
        if "rv_20b" in test.columns:
            out["vol"] = test["rv_20b"]
        preds.append(out)
    if not preds:
        return pd.DataFrame()
    print(f"[wf-intraday {horizon.label}] folds={len(preds)} "
          f"total_preds={sum(len(p) for p in preds)}")
    return pd.concat(preds).sort_index()


def _stats(rets: pd.Series, periods_per_year: int) -> dict:
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


def evaluate(pred: pd.DataFrame, cfg: RunConfig, holding: int,
             periods_per_year: int) -> tuple[dict, pd.DataFrame]:
    """Mark-to-market the strategy per period (each price move counted once),
    with transaction costs.

    Sizing: a precomputed `position` column is used directly when present
    (volatility-targeted trend path); otherwise the legacy Kelly path runs.
    """
    if pred is None or pred.empty:
        return {"empty": True}, pd.DataFrame()
    pred = pred.sort_index().copy()
    pred = pred[pred["close"].notna()]
    if pred.empty:
        return {"empty": True}, pd.DataFrame()

    if "position" in pred.columns:
        pred["target_position"] = pred["position"].clip(
            -cfg.max_leverage, cfg.max_leverage)
    else:
        pred["target_position"] = kelly_size(
            pred["prob_up"].to_numpy(), pred["confidence"].to_numpy(),
            cfg.kelly_fraction, cfg.confidence_threshold,
        )
        if "vol" in pred.columns and pred["vol"].notna().any():
            med = pred["vol"].median()
            scalar = (med / pred["vol"]).clip(0.3, 2.5).fillna(1.0)
            pred["target_position"] *= scalar

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
        "strategy": _stats(pred["pnl"], periods_per_year),
        "benchmark": _stats(pred["bh_pnl"], periods_per_year),
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
        metrics["vt_benchmark"] = _stats(pred["vt_bh_pnl"], periods_per_year)

    # Volatility-forecast skill.
    if {"vol_fcst", "realized_rv"}.issubset(pred.columns):
        v = pred[["vol_fcst", "realized_rv"]].dropna()
        if len(v) > 30:
            ss_res = float(((v["vol_fcst"] - v["realized_rv"]) ** 2).sum())
            ss_tot = float(((v["realized_rv"] - v["realized_rv"].mean()) ** 2).sum())
            metrics["vol_r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            metrics["vol_corr"] = float(v["vol_fcst"].corr(v["realized_rv"]))

    # Directional hit rate (secondary - the prediction's sign vs realised).
    if "prob_up" in pred.columns:
        dir_pred = np.sign(pred["prob_up"] - 0.5)
    elif "trend_signal" in pred.columns:
        dir_pred = np.sign(pred["trend_signal"])
    else:
        dir_pred = pd.Series(0.0, index=pred.index)
    if "target_ret" in pred.columns:
        pred["dir_correct"] = (dir_pred == np.sign(pred["target_ret"])).astype(float)
        traded = pred[pred["target_position"] != 0]
        metrics["hit_rate"] = (float(traded["dir_correct"].mean())
                               if len(traded) else 0.0)

    # Classification metrics - only meaningful for a probabilistic prediction.
    if "prob_up" in pred.columns and "target_ret" in pred.columns:
        y_true = (pred["target_ret"] > 0).astype(int)
        valid = pred["prob_up"].notna() & pred["target_ret"].notna()
        if valid.any():
            metrics["brier"] = float(brier_score_loss(
                y_true[valid], pred["prob_up"][valid]))
            metrics["log_loss"] = float(log_loss(
                y_true[valid], pred["prob_up"][valid].clip(1e-4, 1 - 1e-4)))
    if "confidence" in pred.columns and "dir_correct" in pred.columns:
        bins = pd.cut(pred["confidence"], bins=[-0.01, 0.1, 0.3, 0.6, 1.01],
                      labels=["very_low", "low", "med", "high"])
        metrics["by_confidence"] = pred.groupby(bins, observed=True).agg(
            n=("dir_correct", "size"),
            hit=("dir_correct", "mean"),
            mean_pnl=("pnl", "mean"),
        )
    return metrics, pred
