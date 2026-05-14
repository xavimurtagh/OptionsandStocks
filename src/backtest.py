from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .config import RunConfig
from .model import HorizonModel, predict_multi_horizon, train_multi_horizon


def kelly_size(prob_up: np.ndarray, confidence: np.ndarray,
               fraction: float, threshold: float = 0.0) -> np.ndarray:
    edge = 2.0 * prob_up - 1.0
    raw = edge * confidence * fraction
    raw = np.clip(raw, -1.0, 1.0)
    return np.where(confidence < threshold, 0.0, raw)


def walk_forward_daily(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    df = df.sort_index().copy()
    train_min = pd.Timedelta(days=365 * cfg.train_min_years)
    start_test = df.index.min() + train_min
    test_dates = df.index[df.index >= start_test]
    if len(test_dates) == 0:
        raise ValueError("Not enough history for walk-forward")

    target_col_bt = f"target_ret_{cfg.backtest_horizon}d"
    labeled = df.dropna(subset=[target_col_bt])
    embargo = max(cfg.daily_horizons) + 1

    preds = []
    step = cfg.step_days
    for i in range(0, len(test_dates), step):
        t0 = test_dates[i]
        t1 = test_dates[min(i + step, len(test_dates) - 1)]
        train_cutoff = t0 - pd.Timedelta(days=embargo)
        train = labeled.loc[:train_cutoff]
        test = labeled.loc[t0:t1]
        if len(train) < 252 or test.empty:
            continue
        models = train_multi_horizon(
            train, cfg.daily_horizons, "target_up_{h}d", n_models=cfg.n_ensemble
        )
        if not models:
            continue
        out = predict_multi_horizon(models, test)
        out["target_ret"] = test[target_col_bt]
        out["close"] = test["close"]
        preds.append(out)
        print(f"[wf-daily] train<= {train_cutoff.date()} test {t0.date()}->{t1.date()} "
              f"n_train={len(train)} n_test={len(test)} horizons={list(models)}")
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
        models = train_multi_horizon(
            train, [horizon.label], "target_up", n_models=cfg.n_ensemble
        )
        if not models:
            continue
        out = predict_multi_horizon(models, test)
        out["target_ret"] = test["target_ret"]
        out["close"] = test["close"]
        preds.append(out)
    if not preds:
        return pd.DataFrame()
    print(f"[wf-intraday {horizon.label}] folds={len(preds)} total_preds={sum(len(p) for p in preds)}")
    return pd.concat(preds).sort_index()


def evaluate(pred: pd.DataFrame, cfg: RunConfig,
             bars_per_year: int = 50) -> dict:
    if pred.empty:
        return {"empty": True}
    pred = pred.dropna(subset=["target_ret"]).copy()

    pred["position"] = kelly_size(
        pred["prob_up"].values, pred["confidence"].values,
        cfg.kelly_fraction, cfg.confidence_threshold,
    )
    pred["pnl"] = pred["position"] * pred["target_ret"]
    pred["bh_pnl"] = pred["target_ret"]

    def _stats(rets: pd.Series) -> dict:
        if rets.std(ddof=0) == 0:
            sharpe = 0.0
        else:
            sharpe = rets.mean() / rets.std(ddof=0) * np.sqrt(bars_per_year)
        ds = rets[rets < 0].std(ddof=0)
        sortino = rets.mean() / ds * np.sqrt(bars_per_year) if ds and ds > 0 else 0.0
        equity = (1 + rets).cumprod()
        dd = (equity / equity.cummax() - 1).min() if len(equity) else 0.0
        return {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_dd": float(dd),
            "final_equity": float(equity.iloc[-1]) if len(equity) else 1.0,
        }

    strat = _stats(pred["pnl"])
    bench = _stats(pred["bh_pnl"])

    y_true = (pred["target_ret"] > 0).astype(int)
    brier = brier_score_loss(y_true, pred["prob_up"])
    ll = log_loss(y_true, pred["prob_up"].clip(1e-4, 1 - 1e-4))
    hit = (np.sign(pred["pnl"]) > 0).mean()

    bins = pd.cut(pred["confidence"], bins=[-0.01, 0.1, 0.3, 0.6, 1.01],
                  labels=["very_low", "low", "med", "high"])
    by_conf = pred.groupby(bins, observed=True).agg(
        n=("pnl", "size"),
        mean_pnl=("pnl", "mean"),
        hit=("pnl", lambda x: (x > 0).mean()),
    )

    return {
        "strategy": strat,
        "benchmark": bench,
        "brier": float(brier),
        "log_loss": float(ll),
        "hit_rate": float(hit),
        "n_trades": int((pred["position"] != 0).sum()),
        "n_predictions": int(len(pred)),
        "by_confidence": by_conf,
    }
