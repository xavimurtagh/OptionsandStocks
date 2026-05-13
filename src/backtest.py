from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .config import RunConfig
from .model import predict_ensemble, train_ensemble


def kelly_size(prob_up: np.ndarray, confidence: np.ndarray,
               fraction: float) -> np.ndarray:
    edge = 2.0 * prob_up - 1.0
    return np.clip(edge * confidence * fraction, -1.0, 1.0)


def walk_forward(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    df = df.sort_index().copy()
    train_min = pd.Timedelta(days=365 * cfg.train_min_years)
    start_test = df.index.min() + train_min
    test_dates = df.index[df.index >= start_test]
    if len(test_dates) == 0:
        raise ValueError("Not enough history for walk-forward")

    preds = []
    step = cfg.step_days
    embargo = cfg.horizon + 1
    labeled = df.dropna(subset=["target_ret"])
    for i in range(0, len(test_dates), step):
        t0 = test_dates[i]
        t1 = test_dates[min(i + step, len(test_dates) - 1)]
        train_cutoff = t0 - pd.Timedelta(days=embargo)
        train = labeled.loc[:train_cutoff]
        test = labeled.loc[t0:t1]
        if len(train) < 252 or test.empty:
            continue
        models, iso, cols = train_ensemble(train, n_models=cfg.n_ensemble)
        out = predict_ensemble(models, iso, cols, test)
        out["target_ret"] = test["target_ret"]
        out["close"] = test["close"]
        preds.append(out)
        print(f"[wf] train<= {train_cutoff.date()} test {t0.date()}->{t1.date()} "
              f"n_train={len(train)} n_test={len(test)}")
    return pd.concat(preds).sort_index()


def evaluate(pred: pd.DataFrame, cfg: RunConfig) -> dict:
    pred = pred.dropna(subset=["target_ret"]).copy()
    pred["position"] = kelly_size(
        pred["prob_up"].values, pred["confidence"].values, cfg.kelly_fraction
    )
    # Holding period is `horizon` days; scale per-day return to single position
    # over non-overlapping rebalance bars.
    pred["pnl"] = pred["position"] * pred["target_ret"]
    n_bars = max(1, 252 // cfg.horizon)
    sharpe = (pred["pnl"].mean() / pred["pnl"].std(ddof=0)) * np.sqrt(n_bars) if pred["pnl"].std(ddof=0) > 0 else 0.0
    downside = pred["pnl"][pred["pnl"] < 0].std(ddof=0)
    sortino = (pred["pnl"].mean() / downside) * np.sqrt(n_bars) if downside and downside > 0 else 0.0
    equity = (1 + pred["pnl"]).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    hit_rate = (np.sign(pred["pnl"]) > 0).mean()
    y_true = (pred["target_ret"] > 0).astype(int)
    brier = brier_score_loss(y_true, pred["prob_up"])
    ll = log_loss(y_true, pred["prob_up"].clip(1e-4, 1 - 1e-4))

    bins = pd.cut(pred["confidence"], bins=[-0.01, 0.1, 0.3, 0.6, 1.01],
                  labels=["very_low", "low", "med", "high"])
    by_conf = pred.groupby(bins, observed=True).agg(
        n=("pnl", "size"),
        mean_pnl=("pnl", "mean"),
        hit=("pnl", lambda x: (x > 0).mean()),
    )

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "hit_rate": float(hit_rate),
        "brier": float(brier),
        "log_loss": float(ll),
        "n_trades": int(len(pred)),
        "final_equity": float(equity.iloc[-1]) if len(equity) else 1.0,
        "by_confidence": by_conf,
    }
