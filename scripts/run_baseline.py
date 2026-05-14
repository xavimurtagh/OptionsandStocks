"""End-to-end gold/silver pipeline (daily multi-horizon + intraday).

Usage:
    python scripts/run_baseline.py              # both assets, both pipelines
    python scripts/run_baseline.py gold
    python scripts/run_baseline.py silver
    python scripts/run_baseline.py --no-intraday
    python scripts/run_baseline.py --no-daily

Outputs land in artifacts/ for the Streamlit UI to consume.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import (evaluate, kelly_size, walk_forward_daily,
                          walk_forward_intraday)
from src.config import ART_DIR, ASSETS, RunConfig
from src.data import load_all, load_intraday
from src.features import build_daily_features, build_intraday_features
from src.model import predict_multi_horizon, train_multi_horizon


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="list")
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def _save_metrics(name: str, payload: dict) -> None:
    out = {}
    for k, v in payload.items():
        if k == "by_confidence" and isinstance(v, pd.DataFrame):
            out[k] = v.reset_index().to_dict(orient="list")
        else:
            out[k] = _to_jsonable(v)
    (ART_DIR / f"metrics_{name}.json").write_text(json.dumps(out, indent=2, default=str))


def run_daily(name: str, data: dict, cfg: RunConfig) -> dict:
    asset = ASSETS[name]
    print(f"\n=== DAILY {name.upper()} ({asset.ticker}) ===")
    feats = build_daily_features(data, asset, cfg)
    print(f"feature matrix: {feats.shape}, span {feats.index.min().date()} -> {feats.index.max().date()}")

    bt = walk_forward_daily(feats, cfg)
    summary = {"empty": True}
    if not bt.empty:
        summary = evaluate(bt, cfg, bars_per_year=252 // cfg.backtest_horizon)
        bt["position"] = kelly_size(
            bt["prob_up"].values, bt["confidence"].values,
            cfg.kelly_fraction, cfg.confidence_threshold,
        )
        bt.to_parquet(ART_DIR / f"daily_predictions_{name}.parquet")
        print("-- strategy --", summary["strategy"])
        print("-- benchmark --", summary["benchmark"])
        print(f"hit={summary['hit_rate']:.3f} brier={summary['brier']:.3f} "
              f"log_loss={summary['log_loss']:.3f} n_trades={summary['n_trades']}")
        print(summary["by_confidence"])

    labeled = feats.dropna(subset=[f"target_up_{cfg.backtest_horizon}d"])
    models = train_multi_horizon(
        labeled, cfg.daily_horizons, "target_up_{h}d", n_models=cfg.n_ensemble
    )
    latest_signal = {}
    if models:
        latest_row = feats.iloc[[-1]]
        pred = predict_multi_horizon(models, latest_row).iloc[0]
        pos = kelly_size(np.array([pred["prob_up"]]),
                         np.array([pred["confidence"]]),
                         cfg.kelly_fraction, cfg.confidence_threshold)[0]
        latest_signal = {
            "asof": str(latest_row.index[-1].date()),
            "prob_up": float(pred["prob_up"]),
            "confidence": float(pred["confidence"]),
            "dispersion": float(pred["dispersion"]),
            "position": float(pos),
            "by_horizon": {
                str(h): {
                    "prob_up": float(pred[f"prob_up_{h}"]),
                    "prob_std": float(pred[f"prob_std_{h}"]),
                } for h in models
            },
        }
        fi = pd.concat({h: hm.feature_importance for h, hm in models.items()}, axis=1)
        fi.to_parquet(ART_DIR / f"feature_importance_{name}.parquet")

    _save_metrics(f"daily_{name}", {**summary, "latest_signal": latest_signal})
    return {"summary": summary, "latest": latest_signal}


def run_intraday(name: str, cfg: RunConfig) -> dict:
    asset = ASSETS[name]
    out_per_h = {}
    for h in cfg.intraday_horizons:
        print(f"\n=== INTRADAY {name.upper()} {h.label} ===")
        bars = load_intraday(asset.ticker, h.interval, h.period)
        if bars.empty:
            print(f"  no bars returned for {asset.ticker} {h.interval}")
            continue
        feats = build_intraday_features(bars, h)
        print(f"  bars: {len(feats)}  span: {feats.index.min()} -> {feats.index.max()}")

        bt = walk_forward_intraday(feats, h, cfg)
        summary = {"empty": True}
        if not bt.empty:
            summary = evaluate(bt, cfg, bars_per_year=h.forward_bars * 250)
            bt["position"] = kelly_size(
                bt["prob_up"].values, bt["confidence"].values,
                cfg.kelly_fraction, cfg.confidence_threshold,
            )
            bt.to_parquet(ART_DIR / f"intraday_predictions_{name}_{h.label}.parquet")
            print("-- strategy --", summary["strategy"])
            print("-- benchmark --", summary["benchmark"])

        labeled = feats.dropna(subset=["target_up"])
        models = train_multi_horizon(labeled, [h.label], "target_up", n_models=cfg.n_ensemble)
        latest = {}
        if models:
            last_row = feats.iloc[[-1]]
            pred = predict_multi_horizon(models, last_row).iloc[0]
            pos = kelly_size(np.array([pred["prob_up"]]),
                             np.array([pred["confidence"]]),
                             cfg.kelly_fraction, cfg.confidence_threshold)[0]
            latest = {
                "asof": str(last_row.index[-1]),
                "prob_up": float(pred["prob_up"]),
                "confidence": float(pred["confidence"]),
                "position": float(pos),
            }
        out_per_h[h.label] = {"summary": summary, "latest": latest}
        _save_metrics(f"intraday_{name}_{h.label}",
                      {**summary, "latest_signal": latest})
    return out_per_h


def main(argv: list[str]) -> None:
    do_daily = "--no-daily" not in argv
    do_intraday = "--no-intraday" not in argv
    targets = [a for a in argv[1:] if not a.startswith("--")] or list(ASSETS)
    cfg = RunConfig()

    data = None
    if do_daily:
        print(f"Loading daily data ({cfg.start} -> today)...")
        data = load_all(cfg, ASSETS)
        print(f"  prices: {data['prices'].shape}  fred: {data['fred'].shape}  cot: {data['cot'].shape}")

    summary = {}
    for name in targets:
        if name not in ASSETS:
            print(f"unknown asset: {name}")
            continue
        summary.setdefault(name, {})
        if do_daily:
            summary[name]["daily"] = run_daily(name, data, cfg)
        if do_intraday:
            summary[name]["intraday"] = run_intraday(name, cfg)

    (ART_DIR / "summary.json").write_text(
        json.dumps(_to_jsonable(summary), indent=2, default=str)
    )
    print(f"\nArtifacts written to {ART_DIR}")


if __name__ == "__main__":
    main(sys.argv)
