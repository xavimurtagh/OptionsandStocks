"""End-to-end gold/silver pipeline.

Daily multi-horizon + intraday, triple-barrier labels, meta-labelling
confidence, volatility-targeted sizing, and an optional GPU neural model.

Usage:
    python scripts/run_baseline.py              # both assets, everything
    python scripts/run_baseline.py gold
    python scripts/run_baseline.py --no-intraday
    python scripts/run_baseline.py --no-neural

Outputs land in artifacts/ for the Streamlit UI.
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
from src.meta import MetaStrategy
from src.model import explain_primary
from src.neural import HAS_TORCH, neural_latest, neural_predictions


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


def _signal_from_pred(pred_row: pd.Series, cfg: RunConfig,
                      horizons=None) -> dict:
    pos = kelly_size(np.array([pred_row["prob_up"]]),
                     np.array([pred_row["confidence"]]),
                     cfg.kelly_fraction, cfg.confidence_threshold)[0]
    sig = {
        "asof": str(getattr(pred_row, "name", "")),
        "prob_up": float(pred_row["prob_up"]),
        "confidence": float(pred_row["confidence"]),
        "position": float(pos),
    }
    if "meta_prob" in pred_row:
        sig["meta_prob"] = float(pred_row["meta_prob"])
    if "dispersion" in pred_row:
        sig["dispersion"] = float(pred_row["dispersion"])
    if horizons:
        sig["by_horizon"] = {
            str(h): {
                "prob_up": float(pred_row[f"prob_up_{h}"]),
                "prob_std": float(pred_row[f"prob_std_{h}"]),
            } for h in horizons if f"prob_up_{h}" in pred_row
        }
    return sig


def run_daily(name: str, data: dict, cfg: RunConfig) -> dict:
    asset = ASSETS[name]
    print(f"\n=== DAILY {name.upper()} ({asset.ticker}) ===")
    feats = build_daily_features(data, asset, cfg)
    print(f"feature matrix: {feats.shape}, "
          f"span {feats.index.min().date()} -> {feats.index.max().date()}")

    bt = walk_forward_daily(feats, cfg)
    summary = {"empty": True}
    if not bt.empty:
        summary, enriched = evaluate(bt, cfg, holding=cfg.backtest_horizon,
                                     periods_per_year=252)
        enriched.to_parquet(ART_DIR / f"daily_predictions_{name}.parquet")
        print("-- strategy --", summary["strategy"])
        print("-- benchmark --", summary["benchmark"])
        print(f"hit={summary['hit_rate']:.3f} log_loss={summary['log_loss']:.3f} "
              f"n_active={summary['n_active']}")
        print(summary["by_confidence"])

    ret_col = f"target_ret_{cfg.backtest_horizon}d"
    strat = MetaStrategy(cfg.daily_horizons, "target_up_{h}d", ret_col,
                         "weight_{h}d", n_ensemble=cfg.n_ensemble,
                         device=cfg.device).fit(feats.dropna(subset=[ret_col]))
    latest_signal = {}
    if strat.ok:
        latest_row = feats.iloc[[-1]]
        pred = strat.predict(latest_row).iloc[0]
        latest_signal = _signal_from_pred(pred, cfg, horizons=list(strat.primary))
        latest_signal["drivers"] = [
            {"feature": f, "contribution": c}
            for f, c in explain_primary(strat.primary, latest_row)
        ]
        fi = pd.concat({h: hm.feature_importance
                        for h, hm in strat.primary.items()}, axis=1)
        fi.to_parquet(ART_DIR / f"feature_importance_{name}.parquet")

    _save_metrics(f"daily_{name}", {**summary, "latest_signal": latest_signal})
    return {"summary": summary, "latest": latest_signal}


def run_intraday(name: str, cfg: RunConfig, do_neural: bool) -> dict:
    asset = ASSETS[name]
    out_per_h = {}
    for h in cfg.intraday_horizons:
        print(f"\n=== INTRADAY {name.upper()} {h.label} ===")
        bars = load_intraday(asset.ticker, h.interval, h.period)
        if bars.empty:
            print(f"  no bars for {asset.ticker} {h.interval}")
            continue
        feats = build_intraday_features(bars, h)
        print(f"  bars: {len(feats)}  span: {feats.index.min()} -> {feats.index.max()}")

        bt = walk_forward_intraday(feats, h, cfg)
        summary = {"empty": True}
        if not bt.empty:
            summary, enriched = evaluate(bt, cfg, holding=h.forward_bars,
                                         periods_per_year=h.bars_per_year)
            enriched.to_parquet(ART_DIR / f"intraday_predictions_{name}_{h.label}.parquet")
            print("-- strategy --", summary["strategy"])

        strat = MetaStrategy([h.label], "target_up", "target_ret", "weight",
                             n_ensemble=cfg.n_ensemble,
                             device=cfg.device).fit(feats.dropna(subset=["target_ret"]))
        latest = {}
        if strat.ok:
            latest = _signal_from_pred(strat.predict(feats.iloc[[-1]]).iloc[0],
                                       cfg, horizons=[h.label])
        _save_metrics(f"intraday_{name}_{h.label}",
                      {**summary, "latest_signal": latest})

        neural_info = {}
        if do_neural and HAS_TORCH:
            print(f"  [neural] training TCN for {h.label} ...")
            npred = neural_predictions(feats, "target_up", device=cfg.device)
            nsummary = {"empty": True}
            if not npred.empty:
                nsummary, nenriched = evaluate(npred, cfg, holding=h.forward_bars,
                                               periods_per_year=h.bars_per_year)
                nenriched.to_parquet(ART_DIR / f"neural_predictions_{name}_{h.label}.parquet")
                print("  -- neural strategy --", nsummary["strategy"])
            nlatest = neural_latest(feats, "target_up", device=cfg.device)
            _save_metrics(f"neural_{name}_{h.label}",
                          {**nsummary, "latest_signal": nlatest})
            neural_info = {"summary": nsummary, "latest": nlatest}
        elif do_neural:
            print("  [neural] torch not installed - skipping TCN")

        out_per_h[h.label] = {"summary": summary, "latest": latest,
                              "neural": neural_info}
    return out_per_h


def main(argv: list[str]) -> None:
    do_daily = "--no-daily" not in argv
    do_intraday = "--no-intraday" not in argv
    do_neural = "--no-neural" not in argv
    targets = [a for a in argv[1:] if not a.startswith("--")] or list(ASSETS)
    cfg = RunConfig()

    data = None
    if do_daily:
        print(f"Loading daily data ({cfg.start} -> today)...")
        data = load_all(cfg, ASSETS)
        print(f"  prices: {data['prices'].shape}  fred: {data['fred'].shape}  "
              f"cot: {data['cot'].shape}")

    summary = {}
    for name in targets:
        if name not in ASSETS:
            print(f"unknown asset: {name}")
            continue
        summary.setdefault(name, {})
        if do_daily:
            summary[name]["daily"] = run_daily(name, data, cfg)
        if do_intraday:
            summary[name]["intraday"] = run_intraday(name, cfg, do_neural)

    (ART_DIR / "summary.json").write_text(
        json.dumps(_to_jsonable(summary), indent=2, default=str))
    print(f"\nArtifacts written to {ART_DIR}")


if __name__ == "__main__":
    main(sys.argv)
