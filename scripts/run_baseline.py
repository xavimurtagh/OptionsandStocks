"""End-to-end gold/silver pipeline.

Daily volatility-targeted trend-following model: a multi-horizon LightGBM
ensemble forecasts forward realized volatility, which sizes a time-series
momentum position. The retired direction classifier / meta-labelling / TCN
remain on disk but are opt-in only.

Usage:
    python scripts/run_baseline.py              # both assets, daily model
    python scripts/run_baseline.py gold
    python scripts/run_baseline.py --intraday   # also run retired intraday
    python scripts/run_baseline.py --intraday --neural

Outputs land in artifacts/ for the Streamlit UI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import (evaluate, kelly_size, walk_forward_intraday,
                          walk_forward_vol_daily)
from src.config import ART_DIR, ASSETS, RunConfig
from src.data import load_all, load_intraday
from src.features import build_daily_features, build_intraday_features
from src.meta import MetaStrategy
from src.model import (explain_primary, predict_vol_multi_horizon,
                       train_vol_multi_horizon)
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


def _vol_signal(feat_row: pd.Series, fc_row: pd.Series, cfg: RunConfig) -> dict:
    trend = float(feat_row["trend_signal"])
    vol_fcst = float(fc_row["vol_fcst"])
    ratio = min(max(cfg.target_vol / vol_fcst, 0.0), cfg.max_leverage)
    pos = trend * ratio
    if (cfg.vrp_filter and "opt_iv" in feat_row.index
            and pd.notna(feat_row["opt_iv"])
            and feat_row["opt_iv"] / vol_fcst > 1.5):
        pos *= 0.5
    direction = "LONG" if trend > 0.05 else "SHORT" if trend < -0.05 else "FLAT"
    return {
        "asof": str(feat_row.name),
        "trend_signal": trend,
        "trend_direction": direction,
        "vol_forecast": vol_fcst,
        "realized_vol_20d": float(feat_row.get("rv_20d", float("nan"))),
        "target_position": pos,
        "by_horizon": {
            str(h): {
                "vol_fcst": float(fc_row[f"vol_fcst_{h}"]),
                "vol_fcst_std": float(fc_row[f"vol_fcst_std_{h}"]),
            }
            for h in cfg.daily_horizons if f"vol_fcst_{h}" in fc_row.index
        },
    }


def run_daily(name: str, data: dict, cfg: RunConfig) -> dict:
    asset = ASSETS[name]
    print(f"\n=== DAILY {name.upper()} ({asset.ticker}) ===")
    feats = build_daily_features(data, asset, cfg)
    print(f"feature matrix: {feats.shape}, "
          f"span {feats.index.min().date()} -> {feats.index.max().date()}")

    bt = walk_forward_vol_daily(feats, cfg)
    summary = {"empty": True}
    if not bt.empty:
        summary, enriched = evaluate(bt, cfg, holding=cfg.backtest_horizon,
                                     periods_per_year=252)
        enriched.to_parquet(ART_DIR / f"daily_predictions_{name}.parquet")
        print("-- strategy     --", summary["strategy"])
        print("-- buy & hold   --", summary["benchmark"])
        if "vt_benchmark" in summary:
            print("-- vol-tgt b&h  --", summary["vt_benchmark"])
        print(f"vol_r2={summary.get('vol_r2', float('nan')):.3f} "
              f"vol_corr={summary.get('vol_corr', float('nan')):.3f} "
              f"trend_hit={summary.get('hit_rate', 0):.3f}")

    rv_col = f"fwd_rv_{cfg.backtest_horizon}d"
    models = train_vol_multi_horizon(feats.dropna(subset=[rv_col]),
                                     cfg.daily_horizons, n_models=cfg.n_ensemble,
                                     device=cfg.device)
    latest_signal = {}
    if models:
        latest_row = feats.iloc[[-1]]
        fc_row = predict_vol_multi_horizon(models, latest_row).iloc[0]
        latest_signal = _vol_signal(feats.iloc[-1], fc_row, cfg)
        latest_signal["drivers"] = [
            {"feature": f, "contribution": c}
            for f, c in explain_primary(models, latest_row)
        ]
        fi = pd.concat({h: hm.feature_importance
                        for h, hm in models.items()}, axis=1)
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
    # Daily vol-targeted trend model is the pipeline. Intraday and the neural
    # TCN are retired direction models - opt in explicitly to run them.
    do_daily = "--no-daily" not in argv
    do_intraday = "--intraday" in argv
    do_neural = "--neural" in argv
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
