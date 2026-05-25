"""Multi-asset volatility-targeted trend portfolio.

For each asset in cfg.universe, train a per-asset vol forecast (LightGBM
ensemble), size a time-series-momentum position by target_vol / vol_fcst,
walk-forward backtest, and aggregate into an equal-risk-weighted portfolio.

Usage:
    python scripts/run_baseline.py              # full universe
    python scripts/run_baseline.py gold silver  # subset by name
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import aggregate_portfolio, evaluate, walk_forward_vol_daily
from src.config import ART_DIR, ASSETS, RunConfig
from src.data import load_all
from src.features import build_daily_features
from src.model import (explain_primary, predict_vol_multi_horizon,
                       train_vol_multi_horizon)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="list")
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def _save_metrics(name: str, payload: dict) -> None:
    (ART_DIR / f"metrics_{name}.json").write_text(
        json.dumps(_to_jsonable(payload), indent=2, default=str))


def _vol_signal(feat_row: pd.Series, fc_row: pd.Series, cfg: RunConfig) -> dict:
    trend = float(feat_row["trend_signal"])
    vol_fcst = float(fc_row["vol_fcst"])
    ratio = min(max(cfg.target_vol / vol_fcst, 0.0), cfg.max_leverage)
    pos = trend * ratio
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


def run_daily(name: str, data: dict, cfg: RunConfig) -> tuple[dict, pd.DataFrame]:
    asset = ASSETS[name]
    print(f"\n=== {name.upper()} ({asset.ticker}) ===")
    feats = build_daily_features(data, asset, cfg)
    if feats.empty:
        print(f"  no feature matrix - skipping {name}")
        return {"empty": True}, pd.DataFrame()
    print(f"  feature matrix: {feats.shape}, "
          f"span {feats.index.min().date()} -> {feats.index.max().date()}")

    bt = walk_forward_vol_daily(feats, cfg)
    summary = {"empty": True}
    enriched = pd.DataFrame()
    if not bt.empty:
        summary, enriched = evaluate(bt, cfg, holding=cfg.backtest_horizon)
        enriched.to_parquet(ART_DIR / f"daily_predictions_{name}.parquet")
        print(f"  strategy Sharpe={summary['strategy']['sharpe']:+.2f}  "
              f"b&h Sharpe={summary['benchmark']['sharpe']:+.2f}  "
              f"vol_r2={summary.get('vol_r2', float('nan')):+.3f}  "
              f"hit={summary.get('hit_rate', 0):.3f}")

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

    _save_metrics(f"daily_{name}", {**summary, "latest_signal": latest_signal,
                                    "ticker": asset.ticker,
                                    "asset_class": asset.asset_class})
    return summary, enriched


def main(argv: list[str]) -> None:
    cfg = RunConfig()
    requested = [a for a in argv[1:] if not a.startswith("--")]
    universe = requested or cfg.universe
    universe = [a for a in universe if a in ASSETS]
    if not universe:
        print(f"no valid assets in: {requested}")
        return

    sub_assets = {n: ASSETS[n] for n in universe}
    print(f"Loading data for {len(universe)} assets ({cfg.start} -> today)...")
    data = load_all(cfg, sub_assets)
    print(f"  prices: {data['prices'].shape}  fred: {data['fred'].shape}  "
          f"cot: {data['cot'].shape}")

    per_asset_pred: dict[str, pd.DataFrame] = {}
    per_asset_summary: dict[str, dict] = {}
    for name in universe:
        summary, enriched = run_daily(name, data, cfg)
        per_asset_summary[name] = summary
        if not enriched.empty:
            per_asset_pred[name] = enriched

    print(f"\n=== PORTFOLIO (N={len(per_asset_pred)}, scale={cfg.portfolio_scale}) ===")
    port_metrics, port_pred = aggregate_portfolio(per_asset_pred, cfg)
    if not port_pred.empty:
        port_pred.to_parquet(ART_DIR / "daily_predictions_portfolio.parquet")
        s = port_metrics["strategy"]
        b = port_metrics["benchmark"]
        print(f"  portfolio Sharpe={s['sharpe']:+.2f}  "
              f"CAGR={s['cagr']:+.1%}  max_dd={s['max_dd']:.1%}")
        print(f"  equal-weight b&h Sharpe={b['sharpe']:+.2f}  "
              f"CAGR={b['cagr']:+.1%}  max_dd={b['max_dd']:.1%}")
        if "vt_benchmark" in port_metrics:
            v = port_metrics["vt_benchmark"]
            print(f"  vol-tgt eq-wt b&h Sharpe={v['sharpe']:+.2f}  "
                  f"CAGR={v['cagr']:+.1%}")
    _save_metrics("portfolio", port_metrics)

    (ART_DIR / "summary.json").write_text(
        json.dumps(_to_jsonable({"universe": universe,
                                 "portfolio": port_metrics,
                                 "per_asset": per_asset_summary}),
                   indent=2, default=str))
    print(f"\nArtifacts written to {ART_DIR}")


if __name__ == "__main__":
    main(sys.argv)
