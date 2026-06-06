"""Multi-asset volatility-targeted trend portfolio.

For each asset in cfg.universe, train a per-asset vol forecast (LightGBM
ensemble) and combine three signals (per-asset TSMOM, cross-sectional
momentum, cross-sectional value), gated by long-only + magnitude threshold,
then size the position by target_vol / vol_fcst. Aggregates equal-risk
into a portfolio.

Cached per-asset parquets from pre-combine schema are auto-detected
(missing combined_signal column) and force-retrained.

Usage:
    python scripts/run_baseline.py              # full universe (resumes if interrupted)
    python scripts/run_baseline.py gold silver  # subset by name
    python scripts/run_baseline.py --fresh      # discard cached per-asset artifacts
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
from src.features import (build_daily_features, cross_sectional_momentum,
                          cross_sectional_value)
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
    xsmom = float(feat_row.get("xsmom_signal", 0.0) or 0.0)
    value = float(feat_row.get("value_signal", 0.0) or 0.0)
    w = cfg.signal_weights
    combined = (w.get("tsmom", 1.0) * trend
                + w.get("xsmom", 0.0) * xsmom
                + w.get("value", 0.0) * value)
    if cfg.long_only:
        combined = max(combined, 0.0)
    if abs(combined) < cfg.signal_threshold:
        combined = 0.0
    vol_fcst = float(fc_row["vol_fcst"])
    ratio = min(max(cfg.target_vol / vol_fcst, 0.0), cfg.max_leverage)
    pos = combined * ratio
    direction = "LONG" if combined > 0.05 else "SHORT" if combined < -0.05 else "FLAT"
    return {
        "asof": str(feat_row.name),
        "trend_signal": trend,
        "xsmom_signal": xsmom,
        "value_signal": value,
        "combined_signal": combined,
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
        summary, enriched = evaluate(bt, cfg, holding=cfg.backtest_horizon,
                                     ticker=asset.ticker)
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
    fresh = "--fresh" in argv
    requested = [a for a in argv[1:] if not a.startswith("--")]
    universe = requested or cfg.universe
    universe = [a for a in universe if a in ASSETS]
    if not universe:
        print(f"no valid assets in: {requested}")
        return
    if fresh:
        print("--fresh: removing existing per-asset artifacts")
        for name in universe:
            for p in (ART_DIR / f"daily_predictions_{name}.parquet",
                      ART_DIR / f"metrics_daily_{name}.json",
                      ART_DIR / f"feature_importance_{name}.parquet"):
                p.unlink(missing_ok=True)

    sub_assets = {n: ASSETS[n] for n in universe}
    # Identify which assets still need a backtest run. Anything with a cached
    # predictions parquet is loaded as-is unless it's from an older signal
    # schema (missing combined_signal column). Pass --fresh to retrain all.
    todo, done = [], []
    for n in universe:
        p = ART_DIR / f"daily_predictions_{n}.parquet"
        if not p.exists():
            todo.append(n)
            continue
        try:
            cached_cols = set(pd.read_parquet(p, columns=None).columns)
        except Exception:
            todo.append(n)
            continue
        if "combined_signal" not in cached_cols:
            print(f"  [stale schema] {n} cached without combined_signal -- will retrain")
            todo.append(n)
        else:
            done.append(n)
    if done:
        print(f"resuming: {len(done)} cached  ({', '.join(done)})")
    print(f"to train: {len(todo)}  ({', '.join(todo) if todo else 'none'})")

    data = None
    if todo:
        # Always load the full universe so cross-sectional signals have
        # context across all 16 assets, even when retraining a subset.
        full_assets = {n: ASSETS[n] for n in cfg.universe}
        print(f"Loading data for {len(cfg.universe)} assets "
              f"(XSMOM/value need full-universe context)...")
        data = load_all(cfg, full_assets)
        print(f"  prices: {data['prices'].shape}  fred: {data['fred'].shape}  "
              f"cot: {data['cot'].shape}")
        tickers = [a.ticker for a in full_assets.values()]
        data["xsmom"] = cross_sectional_momentum(
            data["prices"], tickers, cfg.xsmom_lookback)
        data["value"] = cross_sectional_value(
            data["prices"], tickers, cfg.value_lookback)
        print(f"  xsmom: {data['xsmom'].shape}  value: {data['value'].shape}")

    per_asset_pred: dict[str, pd.DataFrame] = {}
    per_asset_summary: dict[str, dict] = {}
    for name in universe:
        cache = ART_DIR / f"daily_predictions_{name}.parquet"
        if name in done:
            try:
                per_asset_pred[name] = pd.read_parquet(cache)
                meta = json.loads((ART_DIR / f"metrics_daily_{name}.json").read_text()) \
                    if (ART_DIR / f"metrics_daily_{name}.json").exists() else {}
                per_asset_summary[name] = {k: v for k, v in meta.items()
                                           if k != "latest_signal"}
                print(f"  [cached] {name}")
            except Exception as e:
                print(f"  [cache failed] {name}: {e} -- will retrain")
                done.remove(name)
                todo.append(name)
        if name in todo:
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
