"""End-to-end gold/silver baseline.

Usage:
    python scripts/run_baseline.py            # both assets
    python scripts/run_baseline.py gold
    python scripts/run_baseline.py silver

Outputs:
    data_cache/predictions_<asset>.parquet
    data_cache/equity_<asset>.csv
    Prints metrics + latest trade signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import evaluate, kelly_size, walk_forward
from src.config import ASSETS, DATA_DIR, RunConfig
from src.data import load_all
from src.features import build_features
from src.model import predict_ensemble, train_ensemble


def run_asset(name: str, data: dict, cfg: RunConfig) -> None:
    asset = ASSETS[name]
    print(f"\n=== {name.upper()} ({asset.ticker}) ===")
    feats = build_features(data, asset, cfg)
    print(f"feature matrix: {feats.shape}, span {feats.index.min().date()} -> {feats.index.max().date()}")

    bt = walk_forward(feats, cfg)
    metrics = evaluate(bt, cfg)

    print("\n-- backtest metrics --")
    for k, v in metrics.items():
        if k == "by_confidence":
            continue
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n-- pnl by confidence bucket --")
    print(metrics["by_confidence"])

    bt.to_parquet(DATA_DIR / f"predictions_{name}.parquet")
    equity = (1 + bt["target_ret"] * kelly_size(
        bt["prob_up"].values, bt["confidence"].values, cfg.kelly_fraction
    )).cumprod()
    equity.to_csv(DATA_DIR / f"equity_{name}.csv")

    labeled = feats.dropna(subset=["target_ret"])
    latest = feats.iloc[[-1]]
    models, iso, cols = train_ensemble(labeled, n_models=cfg.n_ensemble)
    pred_today = predict_ensemble(models, iso, cols, latest)
    today = pred_today.iloc[0]
    pos = kelly_size(pred_today["prob_up"].values,
                     pred_today["confidence"].values,
                     cfg.kelly_fraction)[0]
    print(f"\n-- latest signal ({latest.index[-1].date()}) --")
    print(f"  prob_up:    {today['prob_up']:.3f}")
    print(f"  prob_std:   {today['prob_std']:.3f}")
    print(f"  confidence: {today['confidence']:.3f}")
    print(f"  position (fractional Kelly @ {cfg.kelly_fraction}): {pos:+.3f}")


def main(argv: list[str]) -> None:
    cfg = RunConfig()
    print(f"Loading data ({cfg.start} -> today)...")
    data = load_all(cfg, ASSETS)
    print(f"  prices: {data['prices'].shape}")
    print(f"  fred:   {data['fred'].shape}")
    print(f"  cot:    {data['cot'].shape}")
    targets = argv[1:] or list(ASSETS.keys())
    for name in targets:
        if name not in ASSETS:
            print(f"unknown asset: {name}; skipping")
            continue
        run_asset(name, data, cfg)


if __name__ == "__main__":
    main(sys.argv)
