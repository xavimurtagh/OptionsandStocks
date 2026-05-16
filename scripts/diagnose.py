"""Skill diagnostics for an already-run backtest.

Reads the prediction parquets in artifacts/ and reports whether each model
genuinely beats chance - no re-training, runs in seconds.

Usage:
    python scripts/diagnose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / "artifacts"
COIN_FLIP_LOG_LOSS = 0.6931


def _verdict(auc: float, ll: float, monotonic: bool) -> str:
    if auc >= 0.55 and ll < COIN_FLIP_LOG_LOSS and monotonic:
        return "EDGE - worth forward paper-trading"
    if auc >= 0.52 and ll < COIN_FLIP_LOG_LOSS:
        return "MARGINAL - weak signal, not yet tradeable"
    return "NO EDGE - do not trade; calibration/confidence not trustworthy"


def diagnose(label: str, path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_parquet(path)
    need = {"prob_up", "target_ret"}
    if not need.issubset(df.columns):
        print(f"\n## {label}: missing columns, skipped")
        return
    d = df.dropna(subset=["prob_up", "target_ret"]).copy()
    if len(d) < 50:
        print(f"\n## {label}: only {len(d)} rows, too few to judge")
        return

    up = (d["target_ret"] > 0).astype(int)
    base = up.mean()
    p = d["prob_up"].clip(1e-4, 1 - 1e-4)

    auc = roc_auc_score(up, p) if up.nunique() > 1 else float("nan")
    ll = log_loss(up, p)
    brier = brier_score_loss(up, p)
    model_hit = ((np.sign(d["prob_up"] - 0.5)) == np.sign(d["target_ret"])).mean()
    always_long_hit = base

    print(f"\n## {label}   ({len(d)} predictions)")
    print(f"  base rate P(up)        : {base:.3f}")
    print(f"  AUC (prob_up vs up)    : {auc:.3f}   (0.50 = no skill)")
    print(f"  log loss               : {ll:.3f}   (coin flip = {COIN_FLIP_LOG_LOSS})")
    print(f"  Brier                  : {brier:.3f}")
    print(f"  model hit rate         : {model_hit:.3f}")
    print(f"  always-long hit rate   : {always_long_hit:.3f}   "
          f"<- the benchmark to beat")

    # Calibration: are predicted probabilities honest?
    print("  calibration (predicted -> actual):")
    try:
        d["pbin"] = pd.qcut(p, 5, duplicates="drop")
        g = d.groupby("pbin", observed=True)
        cal = pd.DataFrame({
            "pred": g["prob_up"].mean(),
            "actual": g["target_ret"].apply(lambda s: (s > 0).mean()),
            "n": g.size(),
        })
        for _, r in cal.iterrows():
            print(f"    pred {r['pred']:.2f} -> actual {r['actual']:.2f}  "
                  f"(n={int(r['n'])}, gap {r['actual'] - r['pred']:+.2f})")
    except ValueError:
        print("    (not enough spread in probabilities)")

    # Confidence monotonicity: should hit rate rise with confidence?
    monotonic = True
    if "confidence" in d.columns:
        d["cbin"] = pd.cut(d["confidence"], [-0.01, 0.1, 0.3, 0.6, 1.01],
                           labels=["very_low", "low", "med", "high"])
        hits = (np.sign(d["prob_up"] - 0.5) == np.sign(d["target_ret"]))
        conf = d.assign(hit=hits).groupby("cbin", observed=True)["hit"].agg(
            ["mean", "size"])
        print("  hit rate by confidence bucket:")
        vals = []
        for idx, r in conf.iterrows():
            print(f"    {idx:<9}: {r['mean']:.3f}  (n={int(r['size'])})")
            vals.append(r["mean"])
        monotonic = all(x <= y + 0.02 for x, y in zip(vals, vals[1:])) \
            if len(vals) > 1 else True
        if not monotonic:
            print("    !! confidence does NOT track accuracy - sizing is unsafe")

    if {"pnl", "bh_pnl"}.issubset(d.columns):
        strat_ret = (1 + d["pnl"].fillna(0)).prod() - 1
        bh_ret = (1 + d["bh_pnl"].fillna(0)).prod() - 1
        print(f"  total return  strategy {strat_ret:+.2%}   "
              f"buy&hold {bh_ret:+.2%}")

    print(f"  VERDICT: {_verdict(auc, ll, monotonic)}")


def main() -> None:
    if not ART_DIR.exists():
        print("No artifacts/ directory - run scripts/run_baseline.py first")
        sys.exit(1)
    print("=" * 64)
    print("BACKTEST SKILL DIAGNOSTICS")
    print("=" * 64)
    found = False
    for asset in ("gold", "silver"):
        diagnose(f"{asset} daily",
                 ART_DIR / f"daily_predictions_{asset}.parquet")
        for lbl in ("1h", "15m"):
            diagnose(f"{asset} intraday {lbl} (LightGBM)",
                     ART_DIR / f"intraday_predictions_{asset}_{lbl}.parquet")
            diagnose(f"{asset} intraday {lbl} (neural TCN)",
                     ART_DIR / f"neural_predictions_{asset}_{lbl}.parquet")
        found = True
    if found:
        print("\n" + "=" * 64)
        print("Reminder: AUC ~0.50 means the features carry no directional")
        print("signal at that horizon. Calibration/meta fixes make the model")
        print("HONEST, not profitable - profit needs a real AUC > ~0.53.")
        print("=" * 64)


if __name__ == "__main__":
    main()
