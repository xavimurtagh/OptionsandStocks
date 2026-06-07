"""Empirically rank the existing strategy levers before writing new code.

Trains the vol model once per purged fold, then scores a grid of configs
(signal weights x long-only/long-short x threshold x target-vol) on the same
forecasts. Reports OOS CAGR/Sharpe/Calmar with a Deflated Sharpe Ratio that is
corrected for the number of configs tried, plus a sweep-wide PBO. Use it to see
which knobs actually help on gold/silver vs the always-long vt-B&H benchmark.

Usage:
    python scripts/sweep.py                 # gold silver, default grid
    python scripts/sweep.py gold --fast     # 1 ensemble member, 5d horizon
    python scripts/sweep.py gold silver spy --splits 6 --top 12
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ASSETS, RunConfig
from src.data import load_all
from src.features import (build_daily_features, cross_sectional_momentum,
                          cross_sectional_value)
from src.research import lever_sweep


def _flag(argv, flag, default):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def main(argv: list[str]) -> None:
    fast = "--fast" in argv
    n_splits = int(_flag(argv, "--splits", 5))
    top = int(_flag(argv, "--top", 10))
    names = [a for a in argv[1:] if not a.startswith("--")] or ["gold", "silver"]
    names = [n for n in names if n in ASSETS]

    cfg = RunConfig()
    if fast:
        cfg.n_ensemble = 1
        cfg.daily_horizons = [5]

    full_assets = {n: ASSETS[n] for n in cfg.universe}
    print(f"Loading data for {len(full_assets)} assets...")
    data = load_all(cfg, full_assets)
    if data["fred"].empty:
        print("[WARN] FRED macro features unavailable - gold/silver degraded; "
              "re-run when fred.stlouisfed.org is reachable.")
    tickers = [a.ticker for a in full_assets.values()]
    data["xsmom"] = cross_sectional_momentum(data["prices"], tickers, cfg.xsmom_lookback)
    data["value"] = cross_sectional_value(data["prices"], tickers, cfg.value_lookback)

    cols = ["sharpe", "cagr", "maxdd", "calmar", "dsr"]
    for name in names:
        asset = ASSETS[name]
        feats = build_daily_features(data, asset, cfg)
        if feats.empty:
            print(f"\n{name}: no feature matrix"); continue
        summary, pbo = lever_sweep(feats, cfg, asset.ticker, n_splits=n_splits)
        if summary.empty:
            print(f"\n{name}: insufficient data"); continue

        vtbh_cagr = summary.loc["[vt-B&H]", "cagr"] if "[vt-B&H]" in summary.index else float("nan")
        print(f"\n=== {name.upper()} ({asset.ticker})  "
              f"sweep PBO={pbo:.2f}  (n_configs swept={len(summary) - 2}) ===")
        print(f"{'config':<22}{'Sharpe':>8}{'CAGR':>8}{'maxDD':>8}"
              f"{'Calmar':>8}{'DSR':>7}")
        print("-" * 61)
        refs = summary.loc[[i for i in ("[B&H]", "[vt-B&H]") if i in summary.index]]
        body = summary.drop(index=refs.index, errors="ignore").head(top)
        for tbl in (refs, body):
            for cname, r in tbl.iterrows():
                star = " *" if (r["cagr"] > vtbh_cagr and r["dsr"] >= 0.95) else ""
                print(f"{cname:<22}{r['sharpe']:>8.2f}{r['cagr']:>8.1%}"
                      f"{r['maxdd']:>8.1%}{r['calmar']:>8.2f}{r['dsr']:>7.2f}{star}")
        winners = body[(body["cagr"] > vtbh_cagr) & (body["dsr"] >= 0.95)]
        if winners.empty:
            print("  -> no config beats vt-B&H CAGR with DSR>=0.95 "
                  "(honest read: existing levers don't add return here).")
        else:
            print(f"  -> {len(winners)} config(s) beat vt-B&H with real edge (*).")


if __name__ == "__main__":
    main(sys.argv)
