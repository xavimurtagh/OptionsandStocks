"""Where does the champion lose? Per-asset and per-sleeve P&L attribution across
the windows the validation flagged as weak: the 2010s bull, the 2022-23 momentum
whipsaw (the -22% drawdown), and the 2023+ mega-cap bull lag.

For each window it reports:
  * per-asset contribution (which sleeves/markets bled), worst first
  * per-sleeve marginal P&L: champion minus [macro off / carry off / regime off],
    i.e. did each overlay help or hurt that window?
  * signal sign-flips per asset per year (a direct whipsaw gauge)

so we can target a fix (option b) at the real leak instead of guessing.

Usage:
    python scripts/attribution.py                 # champion @ tv0.15, weekly
    python scripts/attribution.py --tv 0.20 --rebal 1
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ASSETS, RunConfig
from src.data import load_all
from src.portfolio import (_target_weights, assemble_panel, combined_signal_panel,
                           portfolio_backtest)

WINDOWS = [
    ("FULL", None, None),
    ("2010s bull", "2010-01-01", "2019-12-31"),
    ("2022-23 whipsaw", "2022-06-01", "2023-10-31"),
    ("2023+ recent", "2023-01-01", None),
]


def _flag(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def _champion(argv) -> RunConfig:
    cfg = RunConfig()
    cfg.signal_weights = {"tsmom": 0.0, "xsmom": 1.0, "value": 0.0}
    cfg.macro_weight = 0.3
    cfg.carry_weight = 0.3
    cfg.regime_filter = True
    cfg.start = _flag(argv, "--start", cfg.start)
    cfg.portfolio_target_vol = float(_flag(argv, "--tv", 0.15))
    cfg.rebalance_days = int(_flag(argv, "--rebal", 5))
    drop = _flag(argv, "--drop", "")
    if drop:
        cfg.universe = [u for u in cfg.universe if u not in drop.split(",")]
    return cfg


def _win(s: pd.Series, lo, hi) -> pd.Series:
    if lo is not None:
        s = s[s.index >= pd.Timestamp(lo)]
    if hi is not None:
        s = s[s.index <= pd.Timestamp(hi)]
    return s


def main(argv: list[str]) -> None:
    cfg = _champion(argv)
    full = {n: ASSETS[n] for n in cfg.universe}
    print(f"Loading {len(full)} assets...")
    data = load_all(cfg, full, use_cache="--fresh" not in argv)
    panel = assemble_panel(data, cfg)
    rets = panel["ret"]
    w = _target_weights(panel, cfg)
    pa = (w.shift(1) * rets).dropna(how="all")           # per-asset daily pnl (gross)
    live = pa.index[w.abs().sum(axis=1).reindex(pa.index).fillna(0) > 1e-9]
    pa = pa.loc[live.min():] if len(live) else pa

    print(f"\n=== ATTRIBUTION  champion tv={cfg.portfolio_target_vol:.0%}, "
          f"rebal={cfg.rebalance_days}d ===")
    print(f"Book {pa.index[0].date()} -> {pa.index[-1].date()}\n")

    # ---- Per-asset contribution (summed daily pnl, percentage points) ----------
    hdr = f"{'asset':<7}" + "".join(f"{n:>16}" for n, _, _ in WINDOWS)
    print("Per-asset P&L contribution (sum of daily returns, pts):")
    print(hdr); print("-" * len(hdr))
    tbl = {n: _win(pa, lo, hi).sum() * 100 for n, lo, hi in WINDOWS}
    tbl = pd.DataFrame(tbl).sort_values("2022-23 whipsaw")
    for asset, r in tbl.iterrows():
        print(f"{asset:<7}" + "".join(f"{r[n]:>16.1f}" for n, _, _ in WINDOWS))
    print(f"{'TOTAL':<7}" + "".join(f"{tbl[n].sum():>16.1f}" for n, _, _ in WINDOWS))

    # ---- Per-sleeve marginal P&L (champion minus sleeve-off), net -------------
    base = portfolio_backtest(panel, cfg)["pnl"]
    variants = {
        "macro tilt": replace(cfg, macro_weight=0.0),
        "carry": replace(cfg, carry_weight=0.0),
        "regime filter": replace(cfg, regime_filter=False, regime_credit=False),
    }
    print("\nPer-sleeve marginal P&L = champion - (sleeve off), pts "
          "(+ = sleeve helped):")
    print(hdr.replace("asset", "sleeve")); print("-" * len(hdr))
    for name, vcfg in variants.items():
        d = (base - portfolio_backtest(panel, vcfg)["pnl"]).dropna()
        print(f"{name:<7}" + "".join(f"{_win(d, lo, hi).sum()*100:>16.1f}"
                                     for n, lo, hi in WINDOWS))

    # ---- Whipsaw gauge: combined-signal sign flips per asset per year ----------
    sig = combined_signal_panel(panel, cfg).reindex(pa.index)
    flips = (np.sign(sig).diff().abs() > 0).sum()           # total flips per asset
    yrs = len(pa) / 252.0
    print(f"\nSignal sign-flips per asset per year (whipsaw gauge), avg "
          f"{flips.mean()/yrs:.1f}:")
    fp = (flips / yrs).sort_values(ascending=False)
    print("  " + "  ".join(f"{a}:{v:.1f}" for a, v in fp.head(8).items()))


if __name__ == "__main__":
    main(sys.argv)
