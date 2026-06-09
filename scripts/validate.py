"""Validate the champion config and turn a good backtest into allocator-grade
evidence: does the edge hold in every regime, or does one period carry it?

Runs the sweep-winning config (cross-sectional momentum + real-yield macro tilt
+ cross-asset carry, vol-targeted, SPY-trend regime filter) over the full
2006-2026 cycle and reports:
  * per-regime Sharpe / CAGR / maxDD vs SPY (GFC, 2010s bull, COVID, 2022, recent)
  * rolling 1y Sharpe stability and turnover / cost drag
  * equity + drawdown curves written to artifacts/ (PNG if matplotlib is present,
    always CSV) so the protection is visible, not just tabulated.

Usage:
    python scripts/validate.py                 # champion @ tv0.20
    python scripts/validate.py --tv 0.15       # lower-vol / lower-drawdown variant
    python scripts/validate.py --regime-credit --fresh
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ART_DIR, ASSETS, RunConfig
from src.data import load_all
from src.portfolio import assemble_panel, portfolio_backtest
from src.research import series_stats

# Calendar regimes - chosen a priori (not fit), so a consistent edge across them
# is evidence the strategy generalizes rather than riding one lucky period.
REGIMES = [
    ("2006-2009 GFC", "2006-01-01", "2009-12-31"),
    ("2010-2019 bull", "2010-01-01", "2019-12-31"),
    ("2020 COVID", "2020-01-01", "2020-12-31"),
    ("2021-2022 bear", "2021-01-01", "2022-12-31"),
    ("2023+ recent", "2023-01-01", None),
    ("FULL", None, None),
]


def _flag(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def _champion() -> RunConfig:
    cfg = RunConfig()
    cfg.signal_weights = {"tsmom": 0.0, "xsmom": 1.0, "value": 0.0}
    cfg.macro_weight = 0.3
    cfg.carry_weight = 0.3
    cfg.portfolio_target_vol = 0.20
    cfg.regime_filter = True
    return cfg


def _slice(s: pd.Series, lo, hi) -> pd.Series:
    if lo is not None:
        s = s[s.index >= pd.Timestamp(lo)]
    if hi is not None:
        s = s[s.index <= pd.Timestamp(hi)]
    return s


def _fmt(st: dict) -> str:
    return f"{st['sharpe']:>6.2f}{st['cagr']:>8.1%}{st['maxdd']:>8.1%}"


def main(argv: list[str]) -> None:
    cfg = _champion()
    cfg.start = _flag(argv, "--start", cfg.start)
    cfg.portfolio_target_vol = float(_flag(argv, "--tv", cfg.portfolio_target_vol))
    cfg.rebalance_days = int(_flag(argv, "--rebal", cfg.rebalance_days))
    cfg.no_trade_band = float(_flag(argv, "--band", cfg.no_trade_band))
    if "--regime-credit" in argv:
        cfg.regime_credit = True
    fresh = "--fresh" in argv

    full = {n: ASSETS[n] for n in cfg.universe}
    print(f"Loading {len(full)} assets{' (fresh)' if fresh else ''}...")
    data = load_all(cfg, full, use_cache=not fresh)
    panel = assemble_panel(data, cfg)
    bt = portfolio_backtest(panel, cfg)
    if bt.empty or "spy_bh" not in bt:
        print("No PnL / no SPY benchmark produced."); return

    print(f"\n=== CHAMPION  xsmom + macro({cfg.macro_weight:g}) + carry"
          f"({cfg.carry_weight:g}), tv={cfg.portfolio_target_vol:.0%}, "
          f"regime={'trend+credit' if cfg.regime_credit else 'trend'} ===")
    print(f"Backtest {bt.index[0].date()} -> {bt.index[-1].date()}\n")

    # ---- Per-regime: strategy vs SPY, the core robustness check ----------------
    print(f"{'regime':<16}{'days':>6}    {'STRAT  Sh   CAGR   maxDD':<24}"
          f"  {'SPY    Sh   CAGR   maxDD':<24}")
    print("-" * 78)
    for name, lo, hi in REGIMES:
        pnl, spy = _slice(bt["pnl"], lo, hi), _slice(bt["spy_bh"], lo, hi)
        if len(pnl) < 20:
            continue
        ss, bs = series_stats(pnl), series_stats(spy)
        mark = "  <" if ss["sharpe"] > bs["sharpe"] and ss["maxdd"] > bs["maxdd"] else ""
        print(f"{name:<16}{len(pnl):>6}    {_fmt(ss):<24}  {_fmt(bs):<24}{mark}")
    print("  ('<' = strategy beat SPY on BOTH Sharpe and drawdown that regime)")

    # ---- Turnover, cost drag, rolling-Sharpe stability -------------------------
    yrs = len(bt) / 252.0
    ann_turn = bt["turnover"].sum() / yrs                 # sum|dw| per year (1x = 100%)
    ann_cost = bt["cost"].sum() / yrs                     # annual return lost to costs
    roll = bt["pnl"].rolling(252).mean() / bt["pnl"].rolling(252).std() * np.sqrt(252)
    roll = roll.dropna()
    spy_full = series_stats(bt["spy_bh"])["sharpe"]
    print(f"\nAnnual turnover ~{ann_turn:.1f}x   cost drag ~{ann_cost*1e4:.0f} bps/yr"
          f"   avg gross {bt['gross'].mean():.2f}x")
    if len(roll):
        print(f"Rolling 1y Sharpe: min {roll.min():.2f}  median {roll.median():.2f}"
              f"   >0: {(roll > 0).mean():.0%}   >SPY({spy_full:.2f}): "
              f"{(roll > spy_full).mean():.0%} of the time")

    # ---- Curves: PNG if matplotlib is around, always CSV ----------------------
    eq = pd.DataFrame({
        "strategy": (1 + bt["pnl"]).cumprod(),
        "SPY": (1 + bt["spy_bh"]).cumprod(),
    })
    if "gold_bh" in bt:
        eq["gold"] = (1 + bt["gold_bh"]).cumprod()
    dd = eq / eq.cummax() - 1.0
    csv = ART_DIR / "validation_equity.csv"
    eq.join(dd, rsuffix="_dd").to_csv(csv)
    print(f"\nEquity + drawdown curves -> {csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not installed - skipping PNG; plot the CSV instead)")
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    eq.plot(ax=a1, logy=True, lw=1.3)
    a1.set_title(f"Champion vs SPY  (tv={cfg.portfolio_target_vol:.0%}) - growth of $1, log")
    a1.set_ylabel("growth of $1"); a1.grid(True, alpha=0.3)
    dd[["strategy", "SPY"]].plot(ax=a2, lw=1.0)
    a2.set_title("Drawdown"); a2.set_ylabel("drawdown"); a2.grid(True, alpha=0.3)
    png = ART_DIR / "validation_curves.png"
    fig.tight_layout(); fig.savefig(png, dpi=120)
    print(f"Chart -> {png}")


if __name__ == "__main__":
    main(sys.argv)
