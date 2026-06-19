"""Deployment frontier: the search is over, this sizes the validated edge.

The cross-asset audit (scripts/uk_robustness.py) confirmed trend-gated,
vol-stepped leverage generalizes - it beat buy-and-hold in both halves on both
the Nasdaq and the S&P, after the real-product cost haircut, and its gate turned
constant-3x's -6%/yr dot-com+GFC catastrophe into +6.7%/yr. Two problems remain,
and both are addressed by *sizing and diversifying*, not by new signals:

  * single-index risk - one sleeve bets everything on one index's regime, so we
    run TWO validated sleeves (Nasdaq + S&P) and split risk across them;
  * a -60%+ drawdown even gated - so we blend the leveraged satellite with the
    unleveraged trend-gated core and let the user choose where on the
    return/drawdown frontier they actually sit.

This prints that frontier: each blend of core (unleveraged trend, diversified)
and satellite (gated leverage, diversified) with its CAGR, worst-case drawdown,
regime split, and terminal wealth - then flags the blend whose max drawdown is
closest to --maxdd (the number you can hold through without capitulating).

Usage:
    python scripts/uk_deploy.py --fresh                 # 3x satellite, 3.4% drag
    python scripts/uk_deploy.py --levmax 2 --maxdd 0.40
    python scripts/uk_deploy.py --etp-drag 0.034 --split 2014
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ART_DIR
from src.data import load_prices
from src.research import series_stats
from src.uk import gated_leverage_sleeve, rotation_backtest, trend_state

TICKERS = ["QQQ", "SPY", "GLD", "^IRX"]
WEIGHTS = [0.0, 0.25, 0.50, 0.75, 1.0]   # satellite share; rest in core


def _flag(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def main(argv: list[str]) -> None:
    fresh = "--fresh" in argv
    levmax = float(_flag(argv, "--levmax", 3.0))
    drag = float(_flag(argv, "--etp-drag", 0.034))
    split = str(_flag(argv, "--split", "2014")) + "-01-01"
    maxdd_target = float(_flag(argv, "--maxdd", 0.40))
    vlo = float(_flag(argv, "--vlo", 0.20))
    vhi = float(_flag(argv, "--vhi", 0.28))

    print(f"Loading {len(TICKERS)} tickers{' (fresh)' if fresh else ''}...")
    px = load_prices(TICKERS, "1998-08-01", use_cache=not fresh)
    close = {t: px[f"{t}_close"].dropna() for t in TICKERS if f"{t}_close" in px}
    irx = close.get("^IRX", pd.Series(dtype=float))
    if len(irx) and irx.median() > 20:
        irx = irx / 10.0

    def sleeves(idx_tk):
        idx = close[idx_tk]
        ret = idx.pct_change()
        rf_ann = (irx / 100.0).reindex(ret.index).ffill().fillna(0.02)
        gold = close["GLD"].pct_change().reindex(ret.index).fillna(rf_ann / 252.0)
        core = rotation_backtest(ret, rf_ann / 252.0,
                                 trend_state(idx, 200), lag=2,
                                 cost_risk=0.0007, cost_safe=0.0)["pnl"]
        sat = gated_leverage_sleeve(idx, ret, rf_ann, gold, lev_max=levmax,
                                    vlo=vlo, vhi=vhi, extra_drag=drag)
        return core, sat

    ndx_core, ndx_sat = sleeves("QQQ")
    spx_core, spx_sat = sleeves("SPY")
    # Diversify each layer 50/50 across the two indices.
    core = (0.5 * ndx_core + 0.5 * spx_core).dropna()
    sat = (0.5 * ndx_sat + 0.5 * spx_sat).dropna()
    idx = core.index.intersection(sat.index)
    core, sat = core.loc[idx], sat.loc[idx]

    print(f"\n=== DEPLOYMENT FRONTIER  (satellite = {levmax:g}x gated, "
          f"diversified NDX+SPX; drag {drag:.1%}/yr) ===")
    print(f"{start_end(idx)}   gate {vlo:.0%}/{vhi:.0%}, split {split[:4]}\n")
    hdr = (f"{'sat %':>6}{'CAGR':>8}{'maxDD':>8}{'Sharpe':>8}"
           f"{'pre' + split[:4]:>9}{split[:4] + '+':>8}{'£10k ->':>12}")
    print(hdr); print("-" * len(hdr))

    rows = {}
    for w in WEIGHTS:
        pnl = (1 - w) * core + w * sat
        st = series_stats(pnl)
        pre = series_stats(pnl[pnl.index < pd.Timestamp(split)])["cagr"]
        post = series_stats(pnl[pnl.index >= pd.Timestamp(split)])["cagr"]
        wealth = 10_000 * float((1 + pnl).prod())
        rows[w] = {"cagr": st["cagr"], "maxdd": st["maxdd"], "sharpe": st["sharpe"],
                   "pre": pre, "post": post, "wealth": wealth}
        print(f"{w * 100:>5.0f}%{st['cagr']:>8.1%}{st['maxdd']:>8.1%}"
              f"{st['sharpe']:>8.2f}{pre:>9.1%}{post:>8.1%}{wealth:>12,.0f}")

    # Pick the blend whose worst-case drawdown is closest to what you can hold.
    pick = min(rows, key=lambda w: abs(abs(rows[w]["maxdd"]) - maxdd_target))
    r = rows[pick]
    print(f"\nFor a max drawdown near {maxdd_target:.0%} you can actually hold: "
          f"{pick * 100:.0f}% satellite / {100 - pick * 100:.0f}% core")
    print(f"  -> CAGR {r['cagr']:.1%}, worst drawdown {r['maxdd']:.0%}, "
          f"£10k -> £{r['wealth']:,.0f} over {len(idx) / 252:.0f}y")
    print(f"  Deployable as: core = 50/50 EQQQ+CSPX on a 200d trend (cash when "
          f"below); satellite = 50/50 QQQ3+3USL gated by vol, SGLN when off.")
    print(f"  Reminder: even the all-core 0% line has a deep equity drawdown; "
          f"leverage only ever widens it. Choose the row, not the dream.")

    out = pd.DataFrame(rows).T
    out.index.name = "satellite_weight"
    csv = ART_DIR / "uk_deploy_frontier.csv"
    out.to_csv(csv)
    print(f"\nFrontier -> {csv}")


def start_end(idx: pd.Index) -> str:
    return f"{idx[0].date()} -> {idx[-1].date()} ({len(idx) / 252:.1f}y)"


if __name__ == "__main__":
    main(sys.argv)
