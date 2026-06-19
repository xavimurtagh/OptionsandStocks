"""Adversarial robustness audit for the trend-gated leverage candidates.

uk_max_wealth.py answers "what won the backtest". This script asks the three
questions that decide whether any of it survives contact with reality:

  1. REGIME SPLIT - is the edge real or just the 2014-2024 low-vol tech bull?
     Split the 26 years at --split (default 2014) and show every candidate's
     CAGR/maxDD in each half. The first half holds the dot-com bust and GFC;
     if a leveraged variant only works in the second half, its "expected
     return" is a regime bet, not an edge.

  2. CROSS-ASSET - does the identical machinery (same MA, same gate, same
     params) work on the S&P 500 as well as the Nasdaq? A mechanism that
     generalizes across indices is an edge; one that only fires on QQQ's
     specific history is a story fitted to one path.

  3. REAL-COST HAIRCUT - the LSE product you'd actually buy (QQQ3.L) tracked
     ~3.4%/yr below the synthetic. --etp-drag haircuts the leveraged legs by
     that much so the comparison is against the deployable instrument.

Plus the gate (vlo,vhi) parameter surface, so a smooth plateau vs a lonely
spike vs a boundary-sliding gradient is visible, not assumed.

Usage:
    python scripts/uk_robustness.py --fresh
    python scripts/uk_robustness.py --etp-drag 0.034        # honest QQQ3.L drag
    python scripts/uk_robustness.py --split 2013 --vlo 0.20 --vhi 0.28
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
from src.uk import (blended_leverage_backtest, rotation_backtest,
                    synth_leveraged_returns, tracking_report, trend_state,
                    vol_gate_leverage)

TICKERS = ["QQQ", "SPY", "GLD", "^IRX", "QLD", "TQQQ", "QQQ3.L", "SSO", "UPRO"]

# One-way cost per leg (half-spread + FX where USD-only), as in uk_max_wealth.
C3, C1, CGOLD = 0.0027, 0.0007, 0.0010


def _flag(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def _family(label, idx_close, idx_ret, rf_ann, rf_daily, gold, real_etps,
            vlo, vhi, lag, drag):
    """Build the three candidates for one index family (Nasdaq or S&P)."""
    l3 = synth_leveraged_returns(idx_ret, rf_ann, 3.0, ter=0.0075, extra_drag=drag)
    state = trend_state(idx_close, window=200, exit_band=0.99, confirm=2)
    gate = vol_gate_leverage(idx_close, state, lo=vlo, hi=vhi, lev_hi=3.0)
    cand = {
        f"{label}-bh": idx_ret,
        f"{label} 1x-tr": rotation_backtest(idx_ret, rf_daily, state, lag=lag,
                                            cost_risk=C1, cost_safe=0.0)["pnl"],
        f"{label} 3x-au": rotation_backtest(l3, gold, state, lag=lag,
                                            cost_risk=C3, cost_safe=CGOLD)["pnl"],
        f"{label} g-au": blended_leverage_backtest(idx_ret, l3, gold, gate,
                                                   lag=lag, cost_3x=C3, cost_1x=C1,
                                                   cost_safe=CGOLD)["pnl"],
    }
    # Validate the synthetic 3x against any real ETP we have for this family.
    val = []
    for lev, tk in real_etps.items():
        real = tk.pct_change() if tk is not None else None
        synth = synth_leveraged_returns(idx_ret, rf_ann, lev, extra_drag=drag)
        rep = tracking_report(synth, real) if real is not None else None
        if rep:
            val.append(f"{lev:g}x corr {rep['weekly_corr']:.3f} "
                       f"({rep['ann_diff']:+.1%}/yr)")
    return cand, "  ".join(val)


def _split_row(name, pnl, split):
    a = series_stats(pnl[pnl.index < pd.Timestamp(split)])
    b = series_stats(pnl[pnl.index >= pd.Timestamp(split)])
    full = series_stats(pnl)
    return (f"{name:<14}{a['cagr']:>8.1%}{a['maxdd']:>8.1%}   "
            f"{b['cagr']:>8.1%}{b['maxdd']:>8.1%}   "
            f"{full['cagr']:>8.1%}{full['maxdd']:>8.1%}")


def main(argv: list[str]) -> None:
    fresh = "--fresh" in argv
    split = str(_flag(argv, "--split", "2014")) + "-01-01"
    vlo = float(_flag(argv, "--vlo", 0.20))
    vhi = float(_flag(argv, "--vhi", 0.28))
    lag = int(_flag(argv, "--lag", 2))
    drag = float(_flag(argv, "--etp-drag", 0.0))

    print(f"Loading {len(TICKERS)} tickers{' (fresh)' if fresh else ''}...")
    px = load_prices(TICKERS, "1998-08-01", use_cache=not fresh)
    close = {t: px[f"{t}_close"].dropna() for t in TICKERS if f"{t}_close" in px}
    irx = close.get("^IRX", pd.Series(dtype=float))
    if len(irx) and irx.median() > 20:
        irx = irx / 10.0

    def family(label, idx_tk, real_etps):
        idx = close[idx_tk]
        ret = idx.pct_change()
        rf_ann = (irx / 100.0).reindex(ret.index).ffill().fillna(0.02)
        gold = close.get("GLD", pd.Series(dtype=float)).pct_change() \
            .reindex(ret.index).fillna(rf_ann / 252.0)
        return _family(label, idx, ret, rf_ann, rf_ann / 252.0, gold,
                       real_etps, vlo, vhi, lag, drag)

    ndq, ndq_val = family("NDX", "QQQ",
                          {2.0: close.get("QLD"), 3.0: close.get("TQQQ")})
    spx, spx_val = family("SPX", "SPY",
                          {2.0: close.get("SSO"), 3.0: close.get("UPRO")})

    print(f"\nSynthetic-ETP validation  NDX: {ndq_val}")
    print(f"                          SPX: {spx_val}")
    print(f"  (extra ETP drag applied: {drag:.1%}/yr)")

    # Common live window so both families and both halves compare like-for-like.
    start = max(pnl.dropna().index[0] for pnl in {**ndq, **spx}.values())
    end = min(pnl.dropna().index[-1] for pnl in {**ndq, **spx}.values())
    clip = lambda s: s.loc[start:end]

    print(f"\n=== REGIME SPLIT  (gate {vlo:.0%}/{vhi:.0%}, lag {lag}) "
          f"{start.date()} -> {end.date()} ===")
    print(f"{'':<14}{'pre-' + split[:4]:>16}   {split[:4] + '+':>16}   {'FULL':>16}")
    print(f"{'':<14}{'CAGR':>8}{'maxDD':>8}   {'CAGR':>8}{'maxDD':>8}   {'CAGR':>8}{'maxDD':>8}")
    print("-" * 74)
    for fam in (ndq, spx):
        for name, pnl in fam.items():
            print(_split_row(name, clip(pnl), split))
        print("-" * 74)
    print("Read: if a leveraged row is strong only in the right column, its edge")
    print("is a post-" + split[:4] + " regime bet. If NDX g-au wins but SPX g-au")
    print("doesn't, the mechanism is fitted to the Nasdaq, not general.")

    # ---- Gate parameter surface (NDX): plateau, spike, or boundary gradient? --
    idx = close["QQQ"]; ret = idx.pct_change()
    rf_ann = (irx / 100.0).reindex(ret.index).ffill().fillna(0.02)
    gold = close["GLD"].pct_change().reindex(ret.index).fillna(rf_ann / 252.0)
    l3 = synth_leveraged_returns(ret, rf_ann, 3.0, extra_drag=drag)
    state = trend_state(idx, 200, 0.99, 2)
    print(f"\n=== GATE SURFACE (NDX g-au FULL CAGR) - is {vlo:.0%}/{vhi:.0%} a "
          f"plateau or a knife-edge? ===")
    his = [0.24, 0.26, 0.28, 0.30, 0.32]
    print("  vlo\\vhi " + "".join(f"{h:>8.0%}" for h in his))
    for lo in [0.14, 0.16, 0.18, 0.20, 0.22]:
        cells = []
        for hi in his:
            if hi <= lo:
                cells.append("      - "); continue
            g = vol_gate_leverage(idx, state, lo=lo, hi=hi, lev_hi=3.0)
            pnl = clip(blended_leverage_backtest(ret, l3, gold, g, lag=lag,
                       cost_3x=C3, cost_1x=C1, cost_safe=CGOLD)["pnl"])
            cells.append(f"{series_stats(pnl)['cagr'] * 100:7.1f}%")
        print(f"   {lo:.2f}   " + "".join(cells))

    # ---- Artifact: regime-split table to CSV ---------------------------------
    rows = {}
    for fam in (ndq, spx):
        for name, pnl in fam.items():
            p = clip(pnl)
            for tag, s in (("pre", p[p.index < pd.Timestamp(split)]),
                           ("post", p[p.index >= pd.Timestamp(split)]),
                           ("full", p)):
                st = series_stats(s)
                rows[(name, tag)] = {"cagr": st["cagr"], "maxdd": st["maxdd"],
                                     "sharpe": st["sharpe"]}
    out = pd.DataFrame(rows).T
    out.index.names = ["candidate", "regime"]
    csv = ART_DIR / "uk_robustness.csv"
    out.to_csv(csv)
    print(f"\nRegime-split table -> {csv}")


if __name__ == "__main__":
    main(sys.argv)
