"""Live signal: what to hold in the ISA right now, and whether a switch is near.

The backtest is done; this is the operational tool. It pulls current prices,
computes the exact same trend filter and vol gate the backtest uses, and prints
today's target allocation across the real LSE/UCITS instruments - plus the last
fortnight of state so an imminent flip (price nearing the 200d line, vol nearing
the gate) is visible before it costs you a whipsaw.

Run it on a fixed cadence (e.g. the first trading day of each month) and trade
only the differences from what you hold. The trend/vol signals are slow, so
monthly checking is plenty - daily would just add cost (the lag study showed
being a few days late helps, not hurts).

Usage:
    python scripts/uk_signal.py --fresh                 # 50% satellite default
    python scripts/uk_signal.py --sat 0.75 --levmax 3
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_prices
from src.uk import (DEPLOY_TICKERS, target_allocation, trend_state,
                    vol_gate_leverage)

TICKERS = ["QQQ", "SPY"]


def _flag(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def _state_today(close, ma, vlo, vhi):
    """Return (trend_on, vol_calm, price, ma_level, vol_now) as of the last bar."""
    state = trend_state(close, window=ma)
    gate = vol_gate_leverage(close, state, lo=vlo, hi=vhi, lev_hi=3.0)
    vol = (close.pct_change().ewm(span=40, min_periods=20).std()
           * np.sqrt(252))
    trend_on = bool(state.iloc[-1])
    vol_calm = bool(gate.iloc[-1] == 3.0)        # gate at high leg => calm
    return (trend_on, vol_calm, float(close.iloc[-1]),
            float(close.rolling(ma).mean().iloc[-1]), float(vol.iloc[-1]), state, vol)


def main(argv: list[str]) -> None:
    fresh = "--fresh" in argv
    sat = float(_flag(argv, "--sat", 0.50))
    ma = int(_flag(argv, "--ma", 200))
    vlo = float(_flag(argv, "--vlo", 0.20))
    vhi = float(_flag(argv, "--vhi", 0.28))

    print(f"Loading {TICKERS}{' (fresh)' if fresh else ''}...")
    px = load_prices(TICKERS, "2018-01-01", use_cache=not fresh)
    qqq = px["QQQ_close"].dropna()
    spy = px["SPY_close"].dropna()
    asof = min(qqq.index[-1], spy.index[-1]).date()

    tn, cn, pn, mn, vn, st_n, vol_n = _state_today(qqq, ma, vlo, vhi)
    ts, cs, ps, ms, vs, st_s, vol_s = _state_today(spy, ma, vlo, vhi)

    print(f"\n=== UK ISA TARGET ALLOCATION  (as of {asof}, {sat:.0%} satellite) ===\n")

    def line(label, trend, calm, price, ma_lvl, vol):
        tr = "ON " if trend else "OFF"
        gp = (price / ma_lvl - 1) * 100
        gate = "CALM" if calm else "LOUD"
        lev = "3x" if (trend and calm) else ("1x" if trend else "OUT")
        print(f"  {label}: trend {tr} ({gp:+.1f}% vs 200d)   "
              f"vol {vol * 100:4.0f}% [{gate} vs {vlo:.0%}/{vhi:.0%}]   -> hold {lev}")

    line("Nasdaq", tn, cn, pn, mn, vn)
    line("S&P 500", ts, cs, ps, ms, vs)

    alloc = target_allocation(tn, cn, ts, cs, sat_weight=sat)
    names = {"EQQQ": "Nasdaq 1x (Invesco EQQQ)", "QQQ3": "Nasdaq 3x (WisdomTree)",
             "CSPX": "S&P 1x (iShares Core)", "3USL": "S&P 3x (WisdomTree)",
             "SGLN": "Gold (iShares Physical)", "CASH": "Cash / money-market"}
    print("\n  TARGET BOOK:")
    for tk, w in alloc.items():
        if w > 1e-9:
            print(f"    {w * 100:5.1f}%  {tk:<5} {names.get(tk, '')}")

    # ---- Proximity warnings: is a switch near? -------------------------------
    print("\n  WATCH (a flip changes the trade):")
    for label, trend, gp, vol, dist_ok in [
        ("Nasdaq trend", tn, (pn / mn - 1) * 100, vn, abs(pn / mn - 1) < 0.04),
        ("S&P trend", ts, (ps / ms - 1) * 100, vs, abs(ps / ms - 1) < 0.04)]:
        if dist_ok:
            print(f"    ! {label} only {gp:+.1f}% from its 200d line - a cross is close")
    for label, vol in [("Nasdaq", vn), ("S&P", vs)]:
        if vlo - 0.03 < vol < vhi + 0.03:
            print(f"    ! {label} vol {vol*100:.0f}% is near the {vlo:.0%}/{vhi:.0%} gate")
    if not any([abs(pn/mn-1) < 0.04, abs(ps/ms-1) < 0.04,
                vlo - 0.03 < vn < vhi + 0.03, vlo - 0.03 < vs < vhi + 0.03]):
        print("    (nothing close - states are firmly held)")

    # ---- Recent state trail so a human can sanity-check --------------------
    print(f"\n  Last 10 sessions (Nasdaq price vs 200d, vol, leg):")
    ma_n = qqq.rolling(ma).mean()
    for d in qqq.index[-10:]:
        on = bool(st_n.loc[d]); v = float(vol_n.loc[d])
        leg = "3x" if (on and v < vhi and v < vlo) else ("hold" if on else "OUT")
        print(f"    {d.date()}  {qqq.loc[d]:8.1f}  vs MA {ma_n.loc[d]:8.1f}"
              f"  vol {v*100:4.0f}%  trend {'ON' if on else 'OFF'}")

    print("\n  Trade only the differences from your current holdings. Re-run "
          "monthly.\n  This is a model signal, not advice; leveraged ETPs can "
          "fall to near zero in a sustained decline.")


if __name__ == "__main__":
    main(sys.argv)
