"""UK max-wealth deployment study: what actually maximizes long-run CAGR in a
Trading212 Stocks & Shares ISA (1x cash account, UCITS funds + LSE-listed
leveraged ETPs, no margin)?

Candidates, all implementable with real LSE tickers:
  SPY-bh     buy & hold S&P 500            (deploy: CSPX / VUAG)
  QQQ-bh     buy & hold Nasdaq-100          (deploy: EQQQ)
  Q1x-tr     Nasdaq-100 above 200d MA, cash below          (EQQQ <-> cash)
  Q2x-tr     2x Nasdaq above 200d MA, cash below           (LQQ2-style ETP)
  Q3x-tr     3x Nasdaq above 200d MA, cash below           (WisdomTree QQQ3)
  Q3x-au     3x Nasdaq above 200d MA, GOLD below           (QQQ3 <-> SGLN)
  50/50      half SPY-bh, half Q3x-tr - the "half sane" blend

US tickers proxy the indices because they carry 26 years of history including
the dot-com bust - the single most important stress test for leveraged Nasdaq.
Leveraged legs are synthesized with real financing (T-bill + spread on the
borrowed notional + TER) and validated against actual ETPs (TQQQ, QQQ3.L)
over their live histories, so the pre-2010 extension is checked, not assumed.

Frictions modeled (the "other algorithms are faster than you" account):
  * 2-close execution lag by default: signal on close t, filled close t+1,
    earning the new position from t+2. --lag 1 shows what the same-close
    academic fill would have earned; the difference is the price of being last.
  * Per-switch costs: half-spread + Trading212's 0.15% FX fee on USD-quoted
    ETPs, both legs. Hysteresis (exit band + 2-day confirm) keeps switches rare.
  * --stress: 3-close lag, +25bps extra slippage per leg, financing spread
    1.5% - the "everything is worse than backtested" run.

Usage:
    python scripts/uk_max_wealth.py                 # base case, 2000->today
    python scripts/uk_max_wealth.py --stress        # pessimistic frictions
    python scripts/uk_max_wealth.py --ma 150 --band 0.985 --lag 1 --fresh
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ART_DIR, RunConfig
from src.data import load_prices
from src.research import series_stats
from src.uk import (blended_leverage_backtest, rotation_backtest,
                    synth_leveraged_returns, tracking_report, trend_state,
                    vol_gate_leverage, vol_target_leverage)

# Regimes chosen a priori. 2000-02 is the reason this script exists: any
# leveraged-Nasdaq idea that can't show you the dot-com bust is selling you
# the 2010s bull market with extra steps.
REGIMES = [
    ("2000-02 dotcom", "2000-01-01", "2002-12-31"),
    ("2003-07 bull",   "2003-01-01", "2007-12-31"),
    ("2008-09 GFC",    "2008-01-01", "2009-12-31"),
    ("2010-19 bull",   "2010-01-01", "2019-12-31"),
    ("2020 COVID",     "2020-01-01", "2020-12-31"),
    ("2021-22 bear",   "2021-01-01", "2022-12-31"),
    ("2023+ recent",   "2023-01-01", None),
    ("FULL",           None,         None),
]

# One-way cost per leg, as a fraction of traded notional: half-spread plus
# Trading212's 0.15% FX fee where the instrument only has a USD line.
# EQQQ/CSPX/SGLN have GBP lines (no FX fee); QQQ3/LQQ2 trade in USD.
COST = {"q3": 0.0012 + 0.0015, "q2": 0.0010 + 0.0015,
        "q1": 0.0007, "gold": 0.0010, "cash": 0.0}

TICKERS = ["QQQ", "SPY", "GLD", "^IRX", "TQQQ", "QLD", "QQQ3.L"]


def _flag(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def _fmt_pct(x):
    return f"{x:+7.1%}"


def main(argv: list[str]) -> None:
    start = _flag(argv, "--start", "2000-01-01")
    ma = int(_flag(argv, "--ma", 200))
    band = float(_flag(argv, "--band", 0.99))
    confirm = int(_flag(argv, "--confirm", 2))
    lag = int(_flag(argv, "--lag", 2))
    stress = "--stress" in argv
    fresh = "--fresh" in argv

    slip = 0.0
    borrow = 0.006
    if stress:
        lag, slip, borrow = max(lag, 3), 0.0025, 0.015
        print("STRESS MODE: lag=3 closes, +25bps/leg slippage, 1.5% financing spread\n")

    # Load with runway before `start` so the 200d MA is live on day one.
    warmup = (pd.Timestamp(start) - pd.Timedelta(days=500)).date().isoformat()
    print(f"Loading {len(TICKERS)} tickers from {warmup}{' (fresh)' if fresh else ''}...")
    px = load_prices(TICKERS, warmup, use_cache=not fresh)
    close = {t: px[f"{t}_close"].dropna() for t in TICKERS if f"{t}_close" in px}

    qqq, spy = close["QQQ"], close["SPY"]
    rq, rs = qqq.pct_change(), spy.pct_change()
    irx = close.get("^IRX", pd.Series(dtype=float))
    if len(irx) and irx.median() > 20:      # legacy x10 quoting
        irx = irx / 10.0
    rf_ann = (irx / 100.0).reindex(rq.index).ffill().fillna(0.02)
    rf_daily = rf_ann / 252.0

    # Synthetic leveraged legs with real financing, then check them against
    # every real ETP we can see before trusting the pre-2010 extension.
    q2 = synth_leveraged_returns(rq, rf_ann, 2.0, ter=0.0060, borrow_spread=borrow)
    q3 = synth_leveraged_returns(rq, rf_ann, 3.0, ter=0.0075, borrow_spread=borrow)
    print("\n--- Synthetic leveraged-ETP validation (weekly-return corr / CAGR synth vs real) ---")
    for label, synth, real_tk in [("2x vs QLD", q2, "QLD"),
                                  ("3x vs TQQQ", q3, "TQQQ"),
                                  ("3x vs QQQ3.L", q3, "QQQ3.L")]:
        real = close.get(real_tk, pd.Series(dtype=float)).pct_change()
        rep = tracking_report(synth, real) if len(real) else None
        if rep is None:
            print(f"  {label:<14} no overlap data (ok if Yahoo lacks the ticker)")
        else:
            print(f"  {label:<14} corr {rep['weekly_corr']:.3f}   "
                  f"CAGR {rep['cagr_synth']:+.1%} vs {rep['cagr_real']:+.1%}   "
                  f"(diff {rep['ann_diff']:+.1%}/yr over {rep['overlap_yrs']:.1f}y)")

    state = trend_state(qqq, window=ma, exit_band=band, confirm=confirm)
    gold = close.get("GLD", pd.Series(dtype=float)).pct_change() \
        .reindex(rq.index).fillna(rf_daily)        # pre-2005: cash fallback

    def rot(risk, leg, safe=None, safe_leg="cash"):
        return rotation_backtest(
            risk, rf_daily if safe is None else safe, state, lag=lag,
            cost_risk=COST[leg] + slip,
            cost_safe=COST[safe_leg] + (slip if safe_leg != "cash" else 0.0))

    bts = {"Q1x-tr": rot(rq, "q1"), "Q2x-tr": rot(q2, "q2"),
           "Q3x-tr": rot(q3, "q3"), "Q3x-au": rot(q3, "q3", gold, "gold")}

    # Vol-targeted leverage (1x-levmax): the fix for the constant-3x death
    # spiral. Scale leverage to a target vol while trend-on; gold when off.
    vtarget = float(_flag(argv, "--vtarget", 0.30))
    levmax = float(_flag(argv, "--levmax", 3.0))
    lev = vol_target_leverage(qqq, state, target_vol=vtarget, lev_max=levmax)
    bts["Qv-au"] = blended_leverage_backtest(
        rq, q3, gold, lev, lag=lag, cost_3x=COST["q3"] + slip,
        cost_1x=COST["q1"] + (slip if slip else 0.0), cost_safe=COST["gold"])

    # Binary vol gate with hysteresis: Qv's real-data audit showed continuous
    # vol scaling anti-times choppy years (0%/yr in 2003-07 vs +8-9%/yr for
    # its own legs) and churns. The gate never trades mid-swing: 3x in calm
    # uptrends, 1x in loud ones, out (gold) off-trend.
    vlo = float(_flag(argv, "--vlo", 0.20))
    vhi = float(_flag(argv, "--vhi", 0.28))
    glev = vol_gate_leverage(qqq, state, lo=vlo, hi=vhi, lev_hi=levmax)
    bts["Qg-au"] = blended_leverage_backtest(
        rq, q3, gold, glev, lag=lag, cost_3x=COST["q3"] + slip,
        cost_1x=COST["q1"] + (slip if slip else 0.0), cost_safe=COST["gold"])

    pnls = {"SPY-bh": rs, "QQQ-bh": rq}
    pnls.update({k: b["pnl"] for k, b in bts.items()})
    pnls["50/50"] = 0.5 * rs.reindex(rq.index).fillna(0) + 0.5 * bts["Q3x-tr"]["pnl"]

    # Common live window: MA warm, requested start onward.
    live = state.index[state.index >= pd.Timestamp(start)]
    live = live[live >= qqq.rolling(ma).mean().first_valid_index()]
    pnls = {k: v.reindex(live).fillna(0.0) for k, v in pnls.items()}
    names = list(pnls)
    yrs = len(live) / 252.0
    print(f"\nBacktest {live[0].date()} -> {live[-1].date()}  ({yrs:.1f} years)"
          f"   MA{ma}, exit band {band:g}, confirm {confirm}d, lag {lag} closes"
          f"   Qv: target {vtarget:.0%}, max {levmax:g}x   Qg: gate {vlo:.0%}/{vhi:.0%}")

    # ---- Regime grid: CAGR per regime per variant -------------------------
    print(f"\n{'regime':<16}" + "".join(f"{n:>9}" for n in names))
    print("-" * (16 + 9 * len(names)))
    for rname, lo, hi in REGIMES:
        cells = []
        for n in names:
            s = pnls[n]
            if lo:
                s = s[s.index >= pd.Timestamp(lo)]
            if hi:
                s = s[s.index <= pd.Timestamp(hi)]
            cells.append(_fmt_pct(series_stats(s)["cagr"]) if len(s) > 20 else "      - ")
        print(f"{rname:<16}" + " ".join(f"{c:>8}" for c in cells))

    # ---- FULL-period stats + the number that answers the actual question --
    print(f"\n{'':<8}{'Sharpe':>7}{'CAGR':>8}{'maxDD':>8}{'GBP10k ->':>12}"
          f"{'switch/yr':>11}{'cost/yr':>9}")
    print("-" * 63)
    strat_only = {}
    for n in names:
        st = series_stats(pnls[n])
        wealth = 10_000 * float((1 + pnls[n]).prod())
        sw = float(bts[n]["switch"].reindex(live).sum() / yrs) if n in bts else 0.0
        cd = float(bts[n]["cost"].reindex(live).sum() / yrs) if n in bts else 0.0
        if n not in ("SPY-bh", "QQQ-bh"):
            strat_only[n] = st
        print(f"{n:<8}{st['sharpe']:>7.2f}{st['cagr']:>8.1%}{st['maxdd']:>8.1%}"
              f"{wealth:>11,.0f} {sw:>10.1f}{cd * 1e4:>8.0f}bp")

    champ = max(strat_only, key=lambda n: strat_only[n]["logwealth"])
    dsr = series_stats(pnls[champ], n_trials=len(strat_only))["dsr"]
    print(f"\nChampion by terminal wealth: {champ}   "
          f"DSR {dsr:.2f} (deflated for {len(strat_only)} variants tried)")

    # ---- What does being slow cost? Champion CAGR at lag 1/2/3 ------------
    # A 200d trend filter is the opposite of latency-sensitive: if more lag
    # doesn't hurt (or helps), the fast-money/HFT worry doesn't apply here.
    leg = {"Q1x-tr": ("q1", rq), "Q2x-tr": ("q2", q2),
           "Q3x-tr": ("q3", q3), "Q3x-au": ("q3", q3)}.get(champ)
    cags = []
    for L in (1, 2, 3):
        if champ in ("Qv-au", "Qg-au"):
            b = blended_leverage_backtest(rq, q3, gold,
                                          lev if champ == "Qv-au" else glev,
                                          lag=L, cost_3x=COST["q3"] + slip,
                                          cost_1x=COST["q1"], cost_safe=COST["gold"])
        elif leg:
            safe = gold if champ == "Q3x-au" else rf_daily
            b = rotation_backtest(leg[1], safe, state, lag=L,
                                  cost_risk=COST[leg[0]] + slip,
                                  cost_safe=COST["gold" if champ == "Q3x-au" else "cash"])
        else:
            cags = []
            break
        cags.append(series_stats(b["pnl"].reindex(live).fillna(0))["cagr"])
    if cags:
        print(f"Lag sensitivity ({champ}):  same-close {cags[0]:+.1%}   "
              f"next-close {cags[1]:+.1%}   two-late {cags[2]:+.1%}"
              f"   (cost of being a day slower: {(cags[0] - cags[1]) * 100:.1f}pp/yr)")

    # ---- Artifacts ---------------------------------------------------------
    eq = pd.DataFrame({n: (1 + pnls[n]).cumprod() for n in names})
    dd = eq / eq.cummax() - 1.0
    csv = ART_DIR / "uk_max_wealth_equity.csv"
    eq.join(dd, rsuffix="_dd").to_csv(csv)
    print(f"\nEquity + drawdown curves -> {csv}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not installed - skipping PNG)")
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    eq.plot(ax=a1, logy=True, lw=1.2)
    a1.set_title(f"UK ISA max-wealth candidates{' [STRESS]' if stress else ''} "
                 f"- growth of GBP1, log")
    a1.grid(True, alpha=0.3)
    dd[[champ, "QQQ-bh", "SPY-bh"]].plot(ax=a2, lw=1.0)
    a2.set_title("Drawdown"); a2.grid(True, alpha=0.3)
    png = ART_DIR / "uk_max_wealth_curves.png"
    fig.tight_layout(); fig.savefig(png, dpi=120)
    print(f"Chart -> {png}")


if __name__ == "__main__":
    main(sys.argv)
