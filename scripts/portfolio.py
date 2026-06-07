"""Return-seeking cross-asset portfolio: can a diversified, dynamically
vol-targeted book beat buy-and-hold on *return*?

Builds the 16-asset book (TSMOM + cross-sectional momentum/value + optional
real-yield macro tilt), levers it to a portfolio vol target, and compares CAGR
/ Sharpe / drawdown against gold, SPY and equal-weight buy-and-hold. Then sweeps
the knobs with a Deflated Sharpe (corrected for #configs) and PBO so we don't
fool ourselves by picking the lucky leverage/target.

Usage:
    python scripts/portfolio.py                 # base config + knob sweep
    python scripts/portfolio.py --tv 0.20       # one-off target-vol override
    python scripts/portfolio.py --no-sweep
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ASSETS, RunConfig
from src.data import load_all
from src.portfolio import assemble_panel, portfolio_backtest, portfolio_sweep
from src.research import series_stats


def _flag(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def _row(label, st, extra=""):
    return (f"{label:<22}{st['sharpe']:>8.2f}{st['cagr']:>8.1%}"
            f"{st['maxdd']:>8.1%}{st['calmar']:>8.2f}{extra}")


def main(argv: list[str]) -> None:
    cfg = RunConfig()
    cfg.portfolio_target_vol = float(_flag(argv, "--tv", cfg.portfolio_target_vol))
    cfg.max_gross_leverage = float(_flag(argv, "--maxlev", cfg.max_gross_leverage))
    if "--macro" in argv:
        cfg.macro_weight = 0.3
    if "--carry" in argv:
        cfg.carry_weight = 0.3
    if "--regime" in argv:
        cfg.regime_filter = True

    full = {n: ASSETS[n] for n in cfg.universe}
    print(f"Loading {len(full)} assets...")
    data = load_all(cfg, full)
    if data["fred"].empty:
        print("[WARN] FRED unavailable - macro tilt disabled this run.")

    panel = assemble_panel(data, cfg)
    carried = [t for t in panel["tickers"] if panel["carry"][t].abs().sum() > 0]
    print(f"Carry active on {len(carried)} assets: {', '.join(carried) or 'none'}"
          + ("  [partial: bonds need ^IRX/^TNX (yfinance) or DGS2; HYG needs "
             "HY OAS]" if len(carried) < 5 else ""))
    bt = portfolio_backtest(panel, cfg)
    if bt.empty:
        print("No portfolio PnL produced."); return

    print(f"\n=== PORTFOLIO  (target_vol={cfg.portfolio_target_vol:.0%}, "
          f"max_gross={cfg.max_gross_leverage:g}x, macro_w={cfg.macro_weight:g}, "
          f"carry_w={cfg.carry_weight:g}, regime={cfg.regime_filter}) ===")
    print(f"{'strategy':<22}{'Sharpe':>8}{'CAGR':>8}{'maxDD':>8}{'Calmar':>8}")
    print("-" * 54)
    print(_row("portfolio", series_stats(bt["pnl"]),
               extra=f"   gross avg {bt['gross'].mean():.2f}x"))
    for label, col in (("equal-weight B&H", "ew_bh"), ("gold B&H", "gold_bh"),
                       ("SPY B&H", "spy_bh")):
        if col in bt.columns:
            print(_row(label, series_stats(bt[col])))

    if "--no-sweep" in argv:
        return
    print("\nKnob sweep (DSR corrected for #configs; PBO across the grid)...")
    summary, pbo = portfolio_sweep(panel, cfg)
    if summary.empty:
        print("  sweep produced nothing"); return
    gold = summary.loc["[gold-B&H]", "cagr"] if "[gold-B&H]" in summary.index else float("nan")
    print(f"\n=== SWEEP  PBO={pbo:.2f}  (n_configs={len(summary) - 3}) ===")
    print(f"{'config':<22}{'Sharpe':>8}{'CAGR':>8}{'maxDD':>8}{'Calmar':>8}{'DSR':>7}")
    print("-" * 61)
    for cname, r in summary.head(14).iterrows():
        star = " *" if (r["cagr"] > gold and r["dsr"] >= 0.95) else ""
        print(f"{cname:<22}{r['sharpe']:>8.2f}{r['cagr']:>8.1%}"
              f"{r['maxdd']:>8.1%}{r['calmar']:>8.2f}{r['dsr']:>7.2f}{star}")
    refs = [i for i in ("[EW-B&H]", "[gold-B&H]", "[SPY-B&H]") if i in summary.index]
    print("  references:")
    for cname in refs:
        r = summary.loc[cname]
        print(f"{cname:<22}{r['sharpe']:>8.2f}{r['cagr']:>8.1%}"
              f"{r['maxdd']:>8.1%}{r['calmar']:>8.2f}{r['dsr']:>7.2f}")
    body = summary.drop(index=refs, errors="ignore")
    winners = body[(body["cagr"] > gold) & (body["dsr"] >= 0.95)]
    if winners.empty:
        print("  -> no config beats gold-B&H CAGR with DSR>=0.95.")
    else:
        print(f"  -> {len(winners)} config(s) beat gold-B&H with real edge (*).")


if __name__ == "__main__":
    main(sys.argv)
