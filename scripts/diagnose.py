"""Skill diagnostics + attribution for the multi-asset trend portfolio.

Reads daily prediction parquets in artifacts/ and reports:
  1. Portfolio Sharpe / drawdown / cost-stress and a verdict.
  2. Per-asset table: which assets is trend timing helping vs hurting?
  3. Per-year portfolio breakdown: which years drove the result?
  4. Per-asset x per-year contribution heatmap to flag specific drags.
  5. Annualised cost drag headline.

Usage: python scripts/diagnose.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / "artifacts"
PPY = 252


def _sharpe(rets: pd.Series) -> float:
    r = rets.dropna()
    sd = r.std(ddof=0)
    return float(r.mean() / sd * np.sqrt(PPY)) if sd > 0 else 0.0


def _max_dd(rets: pd.Series) -> float:
    eq = (1 + rets.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def _r2(pred: pd.Series, actual: pd.Series) -> float:
    ss_res = float(((pred - actual) ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _read(path: Path) -> pd.DataFrame:
    """Read a per-asset parquet, drop duplicate-index rows from pre-fix walks."""
    d = pd.read_parquet(path).sort_index()
    if d.index.has_duplicates:
        d = d[~d.index.duplicated(keep="last")]
    return d


def _per_asset_table() -> pd.DataFrame:
    rows = []
    for p in sorted(ART_DIR.glob("daily_predictions_*.parquet")):
        name = p.stem.replace("daily_predictions_", "")
        if name == "portfolio":
            continue
        df = _read(p)
        if df.empty or "pnl" not in df.columns:
            continue
        v = df[["vol_fcst", "realized_rv"]].dropna() \
            if {"vol_fcst", "realized_rv"}.issubset(df.columns) else pd.DataFrame()
        nv = df[["rv_20d", "realized_rv"]].dropna() \
            if {"rv_20d", "realized_rv"}.issubset(df.columns) else pd.DataFrame()
        vol_r2 = _r2(v["vol_fcst"], v["realized_rv"]) if len(v) > 30 else float("nan")
        naive_r2 = _r2(nv["rv_20d"], nv["realized_rv"]) if len(nv) > 30 else float("nan")
        s_strat = _sharpe(df["pnl"])
        s_vt = _sharpe(df["vt_bh_pnl"]) if "vt_bh_pnl" in df.columns else float("nan")
        rows.append({
            "asset": name,
            "n": int(len(df)),
            "sharpe": s_strat,
            "vt_bh_sharpe": s_vt,
            "delta": s_strat - s_vt if not np.isnan(s_vt) else float("nan"),
            "bh_sharpe": _sharpe(df["bh_pnl"]) if "bh_pnl" in df.columns else float("nan"),
            "vol_r2": vol_r2,
            "naive_r2": naive_r2,
            "naive_better": (not np.isnan(naive_r2)) and (not np.isnan(vol_r2))
                            and naive_r2 > vol_r2,
            "max_dd": _max_dd(df["pnl"]),
        })
    return pd.DataFrame(rows)


def _verdict(sharpe: float, sharpe_2x: float, vt_sharpe: float,
             vol_r2_mean: float, naive_r2_mean: float,
             dd: float, bh_dd: float) -> str:
    beats_naive = vol_r2_mean > naive_r2_mean or np.isnan(naive_r2_mean)
    if (sharpe >= 0.7 and sharpe > vt_sharpe and sharpe_2x > 0.5
            and beats_naive and dd > bh_dd):
        return "REAL EDGE - portfolio is skilful, diversified and cost-robust"
    if sharpe >= 0.4 and beats_naive:
        return "MARGINAL - diversification helps but the bar isn't cleared"
    return "NO EDGE - portfolio fails to convincingly clear the bar"


def _portfolio_scale() -> float:
    mp = ART_DIR / "metrics_portfolio.json"
    if mp.exists():
        try:
            return float(json.loads(mp.read_text()).get("portfolio_scale", 1.0))
        except Exception:
            pass
    return 1.0


def _cost_stress(port_pnl: pd.Series, scale: float) -> tuple[list[str], float]:
    """Reconstruct the equal-risk-weighted cost stream from per-asset files."""
    frames = []
    for p in sorted(ART_DIR.glob("daily_predictions_*.parquet")):
        if p.stem.endswith("_portfolio"):
            continue
        d = _read(p)
        if "cost" in d.columns:
            frames.append(d["cost"].rename(p.stem))
    if not frames:
        return [], _sharpe(port_pnl)
    cdf = pd.concat(frames, axis=1).reindex(port_pnl.index)
    w = cdf.notna().sum(axis=1).clip(lower=1)
    port_cost = cdf.fillna(0).sum(axis=1) / w * scale
    line, sharpe_2x = [], _sharpe(port_pnl)
    for k in (1, 2, 4):
        s = _sharpe(port_pnl - (k - 1) * port_cost)
        line.append(f"{k}x={s:+.2f}")
        if k == 2:
            sharpe_2x = s
    return line, sharpe_2x


def _cost_drag(port_pnl: pd.Series, scale: float) -> tuple[float, float]:
    """Annualised cost drag + gross annualised return for context."""
    frames = []
    for p in sorted(ART_DIR.glob("daily_predictions_*.parquet")):
        if p.stem.endswith("_portfolio"):
            continue
        d = _read(p)
        if "cost" in d.columns:
            frames.append(d["cost"])
    if not frames:
        return 0.0, 0.0
    cdf = pd.concat(frames, axis=1).reindex(port_pnl.index)
    w = cdf.notna().sum(axis=1).clip(lower=1)
    port_cost = (cdf.fillna(0).sum(axis=1) / w * scale).dropna()
    years = len(port_pnl.dropna()) / PPY
    cost_drag_ann = float(port_cost.sum() / years) if years > 0 else 0.0
    gross = port_pnl.fillna(0) + port_cost.reindex(port_pnl.index).fillna(0)
    gross_ann = float(gross.sum() / years) if years > 0 else 0.0
    return cost_drag_ann, gross_ann


def per_year_breakdown(df: pd.DataFrame) -> None:
    print("\n## PER-YEAR PORTFOLIO  (strategy vs equal-wt b&h vs vol-tgt eq-wt b&h)")
    rows = []
    years = df.groupby(df.index.year)
    for yr, sub in years:
        rows.append({
            "year": yr,
            "n": len(sub),
            "strat_ret": float((1 + sub["pnl"].fillna(0)).prod() - 1),
            "strat_sharpe": _sharpe(sub["pnl"]),
            "strat_dd": _max_dd(sub["pnl"]),
            "bh_ret": float((1 + sub["bh_pnl"].fillna(0)).prod() - 1)
                      if "bh_pnl" in sub.columns else float("nan"),
            "bh_sharpe": _sharpe(sub["bh_pnl"]) if "bh_pnl" in sub.columns else float("nan"),
            "vt_bh_sharpe": _sharpe(sub["vt_bh_pnl"])
                            if "vt_bh_pnl" in sub.columns else float("nan"),
        })
    table = pd.DataFrame(rows).set_index("year")
    fmts = {"strat_ret": "{:+.1%}".format, "bh_ret": "{:+.1%}".format,
            "strat_dd": "{:.1%}".format,
            "strat_sharpe": "{:+.2f}".format, "bh_sharpe": "{:+.2f}".format,
            "vt_bh_sharpe": "{:+.2f}".format}
    print(table.to_string(formatters=fmts))


def per_asset_year_heatmap(df: pd.DataFrame) -> None:
    contrib_cols = [c for c in df.columns if c.startswith("pnl_")]
    if not contrib_cols:
        return
    print("\n## PER-ASSET x PER-YEAR CONTRIBUTION  (sum of daily PnL per cell)")
    yearly = df[contrib_cols].groupby(df.index.year).sum()
    yearly.columns = [c.replace("pnl_", "") for c in yearly.columns]
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 200):
        print(yearly.to_string(float_format=lambda v: f"{v:+.2%}"))

    flat = yearly.stack().sort_values()
    worst = flat.head(8)
    print("\n  worst asset-year cells:")
    for (yr, asset), val in worst.items():
        print(f"    {yr}  {asset:>7}  {val:+.2%}")


def diagnose_portfolio() -> None:
    path = ART_DIR / "daily_predictions_portfolio.parquet"
    if not path.exists():
        print("\nNo portfolio prediction file - run scripts/run_baseline.py first")
        return
    df = _read(path)
    if df.empty or "pnl" not in df.columns:
        print("\nportfolio file is empty")
        return

    print(f"\n## PORTFOLIO  ({len(df)} days)")
    sharpe = _sharpe(df["pnl"])
    bh_sharpe = _sharpe(df["bh_pnl"]) if "bh_pnl" in df.columns else 0.0
    vt_sharpe = _sharpe(df["vt_bh_pnl"]) if "vt_bh_pnl" in df.columns else 0.0
    dd = _max_dd(df["pnl"])
    bh_dd = _max_dd(df["bh_pnl"]) if "bh_pnl" in df.columns else 0.0
    total_ret = (1 + df["pnl"].fillna(0)).prod() - 1
    bh_ret = (1 + df["bh_pnl"].fillna(0)).prod() - 1 if "bh_pnl" in df.columns else 0.0

    print(f"  Sharpe                  : {sharpe:+.2f}   "
          f"(equal-wt b&h {bh_sharpe:+.2f}, vol-tgt eq-wt b&h {vt_sharpe:+.2f})")
    print(f"  max drawdown            : {dd:.1%}   (equal-wt b&h {bh_dd:.1%})")
    print(f"  total return            : {total_ret:+.1%}   (equal-wt b&h {bh_ret:+.1%})")
    ann = df["pnl"].dropna().std(ddof=0) * np.sqrt(PPY)
    print(f"  annualized vol          : {ann:.1%}")

    table = _per_asset_table()
    if not table.empty:
        vol_r2_mean = float(table["vol_r2"].mean(skipna=True))
        naive_r2_mean = float(table["naive_r2"].mean(skipna=True))
        print(f"  mean per-asset vol R^2  : {vol_r2_mean:+.3f}   "
              f"(naive rv_20d mean {naive_r2_mean:+.3f})")
    else:
        vol_r2_mean = naive_r2_mean = float("nan")

    scale = _portfolio_scale()
    cost_line, sharpe_2x = _cost_stress(df["pnl"], scale)
    if cost_line:
        print(f"  Sharpe vs cost          : {'  '.join(cost_line)}")
    cost_drag, gross_ann = _cost_drag(df["pnl"], scale)
    print(f"  annualised cost drag    : {cost_drag:+.2%}   "
          f"(gross annualised return {gross_ann:+.2%})")

    print(f"  VERDICT: {_verdict(sharpe, sharpe_2x, vt_sharpe, vol_r2_mean, naive_r2_mean, dd, bh_dd)}")

    if not table.empty:
        print("\n## PER-ASSET BREAKDOWN  (sorted by delta = strategy - vt_bh)")
        print("  negative delta = trend timing is destroying value on that asset")
        print("  naive_better = ML vol model is worse than rv_20d on that asset")
        view = table.sort_values("delta")[
            ["asset", "n", "sharpe", "vt_bh_sharpe", "delta", "bh_sharpe",
             "vol_r2", "naive_r2", "naive_better", "max_dd"]
        ].set_index("asset")
        fmts = {c: "{:+.3f}".format for c in
                ["sharpe", "vt_bh_sharpe", "delta", "bh_sharpe",
                 "vol_r2", "naive_r2"]}
        fmts["max_dd"] = "{:.1%}".format
        print(view.to_string(formatters=fmts))

    per_year_breakdown(df)
    per_asset_year_heatmap(df)


def main() -> None:
    if not ART_DIR.exists():
        print("No artifacts/ directory - run scripts/run_baseline.py first")
        sys.exit(1)
    print("=" * 72)
    print("MULTI-ASSET VOLATILITY-TARGETED TREND - PORTFOLIO DIAGNOSTICS")
    print("=" * 72)
    diagnose_portfolio()
    print("\n" + "=" * 72)
    print("Portfolio TSMOM's edge is risk-adjusted return and shallow drawdowns")
    print("from diversification. Use the per-asset delta column to spot names")
    print("where trend timing is the wrong thing to do, and the per-year table")
    print("to spot regime-driven underperformance.")
    print("=" * 72)


if __name__ == "__main__":
    main()
