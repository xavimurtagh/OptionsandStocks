"""2023 deep-dive: why was the strategy's worst year -1.30 Sharpe?

Loads cached per-asset predictions and rebuilds the portfolio under several
counterfactual signal regimes (TSMOM-only, XSMOM-only, value-only, no
threshold gate, no long-only clip), so we can attribute the 2023 damage
to a specific leg of the combined signal or to a specific gating choice.

Also reports per-year whipsaw counts (signal sign-flips), signal
agreement rates, and a per-asset 2023 PnL breakdown.

Run after `python scripts/run_baseline.py` has populated artifacts/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ART_DIR, ASSETS, RunConfig


def load_all() -> dict[str, pd.DataFrame]:
    out = {}
    for name in ASSETS:
        p = ART_DIR / f"daily_predictions_{name}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "combined_signal" not in df.columns:
            continue
        out[name] = df
    return out


def make_position(signal: pd.Series, vol_fcst: pd.Series,
                  cfg: RunConfig, long_only: bool, threshold: float) -> pd.Series:
    sig = signal.fillna(0).copy()
    if long_only:
        sig = sig.clip(lower=0)
    sig = sig.where(sig.abs() >= threshold, 0.0)
    ratio = (cfg.target_vol / vol_fcst).clip(0, cfg.max_leverage)
    pos = (sig * ratio).clip(-cfg.max_leverage, cfg.max_leverage)
    return pos


def asset_pnl(df: pd.DataFrame, position: pd.Series, cfg: RunConfig,
              holding: int = 5) -> pd.Series:
    book = position.rolling(holding, min_periods=1).mean()
    fwd1 = df["close"].pct_change(fill_method=None).shift(-1)
    turnover = position.diff().abs().fillna(0)
    cost = turnover * (cfg.cost_bps / 1e4)
    return book * fwd1 - cost


def portfolio_pnl(per_asset: dict[str, pd.Series], scale: float) -> pd.Series:
    pdf = pd.DataFrame(per_asset).sort_index()
    w = pdf.notna().sum(axis=1).clip(lower=1)
    return pdf.fillna(0).sum(axis=1) / w * scale


def sharpe(r: pd.Series) -> float:
    s = r.dropna()
    if len(s) < 30 or s.std() == 0:
        return float("nan")
    return (s.mean() / s.std()) * np.sqrt(252)


def yearly_sharpe(pnl: pd.Series) -> pd.Series:
    return pnl.groupby(pnl.index.year).apply(sharpe)


def yearly_return(pnl: pd.Series) -> pd.Series:
    return pnl.groupby(pnl.index.year).sum()


def main() -> None:
    cfg = RunConfig()
    data = load_all()
    if not data:
        print("no cached predictions found - run scripts/run_baseline.py first")
        return
    print(f"loaded {len(data)} assets: {sorted(data)}")

    w = cfg.signal_weights
    scenarios = {
        "A_real": lambda d: w["tsmom"] * d["trend_signal"]
                          + w["xsmom"] * d["xsmom_signal"]
                          + w["value"] * d["value_signal"],
        "B_tsmom_only": lambda d: d["trend_signal"],
        "C_xsmom_only": lambda d: d["xsmom_signal"],
        "D_value_only": lambda d: d["value_signal"],
        "E_tsmom+xsmom": lambda d: 0.625 * d["trend_signal"] + 0.375 * d["xsmom_signal"],
        "F_no_threshold": lambda d: w["tsmom"] * d["trend_signal"]
                                  + w["xsmom"] * d["xsmom_signal"]
                                  + w["value"] * d["value_signal"],
        "G_no_long_only": lambda d: w["tsmom"] * d["trend_signal"]
                                  + w["xsmom"] * d["xsmom_signal"]
                                  + w["value"] * d["value_signal"],
    }
    gating = {
        "A_real":         dict(long_only=True,  threshold=cfg.signal_threshold),
        "B_tsmom_only":   dict(long_only=True,  threshold=cfg.signal_threshold),
        "C_xsmom_only":   dict(long_only=True,  threshold=cfg.signal_threshold),
        "D_value_only":   dict(long_only=True,  threshold=cfg.signal_threshold),
        "E_tsmom+xsmom":  dict(long_only=True,  threshold=cfg.signal_threshold),
        "F_no_threshold": dict(long_only=True,  threshold=0.0),
        "G_no_long_only": dict(long_only=False, threshold=cfg.signal_threshold),
    }

    portfolios: dict[str, pd.Series] = {}
    per_asset_pnl_by_scenario: dict[str, dict[str, pd.Series]] = {}
    for sc, sig_fn in scenarios.items():
        per_asset_pnl_by_scenario[sc] = {}
        for name, df in data.items():
            sig = sig_fn(df)
            pos = make_position(sig, df["vol_fcst"], cfg, **gating[sc])
            per_asset_pnl_by_scenario[sc][name] = asset_pnl(df, pos, cfg)
        portfolios[sc] = portfolio_pnl(per_asset_pnl_by_scenario[sc], cfg.portfolio_scale)

    # --- per-year Sharpe table across scenarios ---
    print("\n" + "=" * 78)
    print("PORTFOLIO SHARPE BY YEAR x SCENARIO")
    print("=" * 78)
    yr_sharpe = pd.DataFrame({sc: yearly_sharpe(p) for sc, p in portfolios.items()})
    print(yr_sharpe.round(2).to_string())
    print("\nFULL-PERIOD SHARPE")
    print(pd.Series({sc: sharpe(p) for sc, p in portfolios.items()}).round(3).to_string())

    # --- 2023 per-asset, real scenario ---
    print("\n" + "=" * 78)
    print("2023 PER-ASSET PnL (REAL scenario) sorted ascending")
    print("=" * 78)
    pnl_2023 = {n: per_asset_pnl_by_scenario["A_real"][n].loc["2023"].sum()
                for n in data}
    print(pd.Series(pnl_2023).sort_values().mul(100).round(2)
          .to_string(header=False, name="pct_2023"))

    # --- signal whipsaw: position sign-flips per year (real scenario) ---
    print("\n" + "=" * 78)
    print("WHIPSAW - mean position SIGN-FLIPS per asset per year (real signal)")
    print("=" * 78)
    flips_by_year = {}
    for name, df in data.items():
        sig = scenarios["A_real"](df)
        pos = make_position(sig, df["vol_fcst"], cfg, **gating["A_real"])
        active = (pos.abs() > 0).astype(int)
        on_off = active.diff().abs().fillna(0)  # flat<->active transitions
        flips_by_year[name] = on_off.groupby(on_off.index.year).sum()
    flips = pd.DataFrame(flips_by_year)
    print("mean across assets:")
    print(flips.mean(axis=1).round(1).to_string())
    print("\n2023 by asset:")
    print(flips.loc[2023].sort_values(ascending=False).astype(int).to_string())

    # --- signal agreement: fraction of days all three signals same-signed ---
    print("\n" + "=" * 78)
    print("SIGNAL AGREEMENT by year (fraction of days the 3 legs agree on sign)")
    print("=" * 78)
    agree_rate = []
    for name, df in data.items():
        s = pd.DataFrame({"t": np.sign(df["trend_signal"].fillna(0)),
                          "x": np.sign(df["xsmom_signal"].fillna(0)),
                          "v": np.sign(df["value_signal"].fillna(0))})
        # all three non-zero and equal
        all_agree = ((s["t"] == s["x"]) & (s["x"] == s["v"]) & (s["t"] != 0))
        agree_rate.append(all_agree.groupby(all_agree.index.year).mean()
                          .rename(name))
    ag = pd.concat(agree_rate, axis=1)
    print("mean across assets:")
    print(ag.mean(axis=1).round(3).to_string())

    # --- 2023 hit rate (next-day direction correct on active days) ---
    print("\n" + "=" * 78)
    print("HIT RATE 2023 vs full-period (next-day direction on active days)")
    print("=" * 78)
    hit_rows = []
    for name, df in data.items():
        sig = scenarios["A_real"](df)
        pos = make_position(sig, df["vol_fcst"], cfg, **gating["A_real"])
        fwd1 = df["close"].pct_change(fill_method=None).shift(-1)
        active = pos.abs() > 0
        dir_match = (np.sign(pos) == np.sign(fwd1)).astype(float)
        hr_full = dir_match[active].mean()
        m23 = active & (pos.index.year == 2023)
        hr_23 = dir_match[m23].mean() if m23.any() else float("nan")
        hit_rows.append({"asset": name, "hit_full": hr_full,
                         "hit_2023": hr_23, "n_active_2023": int(m23.sum())})
    hr_df = pd.DataFrame(hit_rows).set_index("asset").sort_values("hit_2023")
    print(hr_df.round(3).to_string())

    # --- regime feature: portfolio realized vol & trend reversals in 2023 ---
    print("\n" + "=" * 78)
    print("REGIME SNAPSHOT - portfolio fwd1 stats by year")
    print("=" * 78)
    bh = portfolio_pnl({n: data[n]["close"].pct_change(fill_method=None).shift(-1)
                        for n in data}, scale=1.0)
    regime = pd.DataFrame({
        "bh_ann_vol": bh.groupby(bh.index.year).std() * np.sqrt(252),
        "bh_sharpe":  yearly_sharpe(bh),
        "neg_day_frac": bh.groupby(bh.index.year).apply(lambda r: (r < 0).mean()),
    })
    print(regime.round(3).to_string())


if __name__ == "__main__":
    main()
