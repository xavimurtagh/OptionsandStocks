"""Return-seeking cross-asset portfolio engine.

The lever sweep showed the per-asset vol-targeted strategy is a fine risk
manager but cannot out-*return* buy-and-hold: on one trending asset, return =
Sharpe x vol and a single asset caps the Sharpe. The only reliable way to beat
B&H on return is to raise the Sharpe (diversify across the 16-asset universe +
add real signal) and then lever the diversified book back up to a risky-asset
volatility. That is exactly what this module does, and what the static
``portfolio_scale`` in aggregate_portfolio does not: it *measures* the book's
realized volatility and dynamically levers it to ``portfolio_target_vol``.

Design choices for this first cut:
  * Signals are deterministic (TSMOM, cross-sectional momentum/value, and an
    optional real-yield macro tilt) - no per-fold model training, so the whole
    backtest is fast and fully reproducible from price data alone (FRED only
    adds the macro tilt; absent it the engine still runs).
  * Per-asset risk uses EWMA realized vol rather than the LightGBM forecast.
    The sweep showed the ML vol model's edge is in risk control, not return,
    and here the book-level vol target carries the risk budget. The ML forecast
    can be swapped in later if it earns its keep out-of-sample.

Two-layer volatility targeting:
  raw weight   w0_i,t = combined_signal_i,t / ewma_vol_i,t      (inverse-vol)
  book scaler  k_t    = clip(target_vol / trailing_book_vol_t, 0, gross cap)
  weight       w_i,t  = k_t * regime_t * w0_i,t
  pnl_t+1            = sum_i w_i,t * ret_i,t+1  -  per-asset turnover costs
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import ASSETS, RunConfig
from .costs import turnover_cost
from .features import (cross_sectional_momentum, cross_sectional_value,
                       trend_signal)
from .research import series_stats
from .validation import probability_backtest_overfitting

# Sign of each asset's return beta to real yields. Precious metals fall when
# real yields rise (beta < 0) - the most robust macro relationship in the
# universe - so we tilt only where the sign is well established rather than
# fitting a beta per asset. Extendable (e.g. long-duration bonds are also < 0).
MACRO_RY_BETA = {"GLD": -1.0, "SLV": -1.0, "GDX": -1.0, "GDXJ": -1.0}

# Assets whose carry we can source honestly from FRED. Bond/credit carry are
# textbook premia, nearly uncorrelated with momentum, and pay regardless of
# trend - exactly the diversifying return needed to lift Sharpe past a long-only
# momentum book. Commodity roll carry (USO/UNG/DBC) is the biggest carry in the
# universe but needs futures term-structure data we don't have; FX/equity carry
# need per-currency rates / dividend yields. Both are documented TODOs.
CARRY_BOND_TICKERS = ("TLT", "IEF")        # term-structure carry (10y-2y slope)
CARRY_CREDIT_TICKERS = ("HYG",)            # credit carry (high-yield OAS)
CARRY_METAL_TICKERS = ("GLD", "SLV")       # cost-of-carry (-real yield level)


def _ewma_vol(rets: pd.DataFrame, span: int) -> pd.DataFrame:
    return rets.ewm(span=span, min_periods=span // 2).std() * np.sqrt(252)


def _zscore(s: pd.Series, window: int = 252) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    return (s - mu) / sd


def carry_signal_panel(fred: pd.DataFrame | None, tickers: list[str],
                       idx: pd.Index, window: int = 504) -> pd.DataFrame:
    """Deterministic cross-asset carry in [-1, 1] per asset (date x asset).

    Bond carry  = tanh(10y-2y term spread): steep curve -> long duration.
    Credit carry= tanh(z(HY OAS)): wide spreads -> harvest the credit premium.
    Metal carry = tanh(-z(real yield level)): negative real yields cheapen the
                  cost of holding non-yielding metal -> long.
    Carry pays independent of trend, so it fires on assets momentum ignores.
    """
    carry = pd.DataFrame(0.0, index=idx, columns=tickers)
    if fred is None or fred.empty:
        return carry

    def col(name):
        return fred[name].reindex(idx).ffill() if name in fred.columns else None

    def first(names):
        for n in names:
            c = col(n)
            if c is not None:
                return c
        return None

    # Term-structure carry: 10y minus a short rate. Prefer FRED's 2y; fall back
    # to the 3m bill (the classic 10y-3m slope) sourced from Yahoo when FRED is
    # unreachable, so bond carry runs without depending on FRED.
    ten = first(["nominal_yield_10y", "nominal_yield_10y_yf"])
    short = first(["short_yield_2y", "short_yield_3m"])
    slope = np.tanh((ten - short) / 1.5) if ten is not None and short is not None else None
    for t in CARRY_BOND_TICKERS:
        if t in carry.columns and slope is not None:
            carry[t] = slope

    oas = col("hy_oas")
    if oas is not None:
        credit = np.tanh(_zscore(oas, window))
        for t in CARRY_CREDIT_TICKERS:
            if t in carry.columns:
                carry[t] = credit

    ry = col("real_yield_10y")
    if ry is not None:
        metal = np.tanh(-_zscore(ry, window))
        for t in CARRY_METAL_TICKERS:
            if t in carry.columns:
                carry[t] = metal

    return carry.fillna(0.0)


def assemble_panel(data: dict, cfg: RunConfig) -> dict:
    """Build aligned (date x asset) panels of returns and signal families."""
    prices = data["prices"]
    fred = data.get("fred")
    names = [n for n in cfg.universe if n in ASSETS]
    tickers = [ASSETS[n].ticker for n in names]

    closes = pd.DataFrame({ASSETS[n].ticker: prices[f"{ASSETS[n].ticker}_close"]
                           for n in names
                           if f"{ASSETS[n].ticker}_close" in prices.columns})
    tickers = list(closes.columns)
    rets = closes.pct_change(fill_method=None)
    rv60 = rets.rolling(60, min_periods=30).std() * np.sqrt(252)

    tsmom = pd.DataFrame({t: trend_signal(closes[t], rv60[t]) for t in tickers})
    xsmom = cross_sectional_momentum(prices, tickers, cfg.xsmom_lookback)
    value = cross_sectional_value(prices, tickers, cfg.value_lookback)

    macro = pd.DataFrame(0.0, index=closes.index, columns=tickers)
    if fred is not None and not fred.empty and "real_yield_10y" in fred.columns:
        ry = fred["real_yield_10y"].reindex(closes.index).ffill()
        ry_chg_z = _zscore(ry.diff(63))   # +ve when real yields are rising
        for t in tickers:
            beta = MACRO_RY_BETA.get(t, 0.0)
            if beta:   # e.g. metals: beta<0 -> falling yields give +signal
                macro[t] = np.tanh(beta * ry_chg_z / 1.5)

    carry = carry_signal_panel(fred, tickers, closes.index)

    # Trailing credit-stress score for the risk-off overlay: HY OAS z-score, high
    # when spreads are wide (equity-stress regime). Coincident with drawdowns and
    # trailing-only, so it can time gross without look-ahead.
    credit_z = None
    if fred is not None and not fred.empty and "hy_oas" in fred.columns:
        credit_z = _zscore(fred["hy_oas"].reindex(closes.index).ffill(), 504)

    def _al(df):
        return df.reindex(index=closes.index, columns=tickers)

    return {"close": closes, "ret": rets, "tsmom": _al(tsmom),
            "xsmom": _al(xsmom), "value": _al(value), "macro": macro,
            "carry": carry, "credit_z": credit_z, "tickers": tickers}


def combined_signal_panel(panel: dict, cfg: RunConfig) -> pd.DataFrame:
    w = cfg.signal_weights
    sig = (w.get("tsmom", 0.0) * panel["tsmom"].fillna(0)
           + w.get("xsmom", 0.0) * panel["xsmom"].fillna(0)
           + w.get("value", 0.0) * panel["value"].fillna(0)
           + cfg.macro_weight * panel["macro"].fillna(0))
    carry = panel.get("carry")
    if cfg.carry_weight and carry is not None and not carry.empty:
        sig = sig + cfg.carry_weight * carry.reindex(
            index=sig.index, columns=sig.columns).fillna(0)
    if cfg.long_only:
        sig = sig.clip(lower=0.0)
    sig = sig.where(sig.abs() >= cfg.signal_threshold, 0.0)
    return sig.clip(-1.5, 1.5)


def regime_scalar(panel: dict, cfg: RunConfig) -> pd.Series:
    """Gross-exposure multiplier in [regime_floor, 1]: full risk when SPY is
    above its 200d average, cut to the floor when below. With regime_credit on,
    also cuts gross as credit spreads (HY OAS) blow out - a coincident risk-off
    signal that catches fast crashes (e.g. Mar-2020) before the slow 200d filter,
    taking the more defensive of the two (min)."""
    idx = panel["close"].index
    if not cfg.regime_filter or "SPY" not in panel["close"].columns:
        return pd.Series(1.0, index=idx)
    spy = panel["close"]["SPY"]
    risk_on = spy > spy.rolling(200, min_periods=100).mean()
    trend = risk_on.reindex(idx).astype(float).clip(lower=cfg.regime_floor) \
        .where(risk_on.notna(), 1.0).clip(lower=cfg.regime_floor)
    cz = panel.get("credit_z")
    if not getattr(cfg, "regime_credit", False) or cz is None:
        return trend
    # Wide spreads (z>1) ramp gross down toward the floor; calm leaves it at 1.
    cz = cz.reindex(idx).ffill()
    credit = (1.0 - 0.5 * (cz - 1.0).clip(lower=0.0, upper=2.0)) \
        .clip(lower=cfg.regime_floor, upper=1.0).where(cz.notna(), 1.0)
    return pd.concat([trend, credit], axis=1).min(axis=1)


def portfolio_backtest(panel: dict, cfg: RunConfig) -> pd.DataFrame:
    """Mark-to-market the dynamically vol-targeted book plus benchmarks."""
    rets = panel["ret"]
    tickers = panel["tickers"]
    rvol = _ewma_vol(rets, cfg.vol_span).clip(lower=0.02)
    sig = combined_signal_panel(panel, cfg)

    raw = (sig / rvol).replace([np.inf, -np.inf], 0.0).fillna(0.0)  # inverse-vol
    gross_raw = raw.abs().sum(axis=1).replace(0, np.nan)
    book_raw_ret = (raw.shift(1) * rets).sum(axis=1)                 # realized
    book_vol = book_raw_ret.ewm(span=cfg.vol_span,
                                min_periods=cfg.vol_span // 2).std() * np.sqrt(252)

    k = (cfg.portfolio_target_vol / book_vol.replace(0, np.nan))
    k = k.clip(upper=cfg.max_gross_leverage / gross_raw)  # cap sum|w| at max_gross
    k = k.clip(lower=0).fillna(0.0) * regime_scalar(panel, cfg)

    weights = raw.mul(k, axis=0)
    port_ret = (weights.shift(1) * rets).sum(axis=1)

    cost = pd.Series(0.0, index=rets.index)
    for t in tickers:
        turn = weights[t].diff().abs().fillna(weights[t].abs())
        cost = cost + turnover_cost(turn, t, cfg)

    out = pd.DataFrame(index=rets.index)
    out["pnl"] = port_ret - cost
    out["gross"] = weights.abs().sum(axis=1)
    out["ew_bh"] = rets[tickers].mean(axis=1)               # equal-weight universe
    if "GLD" in rets.columns:
        out["gold_bh"] = rets["GLD"]
    if "SPY" in rets.columns:
        out["spy_bh"] = rets["SPY"]
    out = out.dropna(subset=["pnl"])
    # Start the record when the book first goes live so warm-up zeros don't
    # dilute the strategy's stats relative to the always-invested benchmarks.
    active = out["gross"] > 1e-9
    return out.loc[active.idxmax():] if active.any() else out


# --------------------------------------------------------------------------- #
# Knob sweep with multiple-testing controls (DSR corrected for #configs, PBO). #
# --------------------------------------------------------------------------- #
# long_only is fixed True (long/short has lost every sweep in this universe);
# that freed dimension now A/Bs the credit risk-off overlay (rc) so DSR/PBO judge
# it. Grid stays at 48 configs: weights(2) x tv(3) x macro(2) x carry(2) x rc(2).
DEFAULT_PORT_GRID = {
    "weights": {"xsmom": {"tsmom": 0.0, "xsmom": 1.0, "value": 0.0},
                "blend": {"tsmom": 0.3, "xsmom": 0.7, "value": 0.0}},
    "portfolio_target_vol": [0.10, 0.15, 0.20],
    "macro_weight": [0.0, 0.3],
    "carry_weight": [0.0, 0.3],
    "regime_credit": [False, True],
}


def expand_port_grid(base: RunConfig, grid: dict) -> list[tuple[str, RunConfig]]:
    out = []
    for wn, wv in grid["weights"].items():
        for tv in grid["portfolio_target_vol"]:
            for mw in grid["macro_weight"]:
                for cw in grid.get("carry_weight", [0.0]):
                    for rc in grid.get("regime_credit", [False]):
                        cfg = replace(base, signal_weights=dict(wv),
                                      long_only=True, portfolio_target_vol=tv,
                                      macro_weight=mw, carry_weight=cw,
                                      regime_filter=True, regime_credit=rc)
                        out.append(
                            (f"{wn}|tv{tv:g}|mw{mw:g}|cw{cw:g}"
                             f"|rc{int(rc)}", cfg))
    return out


def portfolio_sweep(panel: dict, base_cfg: RunConfig, grid: dict | None = None
                    ) -> tuple[pd.DataFrame, float]:
    """Score every config on the same panel; rank by CAGR with DSR/PBO."""
    grid = grid or DEFAULT_PORT_GRID
    cfgs = expand_port_grid(base_cfg, grid)
    pnls = {}
    bench = None
    for name, c in cfgs:
        bt = portfolio_backtest(panel, c)
        if bt.empty:
            continue
        pnls[name] = bt["pnl"]
        bench = bt  # benchmarks are config-independent
    if not pnls:
        return pd.DataFrame(), float("nan")

    # Common window across configs so PBO/stats compare like with like.
    matrix = pd.DataFrame(pnls).dropna(how="any")
    n_trials = max(len(pnls), 1)
    rows = {n: series_stats(matrix[n], n_trials=n_trials) for n in matrix.columns}
    for label, col in (("[EW-B&H]", "ew_bh"), ("[gold-B&H]", "gold_bh"),
                       ("[SPY-B&H]", "spy_bh")):
        if bench is not None and col in bench.columns:
            rows[label] = series_stats(bench[col].reindex(matrix.index), n_trials=1)

    summary = pd.DataFrame(rows).T.sort_values("cagr", ascending=False)
    pbo = probability_backtest_overfitting(matrix, n_splits=10)["pbo"]
    return summary, pbo
