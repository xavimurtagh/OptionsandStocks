from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AssetConfig, IntradayHorizon, RunConfig
from .labeling import ewma_vol, triple_barrier, uniqueness_weights
from .options import options_snapshot_features


def _zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - m) / sd


def _price_features(close: pd.Series) -> pd.DataFrame:
    r1 = close.pct_change(fill_method=None)
    out = pd.DataFrame(index=close.index)
    for w in (1, 5, 10, 20, 60):
        out[f"ret_{w}d"] = close.pct_change(w, fill_method=None)
    for w in (20, 60):
        out[f"rv_{w}d"] = r1.rolling(w).std() * np.sqrt(252)
    out["vol_regime"] = out["rv_20d"] / out["rv_60d"]
    out["z_50"] = _zscore(close, 50)
    out["z_200"] = _zscore(close, 200)
    out["mom_skew"] = r1.rolling(60).skew()
    return out


def _macro_features(prices: pd.DataFrame, fred: pd.DataFrame,
                    cfg: RunConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    for label, tk in cfg.macro_tickers.items():
        col = f"{tk}_close"
        if col not in prices.columns:
            continue
        s = prices[col]
        out[f"{label}_ret_5d"] = s.pct_change(5, fill_method=None)
        out[f"{label}_ret_20d"] = s.pct_change(20, fill_method=None)
        out[f"{label}_z_60"] = _zscore(s, 60)
    if not fred.empty:
        f = fred.reindex(prices.index).ffill()
        for col in f.columns:
            if col == "gold_iv":  # handled as an option-market feature
                continue
            out[f"fred_{col}_lvl"] = f[col]
            out[f"fred_{col}_chg_5d"] = f[col].diff(5)
            out[f"fred_{col}_chg_20d"] = f[col].diff(20)
        if {"real_yield_10y", "nominal_yield_10y"}.issubset(f.columns):
            be = f["nominal_yield_10y"] - f["real_yield_10y"]
            out["breakeven_10y"] = be
            out["breakeven_10y_chg_20d"] = be.diff(20)
    return out


def _cot_features(cot: pd.DataFrame, cftc_code: str,
                  index: pd.DatetimeIndex) -> pd.DataFrame:
    if cot.empty:
        return pd.DataFrame(index=index)
    sub = cot[cot["cftc_code"] == cftc_code].copy()
    if sub.empty:
        return pd.DataFrame(index=index)
    for col in ("oi", "mm_long", "mm_short", "comm_long", "comm_short",
                "swap_long", "swap_short"):
        sub[col] = pd.to_numeric(sub.get(col), errors="coerce")
    sub["mm_net"] = (sub["mm_long"] - sub["mm_short"]) / sub["oi"]
    sub["comm_net"] = (sub["comm_long"] - sub["comm_short"]) / sub["oi"]
    sub["swap_net"] = (sub["swap_long"] - sub["swap_short"]) / sub["oi"]
    feats = sub[["mm_net", "comm_net", "swap_net"]].copy()
    feats["mm_net_z52"] = _zscore(feats["mm_net"], 52)
    feats["comm_net_z52"] = _zscore(feats["comm_net"], 52)
    feats["mm_net_chg_4w"] = feats["mm_net"].diff(4)
    daily = feats.reindex(index, method="ffill").shift(5)
    daily.columns = [f"cot_{c}" for c in daily.columns]
    return daily


def _options_features(close: pd.Series, fred: pd.DataFrame,
                      snapshots: pd.DataFrame) -> pd.DataFrame:
    """Option-market features.

    Two sources, deliberately kept separate:
      * GVZ (CBOE Gold ETF Volatility Index) - ~17 years of option-implied
        volatility, so these participate in the historical backtest. For
        silver it serves as a precious-metals implied-vol gauge.
      * Accumulated live chain snapshots - richer (skew, term structure,
        put/call flow) but forward-only, so NaN across the backtest and
        only informative for the live signal as snapshots build up.
    """
    out = pd.DataFrame(index=close.index)
    r1 = close.pct_change(fill_method=None)
    rv = r1.rolling(20).std() * np.sqrt(252)

    if fred is not None and not fred.empty and "gold_iv" in fred.columns:
        iv = fred["gold_iv"].reindex(close.index).ffill() / 100.0
        out["opt_iv"] = iv
        out["opt_iv_z_252"] = _zscore(iv, 252)
        out["opt_iv_chg_5d"] = iv.diff(5)
        out["opt_iv_chg_20d"] = iv.diff(20)
        out["opt_iv_pctile_252"] = iv.rolling(252).apply(
            lambda w: float((w[-1] >= w).mean()), raw=True)
        # Variance risk premium: implied minus trailing realized vol.
        out["opt_vrp"] = iv - rv
        out["opt_iv_rv_ratio"] = iv / rv.replace(0, np.nan)

    if snapshots is not None and not snapshots.empty:
        snap = snapshots.reindex(close.index).ffill(limit=5)
        for col in snap.columns:
            out[f"opt_{col}"] = snap[col]
    return out


def _ratio_features(prices: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    if "GLD_close" in prices.columns and "SLV_close" in prices.columns:
        ratio = prices["GLD_close"] / prices["SLV_close"]
        out["gsr"] = ratio
        out["gsr_z_252"] = _zscore(ratio, 252)
        out["gsr_chg_20d"] = ratio.pct_change(20, fill_method=None)
    return out


def vol_targets(close: pd.Series, horizons: list[int]) -> pd.DataFrame:
    """Forward realized volatility - the prediction target.

    `fwd_rv_{h}d` at row t is the annualized std of daily returns over the h
    days strictly after t (no look-ahead - trailing rows become NaN).
    `fwd_ret_{h}d` is the realized h-day forward return, kept for diagnostics
    only and never used as a training label.
    """
    r1 = close.pct_change(fill_method=None)
    out = pd.DataFrame(index=close.index)
    for h in horizons:
        out[f"fwd_rv_{h}d"] = r1.rolling(h).std().shift(-h) * np.sqrt(252)
        out[f"fwd_ret_{h}d"] = close.pct_change(h, fill_method=None).shift(-h)
    return out


def trend_signal(close: pd.Series, rv_60d: pd.Series) -> pd.Series:
    """Vol-normalized multi-horizon time-series momentum, smoothed to [-1, 1].

    Sign gives direction, magnitude gives conviction. Deterministic - no fit.
    """
    lookbacks = (21, 63, 126, 252)
    acc = pd.Series(0.0, index=close.index)
    for L in lookbacks:
        mom = close.pct_change(L, fill_method=None)
        norm = (rv_60d * np.sqrt(L / 252.0)).replace(0, np.nan)
        acc = acc + np.tanh((mom / norm) / 1.5)
    return acc / len(lookbacks)


def build_daily_features(data: dict, asset: AssetConfig,
                         cfg: RunConfig) -> pd.DataFrame:
    prices = data["prices"]
    fred = data["fred"]
    cot = data["cot"]

    close = prices[f"{asset.ticker}_close"]
    feats = _price_features(close)
    feats = feats.join(_macro_features(prices, fred, cfg))
    feats = feats.join(_ratio_features(prices))
    feats = feats.join(_cot_features(cot, asset.cftc_code, prices.index))
    feats = feats.join(_options_features(close, fred,
                                         options_snapshot_features(asset.ticker)))

    feats["trend_signal"] = trend_signal(close, feats["rv_60d"])
    feats = feats.join(vol_targets(close, cfg.daily_horizons))
    feats["close"] = close
    return feats[feats["close"].notna()].copy()


def build_intraday_features(bars: pd.DataFrame,
                            h: IntradayHorizon) -> pd.DataFrame:
    if bars.empty:
        return bars
    close = bars["close"]
    out = pd.DataFrame(index=bars.index)
    r1 = close.pct_change(fill_method=None)
    for w in (1, 2, 4, 8, 20, 40):
        out[f"ret_{w}b"] = close.pct_change(w, fill_method=None)
    for w in (20, 50):
        out[f"rv_{w}b"] = r1.rolling(w).std()
    out["vol_regime"] = out["rv_20b"] / out["rv_50b"]

    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    vol = bars["volume"].astype(float).replace(0, np.nan)
    vwap = (typical * vol).rolling(20).sum() / vol.rolling(20).sum()
    out["vwap_dev"] = (close - vwap) / vwap

    rng = (bars["high"] - bars["low"]) / close
    out["range"] = rng
    out["range_mean_20"] = rng.rolling(20).mean()

    hod = bars.index.hour + bars.index.minute / 60.0
    out["hod_sin"] = np.sin(2 * np.pi * hod / 24.0)
    out["hod_cos"] = np.cos(2 * np.pi * hod / 24.0)
    out["dow"] = bars.index.dayofweek

    tb = triple_barrier(close, ewma_vol(close, span=50),
                        horizon=h.forward_bars)
    out["target_ret"] = tb["tb_ret"]
    out["target_up"] = tb["tb_up"]
    out["weight"] = uniqueness_weights(tb["tb_t1"])
    out["close"] = close
    return out[out["close"].notna()].copy()


# Backwards-compat alias for any imports that still expect `build_features`.
def build_features(data: dict, asset: AssetConfig, cfg: RunConfig) -> pd.DataFrame:
    return build_daily_features(data, asset, cfg)
