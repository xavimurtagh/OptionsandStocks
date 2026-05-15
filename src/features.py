from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AssetConfig, IntradayHorizon, RunConfig
from .labeling import ewma_vol, triple_barrier, uniqueness_weights


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


def _ratio_features(prices: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    if "GLD_close" in prices.columns and "SLV_close" in prices.columns:
        ratio = prices["GLD_close"] / prices["SLV_close"]
        out["gsr"] = ratio
        out["gsr_z_252"] = _zscore(ratio, 252)
        out["gsr_chg_20d"] = ratio.pct_change(20, fill_method=None)
    return out


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

    vol = ewma_vol(close, span=50)
    for h in cfg.daily_horizons:
        tb = triple_barrier(close, vol, horizon=h)
        feats[f"target_ret_{h}d"] = tb["tb_ret"]
        feats[f"target_up_{h}d"] = tb["tb_up"]
        feats[f"weight_{h}d"] = uniqueness_weights(tb["tb_t1"])
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
