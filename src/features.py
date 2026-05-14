from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AssetConfig, RunConfig


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
        fred_d = fred.reindex(prices.index).ffill()
        for col in fred_d.columns:
            out[f"fred_{col}_lvl"] = fred_d[col]
            out[f"fred_{col}_chg_5d"] = fred_d[col].diff(5)
            out[f"fred_{col}_chg_20d"] = fred_d[col].diff(20)
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


def build_features(data: dict, asset: AssetConfig, cfg: RunConfig) -> pd.DataFrame:
    prices = data["prices"]
    fred = data["fred"]
    cot = data["cot"]

    close = prices[f"{asset.ticker}_close"]
    feats = _price_features(close)
    feats = feats.join(_macro_features(prices, fred, cfg))
    feats = feats.join(_ratio_features(prices))
    feats = feats.join(_cot_features(cot, asset.cftc_code, prices.index))

    fwd = close.shift(-cfg.horizon) / close - 1.0
    feats["target_ret"] = fwd
    feats["target_up"] = (fwd > 0).astype(int)
    feats["close"] = close
    # Keep rows whose features are populated; target may be NaN for the last
    # `horizon` rows (unknown forward return) — useful for live prediction.
    feat_cols = [c for c in feats.columns if c not in {"target_ret", "target_up", "close"}]
    return feats.dropna(subset=feat_cols).copy()
