"""Option-market features from accumulated chain snapshots.

`scripts/snapshot_options.py` stores a daily GLD/SLV option chain in
`data_cache/options_snapshots/`. This module turns those snapshots into a
daily time series of forward-looking signals (ATM implied vol, put/call
skew, term-structure slope, put/call open-interest ratio).

These are *forward-only*: they exist from the first snapshot onward, so
they inform the live signal but are absent (NaN) across historical
backtests. Long-history option-implied volatility for the backtest comes
instead from the CBOE GVZ index (see features._options_features).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR

SNAP_DIR = DATA_DIR / "options_snapshots"


def _avg(*vals: float) -> float:
    clean = [v for v in vals if v is not None and not np.isnan(v)]
    return float(np.mean(clean)) if clean else np.nan


def _atm_iv(chain: pd.DataFrame, spot: float) -> float:
    """Mean implied vol of the strikes nearest spot."""
    if chain.empty:
        return np.nan
    near = chain.iloc[(chain["strike"] - spot).abs().argsort().to_numpy()[:4]]
    iv = near["impliedVolatility"]
    iv = iv[(iv > 0.01) & (iv < 3.0)]
    return float(iv.mean()) if len(iv) else np.nan


def _otm_iv(chain: pd.DataFrame, spot: float, moneyness: float) -> float:
    """Implied vol of the strike closest to moneyness*spot."""
    if chain.empty:
        return np.nan
    row = chain.iloc[int((chain["strike"] - moneyness * spot).abs().argmin())]
    iv = float(row["impliedVolatility"])
    return iv if 0.01 < iv < 3.0 else np.nan


def _snapshot_metrics(df: pd.DataFrame, asof: pd.Timestamp) -> dict:
    df = df.copy()
    spot = float(df["spot"].iloc[0])
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["dte"] = (df["expiry"] - asof).dt.days

    expiries = sorted(d for d in df["dte"].unique() if d >= 7)
    if not expiries:
        return {}
    near = expiries[0]
    far = next((d for d in expiries if d >= near + 45), expiries[-1])

    def chain(side: str, dte: int) -> pd.DataFrame:
        return df[(df["side"] == side) & (df["dte"] == dte)]

    near_c, near_p = chain("call", near), chain("put", near)
    atm = _avg(_atm_iv(near_c, spot), _atm_iv(near_p, spot))
    far_atm = _avg(_atm_iv(chain("call", far), spot),
                   _atm_iv(chain("put", far), spot))

    # Risk reversal: out-of-the-money put IV minus OTM call IV. Positive =
    # downside protection is bid up relative to upside = crash fear.
    skew = _otm_iv(near_p, spot, 0.95) - _otm_iv(near_c, spot, 1.05)

    calls, puts = df[df["side"] == "call"], df[df["side"] == "put"]
    pc_oi = puts["openInterest"].sum() / max(calls["openInterest"].sum(), 1.0)
    pc_vol = puts["volume"].sum() / max(calls["volume"].sum(), 1.0)

    return {"atm_iv": atm, "iv_skew": skew, "iv_term_slope": far_atm - atm,
            "pc_oi_ratio": float(pc_oi), "pc_vol_ratio": float(pc_vol)}


def options_snapshot_features(ticker: str) -> pd.DataFrame:
    """Daily option-market features from accumulated chain snapshots.

    Returns an empty frame when no snapshots exist yet; callers treat the
    columns as forward-only (NaN before the first snapshot).
    """
    if not SNAP_DIR.exists():
        return pd.DataFrame()
    rows: dict[pd.Timestamp, dict] = {}
    for path in sorted(SNAP_DIR.glob(f"{ticker}_*.parquet")):
        try:
            date = pd.to_datetime(path.stem.split("_", 1)[1])
            metrics = _snapshot_metrics(pd.read_parquet(path), date)
            if metrics:
                rows[date] = metrics
        except Exception as e:
            print(f"[options] {path.name} skipped: {e}")
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    out.index.name = "date"
    return out
