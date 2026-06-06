"""Shared pytest fixtures. Synthetic, deterministic, no network/yfinance."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def ohlcv(rng):
    """One asset's daily OHLCV as a geometric random walk."""
    n = 900
    idx = pd.bdate_range("2014-01-01", periods=n)
    ret = rng.normal(0.0003, 0.011, n)
    close = pd.Series(100 * np.exp(np.cumsum(ret)), index=idx, name="close")
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    openp = close.shift(1).bfill()
    volume = pd.Series(rng.integers(1_000_000, 5_000_000, n), index=idx)
    return pd.DataFrame({"open": openp, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)


@pytest.fixture
def wide_prices(rng):
    """Wide price frame with `{TK}_close` columns for cross-sectional signals."""
    n = 700
    idx = pd.bdate_range("2015-01-01", periods=n)
    out = {}
    for tk in ("SPY", "GLD", "SLV", "TLT", "USO"):
        ret = rng.normal(rng.normal(0.0002, 0.0003), 0.01, n)
        out[f"{tk}_close"] = pd.Series(100 * np.exp(np.cumsum(ret)), index=idx)
    return pd.DataFrame(out, index=idx)
